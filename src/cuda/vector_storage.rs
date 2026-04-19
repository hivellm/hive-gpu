//! # CUDA Vector Storage
//!
//! Device-local storage for f32 vectors backed by a single contiguous
//! `CudaSlice`. Uploads go through `htod_copy` in batches; growth is handled
//! by allocating a larger slice and performing a device-to-device copy of
//! the live data.
//!
//! Search is implemented on the GPU via cuBLAS SGEMV (for dot products) with
//! per-metric pre/post-processing. Top-K selection happens on the CPU after a
//! single score readback.

use super::context::CudaContext;
use crate::error::{HiveGpuError, Result};
use crate::traits::GpuVectorStorage;
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::cublas::{Gemv, GemvConfig, sys as cublas_sys};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, result as cuda_result};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
const MIN_INITIAL_VECTORS: usize = 1024;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
const MIN_INITIAL_BYTES: usize = 1024 * 1024;

/// CUDA vector storage. Owned by an `Arc<CudaContext>` so cuBLAS handle and
/// device reference are shared across clones.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub struct CudaVectorStorage {
    context: Arc<CudaContext>,
    storage: CudaSlice<f32>,
    buffer_capacity: usize,
    vector_count: usize,
    dimension: usize,
    metric: GpuDistanceMetric,
    vector_id_map: HashMap<String, usize>,
    index_to_id: Vec<String>,
    removed_indices: HashSet<usize>,
    payloads: HashMap<String, HashMap<String, String>>,
    /// Precomputed squared L2 norms per stored vector (CPU-resident).
    norms_sq: Vec<f32>,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl std::fmt::Debug for CudaVectorStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaVectorStorage")
            .field("vector_count", &self.vector_count)
            .field("buffer_capacity", &self.buffer_capacity)
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("removed", &self.removed_indices.len())
            .finish()
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaVectorStorage {
    /// Allocate a new storage on the given CUDA context with a fresh device
    /// buffer sized for the smallest of `MIN_INITIAL_VECTORS` or 1 MiB worth
    /// of f32 data.
    pub fn new(
        context: Arc<CudaContext>,
        dimension: usize,
        metric: GpuDistanceMetric,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(HiveGpuError::InvalidConfiguration(
                "dimension must be > 0".to_string(),
            ));
        }

        let min_vectors_by_bytes =
            (MIN_INITIAL_BYTES / (dimension * std::mem::size_of::<f32>())).max(1);
        let capacity = MIN_INITIAL_VECTORS.max(min_vectors_by_bytes);
        let slots = capacity
            .checked_mul(dimension)
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("capacity overflow".to_string()))?;

        let storage = context
            .device()
            .alloc_zeros::<f32>(slots)
            .map_err(|e| HiveGpuError::CudaError(format!("alloc_zeros({slots}): {e:?}")))?;

        debug!(
            "cuda storage created: dim={} capacity={} bytes={}",
            dimension,
            capacity,
            slots * std::mem::size_of::<f32>()
        );

        Ok(Self {
            context,
            storage,
            buffer_capacity: capacity,
            vector_count: 0,
            dimension,
            metric,
            vector_id_map: HashMap::new(),
            index_to_id: Vec::new(),
            removed_indices: HashSet::new(),
            payloads: HashMap::new(),
            norms_sq: Vec::new(),
        })
    }

    fn validate_vector(&self, vector: &GpuVector) -> Result<()> {
        if vector.data.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.data.len(),
            });
        }
        if vector.id.is_empty() {
            return Err(HiveGpuError::InvalidConfiguration(
                "vector id must be non-empty".to_string(),
            ));
        }
        if vector.id.len() > 256 {
            return Err(HiveGpuError::InvalidConfiguration(
                "vector id must be <= 256 chars".to_string(),
            ));
        }
        if self.vector_id_map.contains_key(&vector.id) {
            return Err(HiveGpuError::InvalidConfiguration(format!(
                "duplicate vector id: {}",
                vector.id
            )));
        }
        for (i, &v) in vector.data.iter().enumerate() {
            if !v.is_finite() {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite component at index {i} in vector {}",
                    vector.id
                )));
            }
        }
        Ok(())
    }

    fn ensure_capacity(&mut self, additional: usize) -> Result<()> {
        let required = self
            .vector_count
            .checked_add(additional)
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("capacity overflow".to_string()))?;
        if required <= self.buffer_capacity {
            return Ok(());
        }

        let mut new_capacity = self.buffer_capacity;
        while new_capacity < required {
            // Metal backend's adaptive growth factor.
            let factor = if new_capacity < 1_000 {
                2.0f32
            } else if new_capacity < 10_000 {
                1.5f32
            } else {
                1.2f32
            };
            new_capacity = ((new_capacity as f32) * factor).ceil() as usize;
            new_capacity = new_capacity.max(required);
        }

        let slots = new_capacity
            .checked_mul(self.dimension)
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("slots overflow".to_string()))?;
        let mut new_buffer = self
            .context
            .device()
            .alloc_zeros::<f32>(slots)
            .map_err(|e| HiveGpuError::CudaError(format!("alloc_zeros({slots}): {e:?}")))?;

        if self.vector_count > 0 {
            let live_bytes = self.vector_count * self.dimension * std::mem::size_of::<f32>();
            // SAFETY: `new_buffer` and `self.storage` were both allocated for
            // `f32` on the same device, both pointers are still live (cudarc
            // only frees in Drop), and the byte count covers only the live
            // portion of the source.
            unsafe {
                cuda_result::memcpy_dtod_sync(
                    *new_buffer.device_ptr_mut(),
                    *self.storage.device_ptr(),
                    live_bytes,
                )
            }
            .map_err(|e| HiveGpuError::CudaError(format!("memcpy_dtod_sync: {e:?}")))?;
        }

        info!(
            "cuda storage expand: {} -> {} vectors ({:.2} MiB)",
            self.buffer_capacity,
            new_capacity,
            (slots * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)
        );

        self.storage = new_buffer;
        self.buffer_capacity = new_capacity;
        Ok(())
    }

    /// Host-side search helper: runs cuBLAS SGEMV to get raw dot products of
    /// the query against every stored vector, then applies metric-specific
    /// post-processing before selecting top-K on the CPU.
    fn gpu_scores(&self, query: &[f32]) -> Result<Vec<f32>> {
        if self.vector_count == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        for (i, &v) in query.iter().enumerate() {
            if !v.is_finite() {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite query component at index {i}"
                )));
            }
        }

        let device = self.context.device();
        let query_dev = device
            .htod_copy(query.to_vec())
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy query: {e:?}")))?;
        let mut scores_dev = device
            .alloc_zeros::<f32>(self.vector_count)
            .map_err(|e| HiveGpuError::CudaError(format!("alloc_zeros scores: {e:?}")))?;

        // Treat the flat row-major storage of shape (vector_count, dimension)
        // as a column-major matrix of shape (dimension, vector_count). Then
        // SGEMV with trans=T gives y[i] = column_i · query = v_i · query.
        let cfg = GemvConfig::<f32> {
            trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            m: self.dimension as i32,
            n: self.vector_count as i32,
            alpha: 1.0,
            lda: self.dimension as i32,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: buffers are of the correct length for the declared matrix
        // shape; all pointers come from live device allocations on the same
        // device as the cuBLAS handle.
        unsafe {
            self.context
                .blas()
                .gemv(cfg, &self.storage, &query_dev, &mut scores_dev)
        }
        .map_err(|e| HiveGpuError::CublasError(format!("sgemv: {e:?}")))?;

        let scores = device
            .dtoh_sync_copy(&scores_dev)
            .map_err(|e| HiveGpuError::CudaError(format!("dtoh_sync_copy scores: {e:?}")))?;
        Ok(scores)
    }

    /// Convert raw dot products into the per-metric score value that sorting
    /// compares. For Cosine/DotProduct higher is better; for L2 we return the
    /// squared distance so lower is better.
    fn apply_metric(&self, raw_scores: &mut [f32], query: &[f32]) {
        let query_norm_sq = dot_self(query);
        match self.metric {
            GpuDistanceMetric::DotProduct => {}
            GpuDistanceMetric::Cosine => {
                let q_norm = query_norm_sq.sqrt();
                for (i, s) in raw_scores.iter_mut().enumerate() {
                    let v_norm = self.norms_sq[i].sqrt();
                    let denom = q_norm * v_norm;
                    *s = if denom > 0.0 { *s / denom } else { 0.0 };
                }
            }
            GpuDistanceMetric::Euclidean => {
                for (i, s) in raw_scores.iter_mut().enumerate() {
                    // ||v - q||^2 = ||v||^2 - 2(v·q) + ||q||^2
                    *s = (self.norms_sq[i] - 2.0 * *s + query_norm_sq).max(0.0);
                }
            }
        }
    }

    fn select_top_k(&self, mut scored: Vec<(usize, f32)>, limit: usize) -> Vec<GpuSearchResult> {
        // Drop removed indices before sorting.
        scored.retain(|(idx, _)| !self.removed_indices.contains(idx));
        match self.metric {
            GpuDistanceMetric::Euclidean => {
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        scored.truncate(limit);
        scored
            .into_iter()
            .map(|(index, score)| {
                let id = self.index_to_id[index].clone();
                let similarity = match self.metric {
                    GpuDistanceMetric::Euclidean => 1.0 / (1.0 + score.sqrt()),
                    _ => score,
                };
                GpuSearchResult {
                    id,
                    score: similarity,
                    index,
                }
            })
            .collect()
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl GpuVectorStorage for CudaVectorStorage {
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        // Pre-validate the whole batch so we never apply a partial upload.
        // `validate_vector` rejects duplicates against existing storage;
        // this extra set catches duplicates *within* the batch.
        let mut seen = HashSet::with_capacity(vectors.len());
        for v in vectors {
            self.validate_vector(v)?;
            if !seen.insert(v.id.as_str()) {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "duplicate vector id within batch: {}",
                    v.id
                )));
            }
        }

        self.ensure_capacity(vectors.len())?;

        // Flatten the batch into a single host vector and htod_copy once.
        let mut flat = Vec::with_capacity(vectors.len() * self.dimension);
        for v in vectors {
            flat.extend_from_slice(&v.data);
        }
        let staging = self
            .context
            .device()
            .htod_copy(flat)
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy batch: {e:?}")))?;

        let bytes = vectors.len() * self.dimension * std::mem::size_of::<f32>();
        let offset_bytes = (self.vector_count * self.dimension * std::mem::size_of::<f32>()) as u64;
        // SAFETY: staging holds exactly `bytes` of f32 data; dst is inside
        // `self.storage` which we just ensured has capacity for the batch.
        unsafe {
            let dst = *self.storage.device_ptr() + offset_bytes;
            cuda_result::memcpy_dtod_sync(dst, *staging.device_ptr(), bytes)
        }
        .map_err(|e| HiveGpuError::CudaError(format!("memcpy_dtod_sync batch: {e:?}")))?;

        let mut indices = Vec::with_capacity(vectors.len());
        for v in vectors {
            let index = self.vector_count;
            self.vector_id_map.insert(v.id.clone(), index);
            self.index_to_id.push(v.id.clone());
            self.payloads.insert(v.id.clone(), v.metadata.clone());
            self.norms_sq.push(dot_self(&v.data));
            self.vector_count += 1;
            indices.push(index);
        }
        Ok(indices)
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
        if limit == 0 || self.vector_count == 0 {
            return Ok(Vec::new());
        }
        let mut scores = self.gpu_scores(query)?;
        self.apply_metric(&mut scores, query);
        let scored: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        Ok(self.select_top_k(scored, limit))
    }

    fn remove_vectors(&mut self, ids: &[String]) -> Result<()> {
        for id in ids {
            if let Some(&index) = self.vector_id_map.get(id) {
                self.removed_indices.insert(index);
                self.payloads.remove(id);
            } else {
                return Err(HiveGpuError::VectorNotFound(id.clone()));
            }
        }
        Ok(())
    }

    fn vector_count(&self) -> usize {
        self.vector_count.saturating_sub(self.removed_indices.len())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn get_vector(&self, id: &str) -> Result<Option<GpuVector>> {
        let Some(&index) = self.vector_id_map.get(id) else {
            return Ok(None);
        };
        if self.removed_indices.contains(&index) {
            return Ok(None);
        }
        // Read the vector back from device memory. This is a targeted dtoh
        // copy of a single dimension-sized slice — fine for occasional lookup
        // but should not be used in hot paths.
        let offset = index * self.dimension;
        let device = self.context.device();
        // SAFETY: slice offsets are within the live storage region; we
        // construct a temporary host Vec of exactly `dimension` f32s.
        let host_view = unsafe {
            let src = *self.storage.device_ptr() + (offset * std::mem::size_of::<f32>()) as u64;
            let mut dst = vec![0f32; self.dimension];
            cuda_result::memcpy_dtoh_sync(&mut dst, src)
                .map_err(|e| HiveGpuError::CudaError(format!("memcpy_dtoh_sync: {e:?}")))?;
            dst
        };
        let _ = device; // suppress unused warning when not debugging

        let metadata = self.payloads.get(id).cloned().unwrap_or_default();
        Ok(Some(GpuVector {
            id: id.to_string(),
            data: host_view,
            metadata,
        }))
    }

    fn clear(&mut self) -> Result<()> {
        self.vector_count = 0;
        self.buffer_capacity = self.buffer_capacity.max(MIN_INITIAL_VECTORS);
        self.vector_id_map.clear();
        self.index_to_id.clear();
        self.removed_indices.clear();
        self.payloads.clear();
        self.norms_sq.clear();
        // We intentionally keep the existing device buffer to avoid churning
        // allocations when callers clear-and-refill.
        Ok(())
    }
}

/// Sum of squares (i.e. squared L2 norm) of a host-side vector.
#[inline]
fn dot_self(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}
