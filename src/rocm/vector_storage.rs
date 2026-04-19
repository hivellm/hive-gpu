//! # ROCm Vector Storage
//!
//! Device-local f32 vector storage backed by a single contiguous HIP
//! allocation. Search is implemented via `rocblas_sgemv` with per-metric
//! post-processing, mirroring the CUDA backend (`src/cuda/vector_storage.rs`).
//!
//! ⚠️ AUTHORED BLIND — see phase3b_add-rocm-backend.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use super::context::RocmContext;
use super::ffi::{
    self, HIP_MEMCPY_DEVICE_TO_DEVICE, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE,
    HipDevicePtr_t, ROCBLAS_OP_T, hip_check, rocblas_check,
};
use crate::error::{HiveGpuError, Result};
use crate::traits::GpuVectorStorage;
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Arc;
use tracing::{debug, info};

const MIN_INITIAL_VECTORS: usize = 1024;
const MIN_INITIAL_BYTES: usize = 1024 * 1024;

/// ROCm vector storage. One HIP allocation holds every stored vector
/// concatenated row-major. Soft-deletion is tracked on the host, matching
/// the Metal and CUDA backends.
pub struct RocmVectorStorage {
    context: Arc<RocmContext>,
    storage: HipDevicePtr_t,
    storage_bytes: usize,
    buffer_capacity: usize,
    vector_count: usize,
    dimension: usize,
    metric: GpuDistanceMetric,
    vector_id_map: HashMap<String, usize>,
    index_to_id: Vec<String>,
    removed_indices: HashSet<usize>,
    payloads: HashMap<String, HashMap<String, String>>,
    /// Precomputed squared L2 norms (host-resident).
    norms_sq: Vec<f32>,
}

// SAFETY: All device-side handles are carried through rocBLAS / HIP which
// serialise on the context's stream. No raw pointer aliasing leaves the
// type.
unsafe impl Send for RocmVectorStorage {}
unsafe impl Sync for RocmVectorStorage {}

impl std::fmt::Debug for RocmVectorStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmVectorStorage")
            .field("vector_count", &self.vector_count)
            .field("buffer_capacity", &self.buffer_capacity)
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("removed", &self.removed_indices.len())
            .finish()
    }
}

impl RocmVectorStorage {
    /// Create an empty storage on the given ROCm context, backed by a
    /// freshly-allocated HIP buffer sized for either
    /// [`MIN_INITIAL_VECTORS`] entries or 1 MiB of f32 — whichever is
    /// larger.
    pub fn new(
        context: Arc<RocmContext>,
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
        let bytes = slots * std::mem::size_of::<f32>();

        let storage = hip_malloc(bytes)?;

        debug!(
            "rocm storage created: dim={} capacity={} bytes={}",
            dimension, capacity, bytes
        );

        Ok(Self {
            context,
            storage,
            storage_bytes: bytes,
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
        let bytes = slots * std::mem::size_of::<f32>();

        let new_buffer = hip_malloc(bytes)?;
        if self.vector_count > 0 {
            let live_bytes = self.vector_count * self.dimension * std::mem::size_of::<f32>();
            hip_memcpy(
                new_buffer,
                self.storage,
                live_bytes,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )?;
        }
        hip_free(self.storage)?;
        info!(
            "rocm storage expand: {} -> {} vectors ({:.2} MiB)",
            self.buffer_capacity,
            new_capacity,
            bytes as f64 / (1024.0 * 1024.0)
        );
        self.storage = new_buffer;
        self.storage_bytes = bytes;
        self.buffer_capacity = new_capacity;
        Ok(())
    }

    /// Device pointer at a given element offset (useful for per-cluster
    /// rocBLAS dispatches from the IVF module).
    pub(crate) fn device_ptr_at(&self, element_offset: usize) -> HipDevicePtr_t {
        // SAFETY: caller guarantees `element_offset * dimension` remains
        // within `storage_bytes / size_of::<f32>()`.
        unsafe { (self.storage as *mut f32).add(element_offset) as HipDevicePtr_t }
    }

    /// Borrowed view of the cached squared norms, ordered to match the
    /// internal index.
    pub(crate) fn norms_sq(&self) -> &[f32] {
        &self.norms_sq
    }

    /// Read per-vector dot products against `query` via rocBLAS SGEMV.
    /// Returns a `Vec<f32>` of length `vector_count`.
    pub(crate) fn gpu_scores(&self, query: &[f32]) -> Result<Vec<f32>> {
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

        let lib = ffi::require_hip_lib()?;
        // Upload query.
        let query_bytes = query.len() * std::mem::size_of::<f32>();
        let query_dev = hip_malloc(query_bytes)?;
        hip_memcpy_from_slice(query_dev, query)?;

        // Allocate scores buffer on device.
        let scores_bytes = self.vector_count * std::mem::size_of::<f32>();
        let scores_dev = hip_malloc(scores_bytes)?;

        // rocBLAS SGEMV — same argument layout as cuBLAS.
        //   y = alpha * A^T * x + beta * y
        // where A (n_vectors × dimension) viewed column-major is
        // (dimension × n_vectors), so OP_T gives y[i] = row_i · query.
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // SAFETY: All device pointers were just allocated; dimensions
        // match the declared matrix layout; rocBLAS shares the stream
        // bound in `RocmContext::new`.
        let status = unsafe {
            (lib.rocblas_sgemv)(
                self.context.rocblas_handle(),
                ROCBLAS_OP_T,
                self.dimension as i32,
                self.vector_count as i32,
                &alpha as *const f32,
                self.storage as *const f32,
                self.dimension as i32,
                query_dev as *const f32,
                1,
                &beta as *const f32,
                scores_dev as *mut f32,
                1,
            )
        };
        rocblas_check(status, "rocblas_sgemv search")?;

        // Ensure the kernel finishes before we read back.
        // SAFETY: `stream` was created by our context and is still live.
        let status = unsafe { (lib.hip_stream_synchronize)(self.context.stream()) };
        hip_check(status, "hipStreamSynchronize")?;

        // Copy scores back to host.
        let mut out = vec![0f32; self.vector_count];
        hip_memcpy_to_slice(out.as_mut_slice(), scores_dev)?;

        // Free the transient buffers.
        let _ = hip_free(query_dev);
        let _ = hip_free(scores_dev);

        Ok(out)
    }

    fn apply_metric(&self, raw_scores: &mut [f32], query: &[f32]) {
        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        match self.metric {
            GpuDistanceMetric::DotProduct => {}
            GpuDistanceMetric::Cosine => {
                let q_norm = query_norm_sq.sqrt();
                for (i, s) in raw_scores.iter_mut().enumerate() {
                    let v_norm = self.norms_sq[i].sqrt();
                    let denom = v_norm * q_norm;
                    *s = if denom > 0.0 { *s / denom } else { 0.0 };
                }
            }
            GpuDistanceMetric::Euclidean => {
                for (i, s) in raw_scores.iter_mut().enumerate() {
                    *s = (self.norms_sq[i] - 2.0 * *s + query_norm_sq).max(0.0);
                }
            }
        }
    }

    fn select_top_k(&self, mut scored: Vec<(usize, f32)>, limit: usize) -> Vec<GpuSearchResult> {
        scored.retain(|(idx, _)| !self.removed_indices.contains(idx));
        match self.metric {
            GpuDistanceMetric::Euclidean => {
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            }
            _ => scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
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

impl Drop for RocmVectorStorage {
    fn drop(&mut self) {
        if !self.storage.is_null() {
            let _ = hip_free(self.storage);
            self.storage = std::ptr::null_mut();
        }
    }
}

impl GpuVectorStorage for RocmVectorStorage {
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
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

        // Flatten and upload in a single memcpy.
        let mut flat = Vec::with_capacity(vectors.len() * self.dimension);
        for v in vectors {
            flat.extend_from_slice(&v.data);
        }
        let offset_bytes = self.vector_count * self.dimension * std::mem::size_of::<f32>();
        // SAFETY: `self.storage` is a valid device pointer sized for at
        // least `storage_bytes`; the destination slot starts inside the
        // buffer thanks to `ensure_capacity`.
        let dst = unsafe { (self.storage as *mut u8).add(offset_bytes) as HipDevicePtr_t };
        hip_memcpy_from_slice(dst, &flat)?;

        let mut indices = Vec::with_capacity(vectors.len());
        for v in vectors {
            let index = self.vector_count;
            self.vector_id_map.insert(v.id.clone(), index);
            self.index_to_id.push(v.id.clone());
            self.payloads.insert(v.id.clone(), v.metadata.clone());
            self.norms_sq
                .push(v.data.iter().map(|&x| x * x).sum::<f32>());
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
        let offset_bytes = index * self.dimension * std::mem::size_of::<f32>();
        // SAFETY: offsets bounded by `self.vector_count * dimension`.
        let src = unsafe { (self.storage as *const u8).add(offset_bytes) as HipDevicePtr_t };
        let mut host = vec![0f32; self.dimension];
        hip_memcpy_to_slice(host.as_mut_slice(), src)?;

        let metadata = self.payloads.get(id).cloned().unwrap_or_default();
        Ok(Some(GpuVector {
            id: id.to_string(),
            data: host,
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
        Ok(())
    }
}

// ---------- HIP memory helpers (pub(crate) so IVF can reuse) ----------

pub(crate) fn hip_malloc(bytes: usize) -> Result<HipDevicePtr_t> {
    let lib = ffi::require_hip_lib()?;
    let mut ptr: HipDevicePtr_t = std::ptr::null_mut();
    // SAFETY: `hip_malloc` writes the new device pointer via the
    // out-parameter on success; a non-zero status leaves `ptr` null.
    let status = unsafe { (lib.hip_malloc)(&mut ptr, bytes) };
    hip_check(status, "hipMalloc")?;
    Ok(ptr)
}

pub(crate) fn hip_free(ptr: HipDevicePtr_t) -> Result<()> {
    if ptr.is_null() {
        return Ok(());
    }
    let lib = ffi::require_hip_lib()?;
    // SAFETY: `ptr` was produced by our `hip_malloc`; freeing twice is
    // guarded by the caller setting it to null after this returns.
    let status = unsafe { (lib.hip_free)(ptr) };
    hip_check(status, "hipFree")
}

pub(crate) fn hip_memcpy(
    dst: HipDevicePtr_t,
    src: HipDevicePtr_t,
    bytes: usize,
    kind: i32,
) -> Result<()> {
    let lib = ffi::require_hip_lib()?;
    // SAFETY: Caller guarantees `src` / `dst` are valid for `bytes`
    // bytes on their respective sides per `kind`.
    let status = unsafe { (lib.hip_memcpy)(dst, src as *const c_void, bytes, kind) };
    hip_check(status, "hipMemcpy")
}

pub(crate) fn hip_memcpy_from_slice(dst: HipDevicePtr_t, src: &[f32]) -> Result<()> {
    let bytes = src.len() * std::mem::size_of::<f32>();
    let lib = ffi::require_hip_lib()?;
    let status = unsafe {
        (lib.hip_memcpy)(
            dst,
            src.as_ptr() as *const c_void,
            bytes,
            HIP_MEMCPY_HOST_TO_DEVICE,
        )
    };
    hip_check(status, "hipMemcpy HtoD")
}

pub(crate) fn hip_memcpy_to_slice(dst: &mut [f32], src: HipDevicePtr_t) -> Result<()> {
    let bytes = dst.len() * std::mem::size_of::<f32>();
    let lib = ffi::require_hip_lib()?;
    let status = unsafe {
        (lib.hip_memcpy)(
            dst.as_mut_ptr() as HipDevicePtr_t,
            src as *const c_void,
            bytes,
            HIP_MEMCPY_DEVICE_TO_HOST,
        )
    };
    hip_check(status, "hipMemcpy DtoH")
}
