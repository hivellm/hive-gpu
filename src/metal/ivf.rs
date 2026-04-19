//! # Metal IVF Index
//!
//! Mirror of [`crate::cuda::ivf::CudaIvfIndex`] on the Metal backend. Uses
//! the `sgemv_dot` and `sgemm_dot` compute kernels added to the existing
//! HNSW shader library for all GPU-side dot-product work; argmin and top-K
//! run on the host, matching the CUDA scope of v1.
//!
//! ⚠️ AUTHORED WITHOUT MAC HARDWARE — the `cfg(target_os = "macos")` gate
//! means this module is not compiled on the Windows host where it was
//! written. A Mac maintainer must run
//! `cargo test --features metal-native --test metal_ivf` before the
//! phase4c_add-metal-ivf-index task is archived.

#![cfg(all(target_os = "macos", feature = "metal-native"))]

use super::context::MetalNativeContext;
use super::vector_storage::run_sgemv_dot;
use crate::error::{HiveGpuError, Result};
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector, IvfConfig};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use std::sync::Arc;
use tracing::{debug, info};

/// IVF index on Metal. API mirrors `CudaIvfIndex` field-for-field to keep
/// downstream code portable between backends.
pub struct MetalIvfIndex {
    context: Arc<MetalNativeContext>,
    dimension: usize,
    metric: GpuDistanceMetric,
    config: IvfConfig,

    /// Centroids stored row-major as a flat `(n_list, dimension)` buffer.
    centroids_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Squared L2 norms of every centroid (host side, length `n_list`).
    centroid_norms_sq: Vec<f32>,

    /// Flat vector storage reordered so each cluster's members are
    /// contiguous. Shape `(total_vectors, dimension)` row-major.
    vectors_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Per-vector squared L2 norms in the reordered order.
    vector_norms_sq: Vec<f32>,
    /// Cluster start offsets (length `n_list + 1`).
    cluster_offsets: Vec<usize>,
    /// Original IDs of stored vectors in reordered order.
    ids_by_local_index: Vec<String>,
    vector_count: usize,
    trained: bool,
}

impl std::fmt::Debug for MetalIvfIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalIvfIndex")
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("n_list", &self.config.n_list)
            .field("nprobe", &self.config.nprobe)
            .field("vector_count", &self.vector_count)
            .field("trained", &self.trained)
            .finish()
    }
}

impl MetalIvfIndex {
    pub fn new(
        context: Arc<MetalNativeContext>,
        dimension: usize,
        metric: GpuDistanceMetric,
        config: IvfConfig,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(HiveGpuError::InvalidConfiguration(
                "dimension must be > 0".to_string(),
            ));
        }
        if config.n_list == 0 {
            return Err(HiveGpuError::InvalidConfiguration(
                "n_list must be > 0".to_string(),
            ));
        }
        if config.nprobe == 0 || config.nprobe > config.n_list {
            return Err(HiveGpuError::InvalidConfiguration(format!(
                "nprobe must be in 1..={}",
                config.n_list
            )));
        }
        Ok(Self {
            context,
            dimension,
            metric,
            config,
            centroids_buffer: None,
            centroid_norms_sq: Vec::new(),
            vectors_buffer: None,
            vector_norms_sq: Vec::new(),
            cluster_offsets: Vec::new(),
            ids_by_local_index: Vec::new(),
            vector_count: 0,
            trained: false,
        })
    }

    pub fn set_nprobe(&mut self, nprobe: usize) -> Result<()> {
        if nprobe == 0 || nprobe > self.config.n_list {
            return Err(HiveGpuError::InvalidConfiguration(format!(
                "nprobe must be in 1..={}",
                self.config.n_list
            )));
        }
        self.config.nprobe = nprobe;
        Ok(())
    }

    pub fn nprobe(&self) -> usize {
        self.config.nprobe
    }
    pub fn n_list(&self) -> usize {
        self.config.n_list
    }
    pub fn vector_count(&self) -> usize {
        self.vector_count
    }
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Train k-means on a sample and populate the index from every input
    /// vector in a single pass. Mirrors `CudaIvfIndex::build`.
    pub fn build(&mut self, vectors: &[GpuVector]) -> Result<()> {
        if vectors.is_empty() {
            return Err(HiveGpuError::InvalidConfiguration(
                "cannot build IVF from empty vector set".to_string(),
            ));
        }
        if vectors.len() < self.config.n_list {
            return Err(HiveGpuError::InvalidConfiguration(format!(
                "need at least n_list={} vectors to train, got {}",
                self.config.n_list,
                vectors.len()
            )));
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.data.len() != self.dimension {
                return Err(HiveGpuError::DimensionMismatch {
                    expected: self.dimension,
                    actual: v.data.len(),
                });
            }
            if v.data.iter().any(|x| !x.is_finite()) {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite component in input vector #{i} (id={})",
                    v.id
                )));
            }
        }

        let sample_size = self.config.training_sample_size.min(vectors.len());
        let flat_sample: Vec<f32> = vectors
            .iter()
            .take(sample_size)
            .flat_map(|v| v.data.iter().copied())
            .collect();

        info!(
            "metal ivf build: dim={} n={} n_list={} training_sample={}",
            self.dimension,
            vectors.len(),
            self.config.n_list,
            sample_size
        );

        let centroids_flat =
            self.train_kmeans(&flat_sample, sample_size, self.config.kmeans_iters)?;
        debug_assert_eq!(centroids_flat.len(), self.config.n_list * self.dimension);

        let flat_all: Vec<f32> = vectors
            .iter()
            .flat_map(|v| v.data.iter().copied())
            .collect();
        let assignments = self.assign_to_centroids(&flat_all, vectors.len(), &centroids_flat)?;
        debug_assert_eq!(assignments.len(), vectors.len());

        let (offsets, perm) = build_cluster_layout(&assignments, self.config.n_list);

        let mut reordered = vec![0f32; flat_all.len()];
        let mut reordered_ids = Vec::with_capacity(vectors.len());
        let mut reordered_norms = Vec::with_capacity(vectors.len());
        for (local_idx, &global_idx) in perm.iter().enumerate() {
            let src = global_idx * self.dimension;
            let dst = local_idx * self.dimension;
            reordered[dst..dst + self.dimension]
                .copy_from_slice(&flat_all[src..src + self.dimension]);
            reordered_ids.push(vectors[global_idx].id.clone());
            reordered_norms.push(dot_self(&flat_all[src..src + self.dimension]));
        }

        // Upload centroids + reordered vectors to private MTLBuffers via a
        // shared staging buffer and a blit copy — same pattern used by
        // `MetalNativeVectorStorage::add_vector`.
        let centroids_buf = upload_private_buffer(&self.context, &centroids_flat)?;
        let vectors_buf = upload_private_buffer(&self.context, &reordered)?;

        let mut centroid_norms_sq = Vec::with_capacity(self.config.n_list);
        for i in 0..self.config.n_list {
            let start = i * self.dimension;
            centroid_norms_sq.push(dot_self(&centroids_flat[start..start + self.dimension]));
        }

        self.centroids_buffer = Some(centroids_buf);
        self.centroid_norms_sq = centroid_norms_sq;
        self.vectors_buffer = Some(vectors_buf);
        self.vector_norms_sq = reordered_norms;
        self.cluster_offsets = offsets;
        self.ids_by_local_index = reordered_ids;
        self.vector_count = vectors.len();
        self.trained = true;

        info!(
            "metal ivf build done: {} vectors across {} clusters",
            self.vector_count, self.config.n_list
        );
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<GpuSearchResult>> {
        if !self.trained {
            return Err(HiveGpuError::InvalidConfiguration(
                "IVF index must be built before search".to_string(),
            ));
        }
        if query.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        if k == 0 || self.vector_count == 0 {
            return Ok(Vec::new());
        }
        for (i, &x) in query.iter().enumerate() {
            if !x.is_finite() {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite query component at index {i}"
                )));
            }
        }

        let centroids_buf = self
            .centroids_buffer
            .as_ref()
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;
        let vectors_buf = self
            .vectors_buffer
            .as_ref()
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;

        // 1. Coarse: query · centroids via sgemv_dot over the full
        //    centroid matrix.
        let coarse_dots = run_sgemv_dot(
            &self.context,
            centroids_buf,
            /* element_offset = */ 0,
            self.config.n_list,
            self.dimension,
            query,
        )?;
        let query_norm_sq = dot_self(query);
        let probe = select_nprobe_clusters(
            &coarse_dots,
            &self.centroid_norms_sq,
            query_norm_sq,
            self.config.nprobe,
        );

        // 2. Refined: per-probed-cluster sgemv_dot over a contiguous
        //    subrange, with metric post-processing and soft-delete masking.
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for cluster_id in &probe {
            let start = self.cluster_offsets[*cluster_id];
            let end = self.cluster_offsets[cluster_id + 1];
            let count = end - start;
            if count == 0 {
                continue;
            }
            let offset_elements = start * self.dimension;
            let scores = run_sgemv_dot(
                &self.context,
                vectors_buf,
                offset_elements,
                count,
                self.dimension,
                query,
            )?;
            for (j, dot) in scores.into_iter().enumerate() {
                let local_idx = start + j;
                let m = self.score_from_dot(dot, local_idx, query_norm_sq);
                candidates.push((local_idx, m));
            }
        }

        Ok(self.finalize_top_k(candidates, k))
    }

    // --- internals ------------------------------------------------------

    /// Dispatch `sgemm_dot` to produce `(n_samples, n_list)` dot products,
    /// then derive assignments on the host via L2-adjusted argmax.
    fn assign_to_centroids(
        &self,
        flat_samples: &[f32],
        n_samples: usize,
        centroids_flat: &[f32],
    ) -> Result<Vec<u32>> {
        let dots = dispatch_sgemm_dot(
            &self.context,
            flat_samples,
            centroids_flat,
            self.dimension,
            self.config.n_list,
            n_samples,
        )?;

        // Precompute centroid squared norms. argmin ||s - c||^2 =
        // argmax (2 s·c - ||c||^2) since ||s||^2 is constant per row.
        let mut centroid_norms_sq = Vec::with_capacity(self.config.n_list);
        for j in 0..self.config.n_list {
            let start = j * self.dimension;
            centroid_norms_sq.push(dot_self(&centroids_flat[start..start + self.dimension]));
        }

        let mut assignments = vec![0u32; n_samples];
        for i in 0..n_samples {
            let row = &dots[i * self.config.n_list..(i + 1) * self.config.n_list];
            let (best_j, _) = row
                .iter()
                .enumerate()
                .map(|(j, &dot)| (j, 2.0 * dot - centroid_norms_sq[j]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("n_list > 0");
            assignments[i] = best_j as u32;
        }
        Ok(assignments)
    }

    fn train_kmeans(
        &self,
        flat_sample: &[f32],
        n_samples: usize,
        n_iter: usize,
    ) -> Result<Vec<f32>> {
        let mut centroids = kmeans_plus_plus_init(
            flat_sample,
            n_samples,
            self.dimension,
            self.config.n_list,
            self.config.seed,
        );
        let mut prev_inertia = f64::INFINITY;
        for iter in 0..n_iter {
            let assignments = self.assign_to_centroids(flat_sample, n_samples, &centroids)?;
            let (new_centroids, inertia) = update_centroids(
                flat_sample,
                n_samples,
                &assignments,
                &centroids,
                self.dimension,
                self.config.n_list,
            );
            centroids = new_centroids;
            debug!("kmeans iter {iter}: inertia={inertia:.6}");
            if (prev_inertia - inertia).abs() <= 1e-6 * prev_inertia.abs().max(1.0) {
                debug!("kmeans converged after {} iters", iter + 1);
                break;
            }
            prev_inertia = inertia;
        }
        Ok(centroids)
    }

    fn score_from_dot(&self, dot: f32, local_idx: usize, query_norm_sq: f32) -> f32 {
        match self.metric {
            GpuDistanceMetric::DotProduct => dot,
            GpuDistanceMetric::Cosine => {
                let v_norm = self.vector_norms_sq[local_idx].sqrt();
                let q_norm = query_norm_sq.sqrt();
                let denom = v_norm * q_norm;
                if denom > 0.0 { dot / denom } else { 0.0 }
            }
            GpuDistanceMetric::Euclidean => {
                (self.vector_norms_sq[local_idx] - 2.0 * dot + query_norm_sq).max(0.0)
            }
        }
    }

    fn finalize_top_k(&self, mut scored: Vec<(usize, f32)>, k: usize) -> Vec<GpuSearchResult> {
        match self.metric {
            GpuDistanceMetric::Euclidean => {
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            }
            _ => scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
        }
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(index, score)| GpuSearchResult {
                id: self.ids_by_local_index[index].clone(),
                score: match self.metric {
                    GpuDistanceMetric::Euclidean => 1.0 / (1.0 + score.sqrt()),
                    _ => score,
                },
                index,
            })
            .collect()
    }
}

// --- shared helpers ----------------------------------------------------

#[inline]
fn dot_self(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn select_nprobe_clusters(
    dots: &[f32],
    centroid_norms_sq: &[f32],
    _query_norm_sq: f32,
    nprobe: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = dots
        .iter()
        .enumerate()
        .map(|(i, &dot)| (i, centroid_norms_sq[i] - 2.0 * dot))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(nprobe);
    scored.into_iter().map(|(i, _)| i).collect()
}

fn build_cluster_layout(assignments: &[u32], n_list: usize) -> (Vec<usize>, Vec<usize>) {
    let mut counts = vec![0usize; n_list];
    for &a in assignments {
        counts[a as usize] += 1;
    }
    let mut offsets = Vec::with_capacity(n_list + 1);
    offsets.push(0);
    for c in &counts {
        offsets.push(*offsets.last().unwrap() + c);
    }
    let mut perm = vec![0usize; assignments.len()];
    let mut cursors = offsets.clone();
    for (global_idx, &a) in assignments.iter().enumerate() {
        let pos = cursors[a as usize];
        perm[pos] = global_idx;
        cursors[a as usize] += 1;
    }
    (offsets, perm)
}

fn kmeans_plus_plus_init(
    flat_sample: &[f32],
    n_samples: usize,
    dimension: usize,
    n_list: usize,
    seed: Option<u64>,
) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed.unwrap_or(0x9E37_79B9_7F4A_7C15));
    let mut centroids = Vec::with_capacity(n_list * dimension);
    let first = (rng.next_u64() as usize) % n_samples;
    centroids.extend_from_slice(&flat_sample[first * dimension..(first + 1) * dimension]);
    let mut min_dist_sq = vec![f32::INFINITY; n_samples];
    for c in 0..n_list - 1 {
        let last = &centroids[c * dimension..(c + 1) * dimension];
        for i in 0..n_samples {
            let d = l2_sq(&flat_sample[i * dimension..(i + 1) * dimension], last);
            if d < min_dist_sq[i] {
                min_dist_sq[i] = d;
            }
        }
        let total: f64 = min_dist_sq.iter().map(|&x| x as f64).sum();
        if total <= 0.0 {
            let pick = (rng.next_u64() as usize) % n_samples;
            centroids.extend_from_slice(&flat_sample[pick * dimension..(pick + 1) * dimension]);
            continue;
        }
        let target = (rng.next_f64() * total) as f32;
        let mut acc = 0f32;
        let mut pick = n_samples - 1;
        for (i, &d) in min_dist_sq.iter().enumerate() {
            acc += d;
            if acc >= target {
                pick = i;
                break;
            }
        }
        centroids.extend_from_slice(&flat_sample[pick * dimension..(pick + 1) * dimension]);
    }
    centroids
}

fn update_centroids(
    flat_sample: &[f32],
    n_samples: usize,
    assignments: &[u32],
    centroids: &[f32],
    dimension: usize,
    n_list: usize,
) -> (Vec<f32>, f64) {
    let mut sums = vec![0f32; n_list * dimension];
    let mut counts = vec![0usize; n_list];
    for (i, &assigned) in assignments.iter().enumerate().take(n_samples) {
        let c = assigned as usize;
        counts[c] += 1;
        let base = c * dimension;
        let sbase = i * dimension;
        for d in 0..dimension {
            sums[base + d] += flat_sample[sbase + d];
        }
    }
    let mut new_centroids = centroids.to_vec();
    for j in 0..n_list {
        if counts[j] == 0 {
            let mut worst = 0usize;
            let mut worst_d = -1f32;
            for i in 0..n_samples {
                let a = assignments[i] as usize;
                let c = &centroids[a * dimension..(a + 1) * dimension];
                let d = l2_sq(&flat_sample[i * dimension..(i + 1) * dimension], c);
                if d > worst_d {
                    worst_d = d;
                    worst = i;
                }
            }
            new_centroids[j * dimension..(j + 1) * dimension]
                .copy_from_slice(&flat_sample[worst * dimension..(worst + 1) * dimension]);
            continue;
        }
        let inv = 1.0 / counts[j] as f32;
        for d in 0..dimension {
            new_centroids[j * dimension + d] = sums[j * dimension + d] * inv;
        }
    }
    let mut inertia = 0f64;
    for i in 0..n_samples {
        let j = assignments[i] as usize;
        let d = l2_sq(
            &flat_sample[i * dimension..(i + 1) * dimension],
            &new_centroids[j * dimension..(j + 1) * dimension],
        );
        inertia += d as f64;
    }
    (new_centroids, inertia)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Upload a flat f32 slice into a `StorageModePrivate` Metal buffer via a
/// transient shared staging buffer and a blit copy. Returns the owned
/// device buffer.
fn upload_private_buffer(
    context: &MetalNativeContext,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let device = context.device();
    let queue = context.command_queue();
    let bytes = data.len() * std::mem::size_of::<f32>();

    let dst = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
        .ok_or_else(|| HiveGpuError::Other("failed to create private buffer".to_string()))?;
    let staging = unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut std::ffi::c_void),
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| HiveGpuError::Other("failed to create staging buffer".to_string()))?
    };
    let cmd_buf = queue
        .commandBuffer()
        .ok_or_else(|| HiveGpuError::Other("failed to create command buffer".to_string()))?;
    let blit = cmd_buf
        .blitCommandEncoder()
        .ok_or_else(|| HiveGpuError::Other("failed to create blit encoder".to_string()))?;
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            &staging, 0, &dst, 0, bytes,
        );
    }
    blit.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
    Ok(dst)
}

/// Dispatch the `sgemm_dot` compute kernel producing row-major output
/// `(n_samples, n_list)` of dot products between every sample and every
/// centroid. Returns the host-readable result.
fn dispatch_sgemm_dot(
    context: &MetalNativeContext,
    samples: &[f32],
    centroids: &[f32],
    dimension: usize,
    n_list: usize,
    n_samples: usize,
) -> Result<Vec<f32>> {
    if n_samples == 0 || n_list == 0 {
        return Ok(Vec::new());
    }
    let device = context.device();
    let queue = context.command_queue();
    let pipeline = context.compute_pipeline("sgemm_dot")?;

    let samples_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new_unchecked(samples.as_ptr() as *mut std::ffi::c_void),
                samples.len() * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| HiveGpuError::Other("failed to create samples buffer".to_string()))?
    };
    let centroids_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new_unchecked(centroids.as_ptr() as *mut std::ffi::c_void),
                centroids.len() * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| HiveGpuError::Other("failed to create centroids buffer".to_string()))?
    };
    let out_len = n_samples * n_list;
    let out_buf = device
        .newBufferWithLength_options(
            out_len * std::mem::size_of::<f32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| HiveGpuError::Other("failed to create output buffer".to_string()))?;

    let cmd = queue
        .commandBuffer()
        .ok_or_else(|| HiveGpuError::Other("failed to create command buffer".to_string()))?;
    let enc = cmd
        .computeCommandEncoder()
        .ok_or_else(|| HiveGpuError::Other("failed to create compute encoder".to_string()))?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&samples_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&centroids_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 2);
    }
    let dim_u32 = dimension as u32;
    let n_list_u32 = n_list as u32;
    let n_samples_u32 = n_samples as u32;
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dim_u32 as *const u32 as *mut std::ffi::c_void),
            std::mem::size_of::<u32>(),
            3,
        );
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&n_list_u32 as *const u32 as *mut std::ffi::c_void),
            std::mem::size_of::<u32>(),
            4,
        );
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&n_samples_u32 as *const u32 as *mut std::ffi::c_void),
            std::mem::size_of::<u32>(),
            5,
        );
    }

    // 2D dispatch — width = n_samples, height = n_list. Threadgroup size
    // chosen as 16 × 16 which fits every Apple GPU family.
    let tgs = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };
    let grid = MTLSize {
        width: n_samples,
        height: n_list,
        depth: 1,
    };
    unsafe {
        enc.dispatchThreads_threadsPerThreadgroup(grid, tgs);
    }
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let mut out = vec![0f32; out_len];
    // SAFETY: out_buf allocated with `out_len * 4` bytes host-visible.
    unsafe {
        let src = out_buf.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out_len);
    }
    Ok(out)
}
