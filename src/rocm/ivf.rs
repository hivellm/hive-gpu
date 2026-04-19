//! # ROCm IVF Index
//!
//! Mirror of [`crate::cuda::CudaIvfIndex`] on the ROCm backend. Uses
//! `rocblas_sgemm` for k-means assignment and `rocblas_sgemv` for both
//! coarse cluster selection and per-cluster refined search; argmin and
//! top-K run on the host, matching the CUDA scope of v1.
//!
//! ⚠️ AUTHORED BLIND — see `phase3b_add-rocm-backend`.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use super::context::RocmContext;
use super::ffi::{self, HipDevicePtr_t, ROCBLAS_OP_N, ROCBLAS_OP_T, hip_check, rocblas_check};
use super::vector_storage::{hip_free, hip_malloc, hip_memcpy_from_slice, hip_memcpy_to_slice};
use crate::error::{HiveGpuError, Result};
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector, IvfConfig};
use std::sync::Arc;
use tracing::{debug, info};

/// IVF index on ROCm. API mirrors `CudaIvfIndex` and `MetalIvfIndex` for
/// cross-backend portability.
pub struct RocmIvfIndex {
    context: Arc<RocmContext>,
    dimension: usize,
    metric: GpuDistanceMetric,
    config: IvfConfig,

    centroids: Option<HipDevicePtr_t>,
    centroid_norms_sq: Vec<f32>,

    vectors: Option<HipDevicePtr_t>,
    vectors_bytes: usize,
    vector_norms_sq: Vec<f32>,
    cluster_offsets: Vec<usize>,
    ids_by_local_index: Vec<String>,
    vector_count: usize,
    trained: bool,
}

// SAFETY: All device handles are bound to the context's stream.
unsafe impl Send for RocmIvfIndex {}
unsafe impl Sync for RocmIvfIndex {}

impl std::fmt::Debug for RocmIvfIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmIvfIndex")
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("n_list", &self.config.n_list)
            .field("nprobe", &self.config.nprobe)
            .field("vector_count", &self.vector_count)
            .field("trained", &self.trained)
            .finish()
    }
}

impl RocmIvfIndex {
    pub fn new(
        context: Arc<RocmContext>,
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
            centroids: None,
            centroid_norms_sq: Vec::new(),
            vectors: None,
            vectors_bytes: 0,
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
            "rocm ivf build: dim={} n={} n_list={} training_sample={}",
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

        // Upload centroids + reordered vectors to device.
        let centroids_bytes = centroids_flat.len() * std::mem::size_of::<f32>();
        let centroids_dev = hip_malloc(centroids_bytes)?;
        hip_memcpy_from_slice(centroids_dev, &centroids_flat)?;

        let vectors_bytes = reordered.len() * std::mem::size_of::<f32>();
        let vectors_dev = hip_malloc(vectors_bytes)?;
        hip_memcpy_from_slice(vectors_dev, &reordered)?;

        let mut centroid_norms_sq = Vec::with_capacity(self.config.n_list);
        for i in 0..self.config.n_list {
            let start = i * self.dimension;
            centroid_norms_sq.push(dot_self(&centroids_flat[start..start + self.dimension]));
        }

        // Release any prior device buffers (idempotent rebuild).
        if let Some(ptr) = self.centroids.take() {
            let _ = hip_free(ptr);
        }
        if let Some(ptr) = self.vectors.take() {
            let _ = hip_free(ptr);
        }

        self.centroids = Some(centroids_dev);
        self.centroid_norms_sq = centroid_norms_sq;
        self.vectors = Some(vectors_dev);
        self.vectors_bytes = vectors_bytes;
        self.vector_norms_sq = reordered_norms;
        self.cluster_offsets = offsets;
        self.ids_by_local_index = reordered_ids;
        self.vector_count = vectors.len();
        self.trained = true;

        info!(
            "rocm ivf build done: {} vectors across {} clusters",
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

        let centroids_dev = self
            .centroids
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;
        let vectors_dev = self
            .vectors
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;

        // 1. Coarse query-to-centroid SGEMV.
        let coarse_dots = self.sgemv_dot(centroids_dev, self.config.n_list, query)?;
        let query_norm_sq = dot_self(query);
        let probe = select_nprobe_clusters(
            &coarse_dots,
            &self.centroid_norms_sq,
            query_norm_sq,
            self.config.nprobe,
        );

        // 2. Refined per-cluster SGEMV.
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for cluster_id in &probe {
            let start = self.cluster_offsets[*cluster_id];
            let end = self.cluster_offsets[cluster_id + 1];
            let count = end - start;
            if count == 0 {
                continue;
            }
            // SAFETY: element offset bounded by `vector_count * dimension`.
            let sub_ptr =
                unsafe { (vectors_dev as *mut f32).add(start * self.dimension) } as HipDevicePtr_t;
            let scores = self.sgemv_dot(sub_ptr, count, query)?;
            for (j, dot) in scores.into_iter().enumerate() {
                let local_idx = start + j;
                let m = self.score_from_dot(dot, local_idx, query_norm_sq);
                candidates.push((local_idx, m));
            }
        }

        Ok(self.finalize_top_k(candidates, k))
    }

    // --- internals ------------------------------------------------------

    /// Dispatch rocBLAS SGEMV to produce `n_rows` dot products.
    fn sgemv_dot(
        &self,
        matrix_dev: HipDevicePtr_t,
        n_rows: usize,
        query: &[f32],
    ) -> Result<Vec<f32>> {
        let lib = ffi::require_hip_lib()?;
        let query_bytes = query.len() * std::mem::size_of::<f32>();
        let query_dev = hip_malloc(query_bytes)?;
        hip_memcpy_from_slice(query_dev, query)?;
        let scores_bytes = n_rows * std::mem::size_of::<f32>();
        let scores_dev = hip_malloc(scores_bytes)?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // SAFETY: All three device pointers are live; rocBLAS handle is
        // bound to the context's stream.
        let status = unsafe {
            (lib.rocblas_sgemv)(
                self.context.rocblas_handle(),
                ROCBLAS_OP_T,
                self.dimension as i32,
                n_rows as i32,
                &alpha as *const f32,
                matrix_dev as *const f32,
                self.dimension as i32,
                query_dev as *const f32,
                1,
                &beta as *const f32,
                scores_dev as *mut f32,
                1,
            )
        };
        rocblas_check(status, "rocblas_sgemv")?;
        // SAFETY: stream created by our context, still live.
        let status = unsafe { (lib.hip_stream_synchronize)(self.context.stream()) };
        hip_check(status, "hipStreamSynchronize")?;

        let mut out = vec![0f32; n_rows];
        hip_memcpy_to_slice(out.as_mut_slice(), scores_dev)?;

        let _ = hip_free(query_dev);
        let _ = hip_free(scores_dev);
        Ok(out)
    }

    /// rocBLAS SGEMM producing `(n_samples, n_list)` of dot products,
    /// followed by host-side argmin for cluster assignment.
    fn assign_to_centroids(
        &self,
        flat_samples: &[f32],
        n_samples: usize,
        centroids_flat: &[f32],
    ) -> Result<Vec<u32>> {
        let lib = ffi::require_hip_lib()?;
        let samples_bytes = flat_samples.len() * std::mem::size_of::<f32>();
        let centroids_bytes = centroids_flat.len() * std::mem::size_of::<f32>();
        let dots_bytes = n_samples * self.config.n_list * std::mem::size_of::<f32>();

        let samples_dev = hip_malloc(samples_bytes)?;
        hip_memcpy_from_slice(samples_dev, flat_samples)?;
        let centroids_dev = hip_malloc(centroids_bytes)?;
        hip_memcpy_from_slice(centroids_dev, centroids_flat)?;
        let dots_dev = hip_malloc(dots_bytes)?;

        // We want `dots[i, j] = sample_i . centroid_j` with shape
        // `(n_samples, n_list)` row-major. rocBLAS is column-major;
        // using the `cublas-style` swap of operands + transpose produces
        // the right result (`rocblas_sgemm(T, N, n_list, n_samples,
        // dim, ...)` with A = centroids, B = samples, C = dots).
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // SAFETY: buffer shapes match the declared dimensions.
        let status = unsafe {
            (lib.rocblas_sgemm)(
                self.context.rocblas_handle(),
                ROCBLAS_OP_T,
                ROCBLAS_OP_N,
                self.config.n_list as i32,
                n_samples as i32,
                self.dimension as i32,
                &alpha as *const f32,
                centroids_dev as *const f32,
                self.dimension as i32,
                samples_dev as *const f32,
                self.dimension as i32,
                &beta as *const f32,
                dots_dev as *mut f32,
                self.config.n_list as i32,
            )
        };
        rocblas_check(status, "rocblas_sgemm assign")?;
        // SAFETY: stream live.
        let status = unsafe { (lib.hip_stream_synchronize)(self.context.stream()) };
        hip_check(status, "hipStreamSynchronize")?;

        let mut host_dots = vec![0f32; n_samples * self.config.n_list];
        hip_memcpy_to_slice(host_dots.as_mut_slice(), dots_dev)?;

        let _ = hip_free(samples_dev);
        let _ = hip_free(centroids_dev);
        let _ = hip_free(dots_dev);

        // Precompute centroid norms for argmin.
        let mut centroid_norms_sq = Vec::with_capacity(self.config.n_list);
        for j in 0..self.config.n_list {
            let start = j * self.dimension;
            centroid_norms_sq.push(dot_self(&centroids_flat[start..start + self.dimension]));
        }

        let mut assignments = vec![0u32; n_samples];
        for i in 0..n_samples {
            let row = &host_dots[i * self.config.n_list..(i + 1) * self.config.n_list];
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

impl Drop for RocmIvfIndex {
    fn drop(&mut self) {
        if let Some(ptr) = self.centroids.take() {
            let _ = hip_free(ptr);
        }
        if let Some(ptr) = self.vectors.take() {
            let _ = hip_free(ptr);
        }
    }
}

// ---------- shared helpers ---------------------------------------------

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
