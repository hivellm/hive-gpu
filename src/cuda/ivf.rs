//! # CUDA IVF Index
//!
//! Inverted File index backed by cuBLAS SGEMM/SGEMV. Clusters the vector
//! space into `n_list` Voronoi cells via k-means and at query time only
//! searches the `nprobe` cells closest to the query.
//!
//! ## Scope of v1
//!
//! - Train-once, build-once, read-only search (no online incremental add
//!   after build — rebuild on dataset changes).
//! - L2-based k-means for clustering (canonical FAISS choice). Search honors
//!   the metric chosen by the caller (Cosine/Euclidean/DotProduct).
//! - All argmin / sort work happens on the host after a single dtoh copy.
//!   For the dataset sizes we target (n_list ≤ 4096, batch ≤ 1M) this is
//!   a non-bottleneck; a GPU argmin kernel is a future optimization.
//!
//! The hot paths that run on the GPU are:
//! - `cuBLAS SGEMM` for training assignment (samples × centroids^T).
//! - `cuBLAS SGEMM` for add-time assignment (vectors × centroids^T).
//! - `cuBLAS SGEMV` for coarse cluster selection at query time.
//! - `cuBLAS SGEMV` per probed cluster for refined scoring.

use super::context::CudaContext;
use crate::error::{HiveGpuError, Result};
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector, IvfConfig};
use std::sync::Arc;
use tracing::{debug, info};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::cublas::{Gemm, GemmConfig, Gemv, GemvConfig, sys as cublas_sys};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::CudaSlice;

/// IVF index on a CUDA backend.
///
/// Build once via [`CudaIvfIndex::build`] then query via
/// [`CudaIvfIndex::search`]. `set_nprobe` adjusts recall/latency at runtime
/// without rebuilding the index.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub struct CudaIvfIndex {
    context: Arc<CudaContext>,
    dimension: usize,
    metric: GpuDistanceMetric,
    config: IvfConfig,
    /// Centroids stored row-major as a flat `(n_list, dimension)` buffer.
    centroids: Option<CudaSlice<f32>>,
    /// Squared L2 norms of every centroid (host side, length `n_list`).
    centroid_norms_sq: Vec<f32>,
    /// Flat vector storage reordered so that cluster 0's members come
    /// first, then cluster 1's, etc. Shape `(total_vectors, dimension)`.
    vectors: Option<CudaSlice<f32>>,
    /// Squared L2 norms per stored vector (host side, same order as
    /// `vectors`).
    vector_norms_sq: Vec<f32>,
    /// Index i contains the start offset (in number of vectors) of cluster
    /// i in `vectors`. Length `n_list + 1`; `cluster_offsets[n_list]`
    /// equals `vector_count`.
    cluster_offsets: Vec<usize>,
    /// Original IDs of every stored vector, ordered to match `vectors`.
    ids_by_local_index: Vec<String>,
    /// Number of vectors currently indexed.
    vector_count: usize,
    /// Whether [`Self::build`] has completed at least once.
    trained: bool,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl std::fmt::Debug for CudaIvfIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaIvfIndex")
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("n_list", &self.config.n_list)
            .field("nprobe", &self.config.nprobe)
            .field("vector_count", &self.vector_count)
            .field("trained", &self.trained)
            .finish()
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaIvfIndex {
    /// Create a new IVF index. The index is empty until
    /// [`Self::build`] is called.
    pub fn new(
        context: Arc<CudaContext>,
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
            vector_norms_sq: Vec::new(),
            cluster_offsets: Vec::new(),
            ids_by_local_index: Vec::new(),
            vector_count: 0,
            trained: false,
        })
    }

    /// Train and populate the index from `vectors` in a single pass.
    ///
    /// Invalidates any prior training state. After this returns the index
    /// is queryable via [`Self::search`].
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

        // 1. Select a training sample for k-means.
        let sample_size = self.config.training_sample_size.min(vectors.len());
        let flat_sample: Vec<f32> = vectors
            .iter()
            .take(sample_size)
            .flat_map(|v| v.data.iter().copied())
            .collect();

        info!(
            "cuda ivf build: dim={} n={} n_list={} training_sample={}",
            self.dimension,
            vectors.len(),
            self.config.n_list,
            sample_size
        );

        // 2. Train centroids on the host (+ cuBLAS for the per-iteration
        //    distance SGEMM).
        let centroids_flat =
            self.train_kmeans(&flat_sample, sample_size, self.config.kmeans_iters)?;
        debug_assert_eq!(centroids_flat.len(), self.config.n_list * self.dimension);

        // 3. Assign every input vector to its nearest centroid.
        let flat_all: Vec<f32> = vectors
            .iter()
            .flat_map(|v| v.data.iter().copied())
            .collect();
        let assignments = self.assign_to_centroids(&flat_all, vectors.len(), &centroids_flat)?;
        debug_assert_eq!(assignments.len(), vectors.len());

        // 4. Reorder vectors so cluster members are contiguous.
        let (offsets, perm) = build_cluster_layout(&assignments, self.config.n_list);

        let mut reordered = vec![0f32; flat_all.len()];
        let mut reordered_ids = Vec::with_capacity(vectors.len());
        let mut reordered_norms = Vec::with_capacity(vectors.len());
        for (local_idx, &global_idx) in perm.iter().enumerate() {
            let src_start = global_idx * self.dimension;
            let dst_start = local_idx * self.dimension;
            reordered[dst_start..dst_start + self.dimension]
                .copy_from_slice(&flat_all[src_start..src_start + self.dimension]);
            reordered_ids.push(vectors[global_idx].id.clone());
            reordered_norms.push(dot_self(&flat_all[src_start..src_start + self.dimension]));
        }

        // 5. Upload centroids and reordered vectors to device memory.
        let device = self.context.device();
        let centroids_dev = device
            .htod_copy(centroids_flat.clone())
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy centroids: {e:?}")))?;
        let vectors_dev = device
            .htod_copy(reordered)
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy vectors: {e:?}")))?;

        // 6. Precompute centroid squared norms.
        let mut centroid_norms_sq = Vec::with_capacity(self.config.n_list);
        for i in 0..self.config.n_list {
            let start = i * self.dimension;
            centroid_norms_sq.push(dot_self(&centroids_flat[start..start + self.dimension]));
        }

        self.centroids = Some(centroids_dev);
        self.centroid_norms_sq = centroid_norms_sq;
        self.vectors = Some(vectors_dev);
        self.vector_norms_sq = reordered_norms;
        self.cluster_offsets = offsets;
        self.ids_by_local_index = reordered_ids;
        self.vector_count = vectors.len();
        self.trained = true;

        info!(
            "cuda ivf build done: {} vectors across {} clusters (min={} max={} avg={:.1})",
            self.vector_count,
            self.config.n_list,
            self.cluster_offsets
                .windows(2)
                .map(|w| w[1] - w[0])
                .min()
                .unwrap_or(0),
            self.cluster_offsets
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap_or(0),
            self.vector_count as f32 / self.config.n_list as f32,
        );

        Ok(())
    }

    /// Update `nprobe` at query time. Must satisfy `1 <= nprobe <= n_list`.
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

    /// Current `nprobe` setting.
    pub fn nprobe(&self) -> usize {
        self.config.nprobe
    }

    /// Configured `n_list`.
    pub fn n_list(&self) -> usize {
        self.config.n_list
    }

    /// Number of indexed vectors.
    pub fn vector_count(&self) -> usize {
        self.vector_count
    }

    /// Whether [`Self::build`] has completed.
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Search for the `k` nearest neighbours of `query`.
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

        // 1. Coarse: compute query · every centroid via SGEMV and pick top
        //    `nprobe` by L2 distance to the query.
        let coarse_scores = self.centroid_dot_products(query)?;
        let query_norm_sq = dot_self(query);
        let coarse_probe = select_nprobe_clusters(
            &coarse_scores,
            &self.centroid_norms_sq,
            query_norm_sq,
            self.config.nprobe,
        );

        // 2. Refined: for each probed cluster, SGEMV over its contiguous
        //    subrange and score with the caller's metric.
        let device = self.context.device();
        let query_dev = device
            .htod_copy(query.to_vec())
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy query: {e:?}")))?;
        let vectors_dev = self.vectors.as_ref().expect("trained => vectors exist");

        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for cluster_id in &coarse_probe {
            let start = self.cluster_offsets[*cluster_id];
            let end = self.cluster_offsets[cluster_id + 1];
            let count = end - start;
            if count == 0 {
                continue;
            }

            let mut scores_dev = device
                .alloc_zeros::<f32>(count)
                .map_err(|e| HiveGpuError::CudaError(format!("alloc scores: {e:?}")))?;

            // Sub-slice the reordered vector buffer down to the cluster
            // subrange. `start * dimension` and `end * dimension` are
            // element offsets, not bytes.
            let slice_start = start * self.dimension;
            let slice_end = end * self.dimension;
            let cluster_view = vectors_dev.slice(slice_start..slice_end);

            let cfg = GemvConfig::<f32> {
                trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                m: self.dimension as i32,
                n: count as i32,
                alpha: 1.0,
                lda: self.dimension as i32,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            // SAFETY: `cluster_view` borrows `vectors_dev` (a live device
            // allocation) for a range we just bounded by `cluster_offsets`;
            // `query_dev` and `scores_dev` are live device buffers of the
            // correct shapes; cuBLAS handle is bound to the same device.
            unsafe {
                self.context
                    .blas()
                    .gemv(cfg, &cluster_view, &query_dev, &mut scores_dev)
                    .map_err(|e| HiveGpuError::CublasError(format!("sgemv cluster: {e:?}")))?;
            }

            let host_scores = device
                .dtoh_sync_copy(&scores_dev)
                .map_err(|e| HiveGpuError::CudaError(format!("dtoh scores: {e:?}")))?;

            for (j, dot) in host_scores.into_iter().enumerate() {
                let local_idx = start + j;
                let metric_score = self.score_from_dot(dot, local_idx, query_norm_sq);
                candidates.push((local_idx, metric_score));
            }
        }

        // 3. Top-K on the host.
        candidates = self.select_top_k(candidates, k);

        Ok(candidates
            .into_iter()
            .map(|(local_idx, score)| GpuSearchResult {
                id: self.ids_by_local_index[local_idx].clone(),
                score: self.similarity_from_metric(score),
                index: local_idx,
            })
            .collect())
    }

    // --- internals ------------------------------------------------------

    /// SGEMM over `(samples × centroids^T)` followed by a host-side argmin
    /// that returns a cluster id per sample.
    fn assign_to_centroids(
        &self,
        flat_samples: &[f32],
        n_samples: usize,
        centroids_flat: &[f32],
    ) -> Result<Vec<u32>> {
        let device = self.context.device();
        let samples_dev = device
            .htod_copy(flat_samples.to_vec())
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy samples: {e:?}")))?;
        let centroids_dev = device
            .htod_copy(centroids_flat.to_vec())
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy centroids: {e:?}")))?;
        let mut dots_dev = device
            .alloc_zeros::<f32>(n_samples * self.config.n_list)
            .map_err(|e| HiveGpuError::CudaError(format!("alloc dots: {e:?}")))?;

        // We want `dots[i, j] = samples[i] · centroids[j]` with shape
        // `(n_samples, n_list)` in row-major.
        //
        // cuBLAS is column-major: a row-major `(M, K) @ (K, N)` becomes a
        // column-major `(K, M)^T @ (K, N)^T -> (M, N)` by swapping operands
        // and transposing. Computed value per entry is still the dot
        // product of row i of samples with row j of centroids.
        let cfg = GemmConfig::<f32> {
            transa: cublas_sys::cublasOperation_t::CUBLAS_OP_T, // centroids^T
            transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N, // samples
            m: self.config.n_list as i32,
            n: n_samples as i32,
            k: self.dimension as i32,
            alpha: 1.0,
            lda: self.dimension as i32,
            ldb: self.dimension as i32,
            beta: 0.0,
            ldc: self.config.n_list as i32,
        };
        // SAFETY: All three buffers were just allocated for `f32` on the
        // same device; shapes match the SGEMM config above; buffers live
        // for the duration of this function.
        unsafe {
            self.context
                .blas()
                .gemm(cfg, &centroids_dev, &samples_dev, &mut dots_dev)
                .map_err(|e| HiveGpuError::CublasError(format!("sgemm assign: {e:?}")))?;
        }

        let host_dots = device
            .dtoh_sync_copy(&dots_dev)
            .map_err(|e| HiveGpuError::CudaError(format!("dtoh dots: {e:?}")))?;

        // L2 distance squared = ||s||^2 + ||c||^2 - 2 s·c. Minimising over j
        // is equivalent to maximising `2 s·c - ||c||^2` since ||s||^2 is
        // constant per row.
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

    /// Full Lloyd training loop. Returns the flat `(n_list, dimension)`
    /// centroid buffer.
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
                &mut centroids,
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

    /// SGEMV of query against every centroid. Returns `n_list` dot products.
    fn centroid_dot_products(&self, query: &[f32]) -> Result<Vec<f32>> {
        let device = self.context.device();
        let centroids_dev = self
            .centroids
            .as_ref()
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;
        let query_dev = device
            .htod_copy(query.to_vec())
            .map_err(|e| HiveGpuError::CudaError(format!("htod_copy query: {e:?}")))?;
        let mut dots_dev = device
            .alloc_zeros::<f32>(self.config.n_list)
            .map_err(|e| HiveGpuError::CudaError(format!("alloc coarse scores: {e:?}")))?;

        // Centroids layout (n_list, dimension) row-major → column-major it is
        // (dimension, n_list). SGEMV with trans=T gives y[j] = col_j · query
        // = centroid_j · query.
        let cfg = GemvConfig::<f32> {
            trans: cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            m: self.dimension as i32,
            n: self.config.n_list as i32,
            alpha: 1.0,
            lda: self.dimension as i32,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: centroids_dev, query_dev, dots_dev are live device buffers
        // of the correct shapes; cublas handle is bound to the same device.
        unsafe {
            self.context
                .blas()
                .gemv(cfg, centroids_dev, &query_dev, &mut dots_dev)
                .map_err(|e| HiveGpuError::CublasError(format!("sgemv coarse: {e:?}")))?;
        }

        device
            .dtoh_sync_copy(&dots_dev)
            .map_err(|e| HiveGpuError::CudaError(format!("dtoh coarse scores: {e:?}")))
    }

    /// Transform a raw dot product into the sorting key for the caller's
    /// metric. For Cosine/DotProduct higher is better; for L2 we return
    /// squared distance (lower is better).
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

    /// Map the internal sorting key back to the user-facing similarity
    /// score used by `GpuSearchResult`.
    fn similarity_from_metric(&self, score: f32) -> f32 {
        match self.metric {
            GpuDistanceMetric::Euclidean => 1.0 / (1.0 + score.sqrt()),
            _ => score,
        }
    }

    fn select_top_k(&self, mut candidates: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
        match self.metric {
            GpuDistanceMetric::Euclidean => candidates
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
            _ => candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
        }
        candidates.truncate(k);
        candidates
    }
}

// --- helpers ------------------------------------------------------------

#[inline]
fn dot_self(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum()
}

/// Squared L2 distance between two slices.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Pick `nprobe` cluster ids with the smallest L2 distance to the query.
///
/// `dots[i] = centroid_i · query`; L2² to centroid i is
/// `||c_i||² - 2·dot + ||q||²`. Since `||q||²` is constant we drop it.
fn select_nprobe_clusters(
    dots: &[f32],
    centroid_norms_sq: &[f32],
    query_norm_sq: f32,
    nprobe: usize,
) -> Vec<usize> {
    let _ = query_norm_sq; // intentionally constant across clusters
    let mut scored: Vec<(usize, f32)> = dots
        .iter()
        .enumerate()
        .map(|(i, &dot)| (i, centroid_norms_sq[i] - 2.0 * dot))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(nprobe);
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Given a per-sample cluster assignment, produce the cluster offset array
/// (length `n_list + 1`) and a permutation `perm[local_idx] = global_idx`
/// that reorders samples so members of cluster 0 come first, then 1, etc.
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

/// K-means++ seeding. Returns a flat `(n_list, dimension)` centroid buffer.
fn kmeans_plus_plus_init(
    flat_sample: &[f32],
    n_samples: usize,
    dimension: usize,
    n_list: usize,
    seed: Option<u64>,
) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed.unwrap_or(0x9E37_79B9_7F4A_7C15));
    let mut centroids = Vec::with_capacity(n_list * dimension);

    // First centroid: pick a random sample.
    let first = (rng.next_u64() as usize) % n_samples;
    centroids.extend_from_slice(&flat_sample[first * dimension..(first + 1) * dimension]);

    // Distance to the nearest chosen centroid, per sample.
    let mut min_dist_sq = vec![f32::INFINITY; n_samples];
    for c in 0..n_list - 1 {
        let last_centroid = &centroids[c * dimension..(c + 1) * dimension];
        // Update nearest-centroid distances using the newly added centroid.
        for i in 0..n_samples {
            let d = l2_sq(
                &flat_sample[i * dimension..(i + 1) * dimension],
                last_centroid,
            );
            if d < min_dist_sq[i] {
                min_dist_sq[i] = d;
            }
        }
        // Weighted pick proportional to D(x)^2.
        let total: f64 = min_dist_sq.iter().map(|&x| x as f64).sum();
        if total <= 0.0 {
            // All samples coincide with existing centroids — duplicate a
            // random sample to keep the count correct.
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

/// Lloyd update step. Returns `(new_centroids, total_inertia)`.
fn update_centroids(
    flat_sample: &[f32],
    n_samples: usize,
    assignments: &[u32],
    centroids: &mut [f32],
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
            // Empty cluster: reseed from the sample furthest from its
            // assigned centroid.
            let mut worst_sample = 0;
            let mut worst_d = -1f32;
            for i in 0..n_samples {
                let a = assignments[i] as usize;
                let centroid = &centroids[a * dimension..(a + 1) * dimension];
                let d = l2_sq(&flat_sample[i * dimension..(i + 1) * dimension], centroid);
                if d > worst_d {
                    worst_d = d;
                    worst_sample = i;
                }
            }
            new_centroids[j * dimension..(j + 1) * dimension].copy_from_slice(
                &flat_sample[worst_sample * dimension..(worst_sample + 1) * dimension],
            );
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

/// Tiny deterministic RNG (SplitMix64). Good enough for seeding k-means
/// without pulling in the `rand` crate.
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
