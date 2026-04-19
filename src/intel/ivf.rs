//! # Intel (Vulkan) IVF Index
//!
//! Mirror of [`crate::cuda::CudaIvfIndex`] / [`crate::rocm::RocmIvfIndex`]
//! on the Vulkan Compute backend. Uses the pre-built `sgemv_dot` and
//! `sgemm_dot` compute pipelines for every GPU-side dot-product pass;
//! argmin and top-K run on the host, matching the CUDA/ROCm/Metal scope
//! of v1.
//!
//! ⚠️ AUTHORED BLIND — see `phase3c_add-intel-backend`.

#![cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))]

use super::context::{IntelContext, SgemmPushConstants, SgemvPushConstants};
use super::vector_storage::{
    VulkanBuffer, allocate_host_visible_buffer, dispatch_three_buffer_compute,
    dispatch_three_buffer_compute_ranged, read_f32_vec, write_f32_slice,
};
use crate::error::{HiveGpuError, Result};
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector, IvfConfig};
use std::sync::Arc;
use tracing::{debug, info};

pub struct IntelIvfIndex {
    context: Arc<IntelContext>,
    dimension: usize,
    metric: GpuDistanceMetric,
    config: IvfConfig,

    centroids: Option<VulkanBuffer>,
    centroid_norms_sq: Vec<f32>,

    vectors: Option<VulkanBuffer>,
    vector_norms_sq: Vec<f32>,
    cluster_offsets: Vec<usize>,
    ids_by_local_index: Vec<String>,
    vector_count: usize,
    trained: bool,
}

impl std::fmt::Debug for IntelIvfIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntelIvfIndex")
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("n_list", &self.config.n_list)
            .field("nprobe", &self.config.nprobe)
            .field("vector_count", &self.vector_count)
            .field("trained", &self.trained)
            .finish()
    }
}

impl IntelIvfIndex {
    pub fn new(
        context: Arc<IntelContext>,
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
            "intel ivf build: dim={} n={} n_list={} training_sample={}",
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

        // Upload centroids.
        let centroids_bytes = centroids_flat.len() * std::mem::size_of::<f32>();
        let centroids_buf = allocate_host_visible_buffer(&self.context, centroids_bytes)?;
        write_f32_slice(&centroids_buf, &centroids_flat)?;

        // Upload reordered vectors.
        let vectors_bytes = reordered.len() * std::mem::size_of::<f32>();
        let vectors_buf = allocate_host_visible_buffer(&self.context, vectors_bytes)?;
        write_f32_slice(&vectors_buf, &reordered)?;

        let mut centroid_norms_sq = Vec::with_capacity(self.config.n_list);
        for i in 0..self.config.n_list {
            let start = i * self.dimension;
            centroid_norms_sq.push(dot_self(&centroids_flat[start..start + self.dimension]));
        }

        if let Some(old) = self.centroids.take() {
            old.destroy(self.context.device());
        }
        if let Some(old) = self.vectors.take() {
            old.destroy(self.context.device());
        }

        self.centroids = Some(centroids_buf);
        self.centroid_norms_sq = centroid_norms_sq;
        self.vectors = Some(vectors_buf);
        self.vector_norms_sq = reordered_norms;
        self.cluster_offsets = offsets;
        self.ids_by_local_index = reordered_ids;
        self.vector_count = vectors.len();
        self.trained = true;

        info!(
            "intel ivf build done: {} vectors across {} clusters",
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

        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;
        let vectors = self
            .vectors
            .as_ref()
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("not trained".to_string()))?;

        // 1. Coarse: sgemv_dot against the centroids buffer.
        let coarse = self.sgemv_dot_whole_buffer(centroids, self.config.n_list, query)?;
        let query_norm_sq = dot_self(query);
        let probe = select_nprobe_clusters(
            &coarse,
            &self.centroid_norms_sq,
            query_norm_sq,
            self.config.nprobe,
        );

        // 2. Refined: for each probed cluster, sgemv_dot against a
        //    sub-range of the vectors buffer.
        let query_bytes = query.len() * std::mem::size_of::<f32>();
        let query_buf = allocate_host_visible_buffer(&self.context, query_bytes)?;
        write_f32_slice(&query_buf, query)?;

        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for cluster_id in &probe {
            let start = self.cluster_offsets[*cluster_id];
            let end = self.cluster_offsets[cluster_id + 1];
            let count = end - start;
            if count == 0 {
                continue;
            }

            let scores_bytes = count * std::mem::size_of::<f32>();
            let scores_buf = allocate_host_visible_buffer(&self.context, scores_bytes)?;

            let matrix_byte_offset = (start * self.dimension * std::mem::size_of::<f32>()) as u64;
            let matrix_byte_range = (count * self.dimension * std::mem::size_of::<f32>()) as u64;

            let pipeline = self.context.sgemv_dot();
            let pc = SgemvPushConstants {
                dimension: self.dimension as u32,
                n_vectors: count as u32,
            };
            let grid = (((count as u32) + 255) / 256, 1, 1);
            dispatch_three_buffer_compute_ranged(
                &self.context,
                pipeline,
                [
                    (vectors, matrix_byte_offset, matrix_byte_range),
                    (&query_buf, 0, ash::vk::WHOLE_SIZE),
                    (&scores_buf, 0, ash::vk::WHOLE_SIZE),
                ],
                pc,
                grid,
            )?;

            let scores = read_f32_vec(&scores_buf, count)?;
            scores_buf.destroy(self.context.device());

            for (j, dot) in scores.into_iter().enumerate() {
                let local_idx = start + j;
                let metric_score = self.score_from_dot(dot, local_idx, query_norm_sq);
                candidates.push((local_idx, metric_score));
            }
        }

        query_buf.destroy(self.context.device());

        Ok(self.finalize_top_k(candidates, k))
    }

    // --- internals ------------------------------------------------------

    fn sgemv_dot_whole_buffer(
        &self,
        matrix_buf: &VulkanBuffer,
        n_rows: usize,
        query: &[f32],
    ) -> Result<Vec<f32>> {
        let query_bytes = query.len() * std::mem::size_of::<f32>();
        let query_buf = allocate_host_visible_buffer(&self.context, query_bytes)?;
        write_f32_slice(&query_buf, query)?;
        let scores_bytes = n_rows * std::mem::size_of::<f32>();
        let scores_buf = allocate_host_visible_buffer(&self.context, scores_bytes)?;

        let pipeline = self.context.sgemv_dot();
        let pc = SgemvPushConstants {
            dimension: self.dimension as u32,
            n_vectors: n_rows as u32,
        };
        let grid = (((n_rows as u32) + 255) / 256, 1, 1);
        dispatch_three_buffer_compute(
            &self.context,
            pipeline,
            [matrix_buf, &query_buf, &scores_buf],
            pc,
            grid,
        )?;

        let scores = read_f32_vec(&scores_buf, n_rows)?;
        query_buf.destroy(self.context.device());
        scores_buf.destroy(self.context.device());
        Ok(scores)
    }

    /// Run sgemm_dot over `(samples × centroids^T)` and reduce to
    /// per-sample cluster ids on the host.
    fn assign_to_centroids(
        &self,
        flat_samples: &[f32],
        n_samples: usize,
        centroids_flat: &[f32],
    ) -> Result<Vec<u32>> {
        let samples_bytes = flat_samples.len() * std::mem::size_of::<f32>();
        let centroids_bytes = centroids_flat.len() * std::mem::size_of::<f32>();
        let out_bytes = n_samples * self.config.n_list * std::mem::size_of::<f32>();

        let samples_buf = allocate_host_visible_buffer(&self.context, samples_bytes)?;
        write_f32_slice(&samples_buf, flat_samples)?;
        let centroids_buf = allocate_host_visible_buffer(&self.context, centroids_bytes)?;
        write_f32_slice(&centroids_buf, centroids_flat)?;
        let out_buf = allocate_host_visible_buffer(&self.context, out_bytes)?;

        let pipeline = self.context.sgemm_dot();
        let pc = SgemmPushConstants {
            dimension: self.dimension as u32,
            n_list: self.config.n_list as u32,
            n_samples: n_samples as u32,
        };
        let grid = (
            ((n_samples as u32) + 15) / 16,
            ((self.config.n_list as u32) + 15) / 16,
            1,
        );
        dispatch_three_buffer_compute(
            &self.context,
            pipeline,
            [&samples_buf, &centroids_buf, &out_buf],
            pc,
            grid,
        )?;

        let host_dots = read_f32_vec(&out_buf, n_samples * self.config.n_list)?;
        samples_buf.destroy(self.context.device());
        centroids_buf.destroy(self.context.device());
        out_buf.destroy(self.context.device());

        // Precompute centroid norms then argmax(2 * dot - ||c||^2).
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

impl Drop for IntelIvfIndex {
    fn drop(&mut self) {
        if let Some(buf) = self.centroids.take() {
            buf.destroy(self.context.device());
        }
        if let Some(buf) = self.vectors.take() {
            buf.destroy(self.context.device());
        }
    }
}

// ---- shared helpers (copied from CUDA/ROCm IVFs; kept local for now so
//       the three backends can evolve independently) ---------------------

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
