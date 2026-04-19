# Proposal: phase4b_add-cuda-ivf-index

## Why

The CUDA backend shipped in 0.2.0 is brute-force only via cuBLAS SGEMV. At
100 K vectors it already beats CPU by 3.25×, but the asymptotic is still
O(N) per query — at 1 M+ vectors latency stops being acceptable. IVF
(Inverted File Index) is the canonical first-stage ANN that restores
sub-linear behaviour: cluster vectors into `n_list` centroids offline, at
query time only probe the `nprobe` closest clusters.

IVF is also well-suited to GPUs because the hot loops reduce to batched
matrix operations:

- Training = Lloyd iterations = SGEMM (assign) + reduction (update
  centroids) — native strengths of cuBLAS.
- Coarse assignment at query time = one SGEMV against centroids.
- Refined search = brute-force SGEMV per probed cluster, same primitive
  the 0.2.0 backend already ships.

The Vectorizer needs an index that scales beyond what brute-force permits
while keeping GPU acceleration. HNSW on GPU is an open research problem
and almost always loses to CPU HNSW; IVF is the pragmatic middle ground
with decades of production deployment behind it (FAISS, Milvus, cuVS).

## What Changes

- New module `src/cuda/ivf.rs` exposing `CudaIvfIndex`:
  - Owns an `Arc<CudaContext>` (device + cuBLAS handle shared with
    `CudaVectorStorage`).
  - Stores centroids in a single device-local `CudaSlice<f32>` of shape
    `(n_list, dimension)`.
  - Stores inverted lists as two device buffers:
    - `cluster_offsets: CudaSlice<u32>` of length `n_list + 1`.
    - `cluster_member_indices: CudaSlice<u32>` of length `total_vectors`.
  - Stores vectors in the same flat layout the brute-force backend uses
    (`CudaSlice<f32>` of shape `(total_vectors, dimension)`), so the
    per-cluster search can reuse the existing SGEMV path.
- Public API:
  - `CudaIvfIndex::new(ctx, dimension, metric, IvfConfig { n_list,
    nprobe })`
  - `train(&mut self, sample: &[GpuVector], n_iter: usize)` — k-means++
    init followed by `n_iter` Lloyd iterations.
  - `add_vectors(&mut self, vectors: &[GpuVector])` — assign each vector
    to its nearest centroid (SGEMV against centroids, argmin), append to
    the appropriate inverted list.
  - `search(&self, query: &[f32], k: usize) -> Vec<GpuSearchResult>` —
    compute query-to-centroid distances, select top-`nprobe` clusters
    (CPU argpartition after score readback, O(n_list)), brute-force
    search only those clusters via cuBLAS SGEMV over the contiguous
    subrange.
  - `set_nprobe(&mut self, nprobe: usize)` for tuning recall/latency at
    query time.
- Configuration:
  - `IvfConfig { n_list: usize, nprobe: usize, training_sample_size:
    usize, kmeans_iters: usize }` with sensible defaults (`n_list =
    sqrt(N)`, `nprobe = n_list / 16`, `kmeans_iters = 20`).
- Custom CUDA kernels (written as embedded PTX compiled offline from
  `.cu`):
  - `argmin_rows.cu` — given an `(M, K)` score matrix row-major, reduce
    each row to its argmin. One block per row, shared-memory reduction.
    Used during both k-means assignment and coarse cluster selection.
  - No custom kernels needed for distance — SGEMV and SGEMM suffice for
    L2 (derivable from dots + precomputed norms), Cosine (SGEMV + norm
    divide), and DotProduct (SGEMV directly).
- Metric handling:
  - Reuse the per-vector squared-norm cache introduced in
    `CudaVectorStorage` and extend with per-centroid norms.
  - L2 per-cluster search computes `||v - q||^2 = ||v||^2 - 2 v·q +
    ||q||^2` from cached norms.
- Persistence API out of scope for this task — training is in-memory
  only. Serialization lives in a follow-up 0.4.x task.
- Tests under `tests/cuda_ivf_*.rs`:
  - `cuda_ivf_training.rs` — k-means convergence on synthetic clustered
    data, verifying assignment stability and inertia monotonicity.
  - `cuda_ivf_recall.rs` — recall@10 against brute-force on the SIFT1M
    or a synthetic analogue: expect ≥0.95 recall at `nprobe = n_list /
    16`, ≥0.99 at `nprobe = n_list / 4`.
  - `cuda_ivf_search_scaling.rs` — latency curves at 100 K / 1 M / 10 M
    vectors, confirming sub-linear scaling vs brute-force.
- Benchmarks in `benches/cuda_ivf.rs`:
  - IVF build time vs. `n_list`.
  - IVF search latency at varying `nprobe` with recall reported.
  - Head-to-head against brute-force at 1 M vectors.
- Docs:
  - `docs/guides/IVF_GUIDE.md` explaining when to choose IVF vs
    brute-force, how to pick `n_list` and `nprobe`, and how to interpret
    the recall/latency trade-off.
  - `docs/benchmarks/PERFORMANCE.md` updated with an IVF section.
- Examples: `examples/cuda_ivf.rs` walking through train + add + search
  end-to-end.

## Impact

- Affected specs: new `cuda-ivf-index` spec under this task.
- Affected code:
  - new `src/cuda/ivf.rs` (primary implementation)
  - new `src/cuda/kernels/argmin_rows.cu` (+ compiled PTX embedded via
    `include_bytes!`)
  - `src/cuda/mod.rs` (export `CudaIvfIndex`, `IvfConfig`)
  - `src/types.rs` (add `IvfConfig` to public types)
  - `build.rs` (compile argmin PTX and expose via `OUT_DIR`)
  - new `tests/cuda_ivf_*.rs`
  - new `benches/cuda_ivf.rs`
  - new `examples/cuda_ivf.rs`
  - new `docs/guides/IVF_GUIDE.md`
  - `docs/benchmarks/PERFORMANCE.md` updated
- Breaking change: NO. `CudaIvfIndex` is additive; existing
  `CudaVectorStorage` and `CudaContext` APIs are untouched.
- User benefit:
  - Sub-linear search latency at 1 M+ vectors, unblocking production
    Vectorizer workloads on NVIDIA.
  - Tunable recall/latency knob via `nprobe` — users can dial in the
    trade-off their application tolerates.
  - Foundation for Phase 5 quantization: IVF-PQ and IVF-SQ become
    natural extensions of `CudaIvfIndex`.
- Dependencies:
  - `phase3a_add-cuda-backend` must be merged (done in 0.2.0) — this
    builds on `CudaContext` + `CudaVectorStorage`.
  - No blocker on `phase4a_finish-metal-bruteforce-search`; CUDA IVF can
    proceed independently.

## Estimated effort

10–15 dev-days:
- 2–3 days: module scaffolding, `IvfConfig`, device buffer layout.
- 3 days: k-means training (init, Lloyd iterations, convergence).
- 2 days: coarse + refined search pipeline and metric post-processing.
- 2 days: `argmin_rows` kernel + PTX build pipeline.
- 2 days: test suite (training, recall, scaling).
- 1 day: benchmarks, docs, example.
- 1–2 days: bug budget (numerical stability, edge cases).
