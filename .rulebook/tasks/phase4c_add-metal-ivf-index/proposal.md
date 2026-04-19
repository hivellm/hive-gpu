# Proposal: phase4c_add-metal-ivf-index

## Why

Parity with the CUDA IVF work in `phase4b_add-cuda-ivf-index`. The
Vectorizer and any hive-gpu consumer on Apple Silicon needs the same
sub-linear search scaling that IVF brings to CUDA. Without it, Metal users
are stuck at O(N) per query even after the brute-force search on Metal is
made real in `phase4a_finish-metal-bruteforce-search`.

Apple provides native BLAS equivalents through Metal Performance Shaders
(MPS): `MPSMatrixMultiplication` for SGEMM and `MPSMatrixVectorMultiplication`
for SGEMV. These are the same primitives the CUDA IVF path uses through
cuBLAS. Metal also supports custom compute kernels via the Metal Shading
Language (MSL), so the per-row `argmin` kernel the CUDA backend ships as
PTX is written as a `.metal` compute function here.

This task deliberately mirrors `phase4b` in scope and structure so the two
backends evolve in lockstep. Divergence between Metal IVF and CUDA IVF
would create a maintenance tax every time the index gains a feature
(quantization, persistence, new metric).

## What Changes

- New module `src/metal/ivf.rs` exposing `MetalIvfIndex` with an API
  surface identical to `CudaIvfIndex`:
  - `MetalIvfIndex::new(ctx, dimension, metric, IvfConfig { n_list,
    nprobe, ... })`
  - `train(&mut self, sample: &[GpuVector], n_iter: usize)` via k-means++
    init + MPS SGEMM Lloyd iterations.
  - `add_vectors(&mut self, vectors: &[GpuVector])` assigning each vector
    to its nearest centroid.
  - `search(&self, query: &[f32], k: usize) -> Vec<GpuSearchResult>`
    using the coarse SGEMV + per-cluster brute-force pattern.
  - `set_nprobe(&mut self, nprobe: usize)` for query-time tuning.
- Reuse the `IvfConfig` type introduced in the CUDA task — it is
  backend-neutral and lives in `src/types.rs`.
- Metal kernel asset `src/metal/shaders/ivf_argmin.metal` implementing
  per-row argmin over an `(M, K)` score matrix. Compiled alongside the
  existing HNSW metal shaders via `device.newLibraryWithSource`. One
  threadgroup per row, threadgroup memory for the reduction.
- Metal-specific buffer layout mirroring CUDA:
  - Centroids in a single private `MTLBuffer` sized `n_list * dimension *
    sizeof::<f32>`.
  - `cluster_offsets` and `cluster_member_indices` as private `MTLBuffer`s
    of `u32`.
  - Flat vector storage reusing the same layout
    `MetalNativeVectorStorage` uses after `phase4a`.
  - Per-vector and per-centroid squared norms cached on the CPU.
- Metric handling identical to CUDA: SGEMV for dot products, Cosine via
  post-hoc normalization with cached norms, Euclidean via `||v||^2 - 2
  v·q + ||q||^2`.
- Tests under `tests/metal_ivf_*.rs` mirroring the CUDA test matrix:
  - `metal_ivf_training.rs` — k-means convergence on synthetic data.
  - `metal_ivf_recall.rs` — recall@10 targets matching CUDA (≥0.95 at
    `nprobe = n_list / 16`, ≥0.99 at `nprobe = n_list / 4`).
  - `metal_ivf_search_scaling.rs` — latency curves at 100 K, 1 M vectors
    demonstrating sub-linear scaling.
- `benches/metal_ivf.rs` recording build time, search latency vs
  `nprobe`, and head-to-head against brute-force.
- Documentation updates:
  - Extend `docs/guides/IVF_GUIDE.md` (created by `phase4b`) with an
    Apple Silicon section covering Metal-specific knobs.
  - `docs/benchmarks/PERFORMANCE.md` gains a Metal IVF table next to the
    CUDA numbers.
- Example: `examples/metal_ivf.rs` walking through train + add + search.

## Impact

- Affected specs: new `metal-ivf-index` spec under this task.
- Affected code:
  - new `src/metal/ivf.rs`
  - new `src/metal/shaders/ivf_argmin.metal` (loaded via existing Metal
    library plumbing)
  - `src/metal/mod.rs` (export `MetalIvfIndex`)
  - `src/types.rs` — IvfConfig is already in place from `phase4b`
  - new `tests/metal_ivf_*.rs`
  - new `benches/metal_ivf.rs`
  - new `examples/metal_ivf.rs`
  - `docs/guides/IVF_GUIDE.md` (section added)
  - `docs/benchmarks/PERFORMANCE.md` updated
- Breaking change: NO. `MetalIvfIndex` is additive; existing
  `MetalNativeVectorStorage` and `MetalNativeContext` APIs are untouched.
- User benefit:
  - Parity with CUDA: sub-linear search latency on Apple Silicon at 1 M+
    vectors.
  - Consistent mental model and configuration surface across Metal and
    CUDA — an `IvfConfig` authored once works on both.
  - Unblocks the Vectorizer on macOS in production scenarios that
    outgrow brute-force.
- Dependencies:
  - **`phase4a_finish-metal-bruteforce-search` must ship first.** The
    per-cluster refined search reuses the MPS SGEMV path introduced in
    that task. Until brute-force on Metal is real, IVF on Metal is not
    testable.
  - **`phase4b_add-cuda-ivf-index` should ship first as well** for the
    shared `IvfConfig` type, the `IVF_GUIDE.md` doc scaffold, and to
    establish the test/benchmark shape this task mirrors. If scheduling
    requires parallelism, `IvfConfig` can be landed in a small shared
    PR before either backend begins implementation.

## Estimated effort

10–15 dev-days, matching the CUDA task:

- 2 days: module scaffolding, buffer layout, library load for the new
  `ivf_argmin.metal` shader.
- 3 days: k-means training loop in Rust + MPS SGEMM + Metal argmin
  kernel.
- 2 days: coarse + refined search pipeline.
- 2 days: test suite (training, recall, scaling).
- 1 day: benchmarks, docs, example.
- 2–3 days: bug budget (numerical stability, unified-memory quirks,
  correct threadgroup sizing across M-series generations).
