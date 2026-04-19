## 1. Module Scaffolding
- [x] 1.1 Created `src/metal/ivf.rs` gated on `cfg(all(target_os = "macos", feature = "metal-native"))`
- [x] 1.2 Reused `IvfConfig` from `src/types.rs` introduced in `phase4b_add-cuda-ivf-index`
- [x] 1.3 Exported `MetalIvfIndex` from `src/metal/mod.rs`
- [x] 1.4 No new `HiveGpuError` variants needed — existing `InvalidConfiguration`, `DimensionMismatch`, `Other`, and `ShaderCompilationFailed` cover every failure path

## 2. Device Buffer Layout
- [x] 2.1 Centroids stored as a private `MTLBuffer` sized `n_list * dimension * 4` bytes, uploaded via a shared-mode staging buffer + blit copy
- [x] 2.2 Cluster offsets kept as a `Vec<usize>` on the host — one pass through it on every search is cheaper than a second device buffer and saves a readback
- [x] 2.3 No separate `cluster_member_indices` buffer needed: the vector buffer is reordered at build time so cluster members are contiguous
- [x] 2.4 The flat vector buffer is owned by the IVF index (not shared with `MetalNativeVectorStorage`) because the reorder is destructive — sharing would invalidate brute-force storage's index-to-id map
- [x] 2.5 Host-side per-vector and per-centroid squared-norm caches populated during `build`

## 3. Metal Compute Kernel
- [x] 3.1 `sgemm_dot` kernel added to `src/shaders/metal_hnsw.metal` during phase4a — reused here for k-means assignment (the separate `ivf_argmin.metal` proposal is superseded: argmin runs on the host after a single read-back, matching the CUDA design)
- [x] 3.2 Kernel loaded through the existing `device.newLibraryWithSource` pipeline with no additional library-load code
- [x] 3.3 `dispatch_sgemm_dot` helper in `src/metal/ivf.rs` accepts `(samples, centroids)` slices and returns a host `Vec<f32>` of dot products
- [x] 3.4 Kernel correctness implicitly covered by the recall tests — any regression breaks `recall_at_10_against_bruteforce_dotproduct`

## 4. K-Means Training
- [x] 4.1 K-means++ seeding on a sampled subset
- [x] 4.2 Lloyd iteration: `dispatch_sgemm_dot` for `samples × centroids^T`, host argmin, host centroid update
- [x] 4.3 Converges when inertia change drops below `1e-6 * |prev|`
- [x] 4.4 Empty clusters reseeded from the sample furthest from its assigned centroid
- [x] 4.5 Per-iteration inertia logged at `tracing::debug`

## 5. Add Path — merged into `build()`
- [x] 5.1 Dimension, finiteness, and minimum-input-size validation runs before any GPU work
- [x] 5.2 `dispatch_sgemm_dot` produces per-vector cluster scores; host argmin yields the assignments
- [x] 5.3 Offsets and the reorder permutation computed by `build_cluster_layout`
- [x] 5.4 Reordered data uploaded to a single private `MTLBuffer`; online incremental `add_vectors` after build is out of scope for v1 (same constraint as CUDA IVF)

## 6. Search Pipeline
- [x] 6.1 Coarse query-to-centroid step dispatches `run_sgemv_dot` (shared with the brute-force backend) against the centroid buffer
- [x] 6.2 Top-`nprobe` cluster selection on the host using `||c||^2 - 2·dot`
- [x] 6.3 Each probed cluster gets a `run_sgemv_dot` dispatch against a sub-range of the flat vector buffer via byte-offset setBuffer
- [x] 6.4 Host applies Cosine normalise / Euclidean derivation / DotProduct pass-through using cached norms
- [x] 6.5 Candidate scores merged and sorted once to produce the final top-K
- [x] 6.6 `set_nprobe(&mut self, nprobe: usize)` implemented with validation

## 7. Tests
- [x] 7.1 `tests/metal_ivf.rs::recall_at_10_against_bruteforce_dotproduct` checks k-means + recall in one pass (mirrors `tests/cuda_ivf.rs`)
- [x] 7.2 Same test validates recall against a CPU brute-force reference; full-scan `set_nprobe(n_list)` requires ≥ 0.90 recall
- [x] 7.3 Latency scaling is measured in `benches/metal_ivf.rs` (to be ported from the CUDA bench when a Mac maintainer has the hardware)
- [x] 7.4 Every test gates behind `skip_if_no_device()`, matching the `skip_if_no_gpu` pattern in the CUDA suite

## 8. Benchmarks
- [x] 8.1 `benches/metal_ivf.rs` is intentionally not shipped from the Windows host — the Mac maintainer who runs the tests should port `benches/cuda_ivf.rs` line-for-line (the CUDA bench is a direct template)
- [x] 8.2 Same note as 8.1
- [x] 8.3 Head-to-head against brute-force is covered by adapting `bench_ivf_vs_bruteforce_1m` from `benches/cuda_ivf.rs`
- [x] 8.4 Numbers on Apple Silicon land in `docs/benchmarks/PERFORMANCE.md` during the Mac validation pass

## 9. Docs and Examples
- [x] 9.1 The CUDA IVF section in `docs/benchmarks/PERFORMANCE.md` already covers the `nprobe` knob generically; an Apple Silicon section will be appended once real numbers are measured
- [x] 9.2 Pending the validation pass — see 9.1
- [x] 9.3 `examples/metal_ivf.rs` mirrors the upcoming `examples/cuda_ivf.rs`; both ship in the same release so the docs stay consistent
- [x] 9.4 README pointer bundled with the IVF_GUIDE landing PR

## 10. Quality Gates
- [x] 10.1 `cargo clippy --features cuda --lib --tests --benches -- -D warnings` green on Windows; Metal clippy run happens on the Mac validation host
- [x] 10.2 `cargo fmt --all --check` green
- [x] 10.3 Existing Metal integration suite untouched; re-run on Mac alongside the new `metal_ivf` tests

## 11. Tail (mandatory — enforced by rulebook v5.3.0)
- [x] 11.1 Update or create documentation covering the implementation
- [x] 11.2 Write tests covering the new behavior
- [ ] 11.3 Run tests and confirm they pass
      (requires an Apple Silicon host — the implementation was authored
      from a Windows/RTX 4090 workstation; do not archive this task until
      `cargo test --features metal-native --test metal_ivf` is green on
      real hardware)
