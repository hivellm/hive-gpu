## 1. Module Scaffolding
- [ ] 1.1 Create `src/cuda/ivf.rs` gated on `cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))`
- [ ] 1.2 Add `IvfConfig` struct in `src/types.rs` with fields `n_list`, `nprobe`, `training_sample_size`, `kmeans_iters`
- [ ] 1.3 Export `CudaIvfIndex` and `IvfConfig` from `src/cuda/mod.rs` and re-export from the crate root
- [ ] 1.4 Add `HiveGpuError::IvfTrainingError(String)` and `IvfEmptyError` variants if needed

## 2. Device Buffer Layout
- [ ] 2.1 Allocate centroids as a single contiguous `CudaSlice<f32>` of shape `(n_list, dimension)`
- [ ] 2.2 Allocate `cluster_offsets: CudaSlice<u32>` of length `n_list + 1`
- [ ] 2.3 Allocate `cluster_member_indices: CudaSlice<u32>` of length equal to stored vector count
- [ ] 2.4 Store the flat vector matrix in the same layout as `CudaVectorStorage`
- [ ] 2.5 Maintain host-side vector squared-norm and centroid squared-norm caches

## 3. argmin_rows CUDA Kernel
- [ ] 3.1 Write `src/cuda/kernels/argmin_rows.cu` computing `argmin` across rows of an `(M, K)` score matrix
- [ ] 3.2 Compile offline to multi-SM PTX (sm_70 through sm_90) via `nvcc -ptx`
- [ ] 3.3 Embed the PTX via `include_bytes!` through `OUT_DIR`
- [ ] 3.4 Add a Rust launcher in `src/cuda/kernels.rs` that takes a `CudaSlice<f32>` and writes a `CudaSlice<u32>`
- [ ] 3.5 Cover the kernel with a unit test comparing against a CPU reference

## 4. K-Means Training
- [ ] 4.1 Implement k-means++ seeding on a sampled subset of the training vectors
- [ ] 4.2 Implement one Lloyd iteration: SGEMM of `samples × centroids^T`, run `argmin_rows`, atomic accumulate sums + counts into new centroids
- [ ] 4.3 Iterate until `kmeans_iters` or inertia change below 1e-6
- [ ] 4.4 Handle empty clusters by reseeding from the furthest outlier
- [ ] 4.5 Produce a training-status report (final inertia, per-iteration deltas)

## 5. Add Path
- [ ] 5.1 `add_vectors` validates dimension and ID uniqueness as the brute-force backend does
- [ ] 5.2 Compute query-to-centroid SGEMV once per batch; run `argmin_rows` to get each vector's cluster id
- [ ] 5.3 Rebuild `cluster_offsets` and `cluster_member_indices` from the updated assignment list
- [ ] 5.4 Append vectors to the flat storage buffer using the same adaptive growth pattern as brute-force

## 6. Search Pipeline
- [ ] 6.1 Compute query-to-centroid SGEMV to get coarse scores
- [ ] 6.2 Select top-`nprobe` clusters on the CPU via `argpartition` (O(n_list))
- [ ] 6.3 For each probed cluster, dispatch cuBLAS SGEMV over the cluster's contiguous subrange using `cluster_offsets`
- [ ] 6.4 Apply metric post-processing (Cosine normalise, Euclidean derive ||v-q||^2)
- [ ] 6.5 Merge per-cluster top-K on the CPU into the global top-K and remap back to global indices
- [ ] 6.6 Expose `set_nprobe(&mut self, nprobe: usize)` for query-time tuning

## 7. Tests
- [ ] 7.1 `tests/cuda_ivf_training.rs` — k-means convergence on synthetic clustered data; verify inertia is monotonically non-increasing
- [ ] 7.2 `tests/cuda_ivf_recall.rs` — recall@10 against brute-force ≥ 0.95 at `nprobe = n_list / 16`, ≥ 0.99 at `nprobe = n_list / 4`
- [ ] 7.3 `tests/cuda_ivf_search_scaling.rs` — search latency at 100 K, 1 M, and 10 M vectors demonstrating sub-linear growth
- [ ] 7.4 Gate every test behind `CudaContext::is_available()` so runners without a GPU exit cleanly

## 8. Benchmarks
- [ ] 8.1 Add `benches/cuda_ivf.rs` measuring build time vs `n_list`
- [ ] 8.2 Measure search latency at several `(nprobe, n_list)` combinations with recall reported
- [ ] 8.3 Head-to-head against brute-force at 1 M vectors
- [ ] 8.4 Capture numbers on the reference RTX 4090 host

## 9. Docs and Examples
- [ ] 9.1 Write `docs/guides/IVF_GUIDE.md` covering when to choose IVF, how to pick `n_list` and `nprobe`, and recall/latency trade-off
- [ ] 9.2 Add an IVF section to `docs/benchmarks/PERFORMANCE.md`
- [ ] 9.3 Ship `examples/cuda_ivf.rs` walking through train + add + search
- [ ] 9.4 Update `README.md` with a short IVF pointer

## 10. Quality Gates
- [ ] 10.1 `cargo clippy --features cuda --lib --tests --benches -- -D warnings` green
- [ ] 10.2 `cargo fmt --all --check` green
- [ ] 10.3 CI workflow `.github/workflows/cuda-build.yml` runs the IVF suite when a GPU is present, exits cleanly otherwise

## 11. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 11.1 Update or create documentation covering the implementation
- [ ] 11.2 Write tests covering the new behavior
- [ ] 11.3 Run tests and confirm they pass
