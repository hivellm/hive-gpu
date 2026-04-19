## 1. Module Scaffolding
- [ ] 1.1 Create `src/metal/ivf.rs` gated on `cfg(all(target_os = "macos", feature = "metal-native"))`
- [ ] 1.2 Reuse `IvfConfig` from `src/types.rs` introduced in `phase4b_add-cuda-ivf-index`
- [ ] 1.3 Export `MetalIvfIndex` from `src/metal/mod.rs` and re-export from the crate root
- [ ] 1.4 Add any Metal-specific variants to `HiveGpuError` that CUDA did not need (e.g. MPS descriptor construction failures)

## 2. Device Buffer Layout
- [ ] 2.1 Allocate centroids as a private `MTLBuffer` sized `n_list * dimension * 4` bytes
- [ ] 2.2 Allocate `cluster_offsets_buf: MTLBuffer` of `(n_list + 1) * 4` bytes
- [ ] 2.3 Allocate `cluster_member_indices_buf: MTLBuffer` sized `total_vectors * 4` bytes
- [ ] 2.4 Share the flat vector storage buffer with `MetalNativeVectorStorage` after phase4a lands
- [ ] 2.5 Maintain host-side vector and centroid squared-norm caches

## 3. Metal Argmin Kernel
- [ ] 3.1 Write `src/metal/shaders/ivf_argmin.metal` implementing per-row argmin over an `(M, K)` score matrix
- [ ] 3.2 Load the kernel through the existing `device.newLibraryWithSource` plumbing in `MetalNativeContext`
- [ ] 3.3 Add a Rust launcher that takes a scores `MTLBuffer` and an output `MTLBuffer<u32>`
- [ ] 3.4 Cover the kernel with a unit test comparing against a CPU reference on random input

## 4. K-Means Training
- [ ] 4.1 Implement k-means++ seeding on a sampled subset of the training vectors
- [ ] 4.2 One Lloyd iteration: MPS SGEMM of `samples × centroids^T`, run the argmin kernel, accumulate sums + counts into new centroids via a dedicated compute shader
- [ ] 4.3 Iterate until `kmeans_iters` or inertia change below 1e-6
- [ ] 4.4 Handle empty clusters by reseeding from the furthest outlier
- [ ] 4.5 Produce a training-status report mirroring the CUDA version

## 5. Add Path
- [ ] 5.1 `add_vectors` validates dimension and ID uniqueness (reuse the same validator shape as the brute-force backend)
- [ ] 5.2 Compute vector-to-centroid MPS SGEMM once per batch; run the argmin kernel to produce cluster ids
- [ ] 5.3 Rebuild `cluster_offsets_buf` and `cluster_member_indices_buf` from the updated assignments
- [ ] 5.4 Append vectors to the flat storage buffer using the same adaptive growth pattern as brute-force

## 6. Search Pipeline
- [ ] 6.1 Compute query-to-centroid MPS SGEMV to get coarse scores
- [ ] 6.2 Select top-`nprobe` clusters on the CPU via `argpartition` (O(n_list))
- [ ] 6.3 For each probed cluster, dispatch MPS SGEMV over the cluster's contiguous subrange using `cluster_offsets`
- [ ] 6.4 Apply metric post-processing (Cosine normalise, Euclidean derive ||v-q||^2)
- [ ] 6.5 Merge per-cluster top-K on the CPU into the global top-K and remap to global indices
- [ ] 6.6 Expose `set_nprobe(&mut self, nprobe: usize)`

## 7. Tests
- [ ] 7.1 `tests/metal_ivf_training.rs` — k-means convergence on synthetic clustered data; verify inertia is monotonically non-increasing
- [ ] 7.2 `tests/metal_ivf_recall.rs` — recall@10 ≥ 0.95 at `nprobe = n_list / 16`, ≥ 0.99 at `nprobe = n_list / 4`
- [ ] 7.3 `tests/metal_ivf_search_scaling.rs` — search latency at 100 K and 1 M vectors showing sub-linear growth
- [ ] 7.4 Gate every test behind a graceful exit when Metal is unavailable, matching the pattern used by the CUDA suite

## 8. Benchmarks
- [ ] 8.1 Add `benches/metal_ivf.rs` measuring build time vs `n_list`
- [ ] 8.2 Measure search latency at several `(nprobe, n_list)` combinations with recall reported
- [ ] 8.3 Head-to-head against brute-force at 1 M vectors
- [ ] 8.4 Capture numbers on an Apple Silicon host

## 9. Docs and Examples
- [ ] 9.1 Extend `docs/guides/IVF_GUIDE.md` with an Apple Silicon section (unified memory considerations, choice of `nprobe`)
- [ ] 9.2 Add Metal IVF numbers to `docs/benchmarks/PERFORMANCE.md`
- [ ] 9.3 Ship `examples/metal_ivf.rs` walking through train + add + search
- [ ] 9.4 Update `README.md` to mention Metal IVF parity

## 10. Quality Gates
- [ ] 10.1 `cargo clippy --features metal-native --lib --tests --benches -- -D warnings` green
- [ ] 10.2 `cargo fmt --all --check` green
- [ ] 10.3 Existing Metal integration suite still passes end-to-end

## 11. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 11.1 Update or create documentation covering the implementation
- [ ] 11.2 Write tests covering the new behavior
- [ ] 11.3 Run tests and confirm they pass
