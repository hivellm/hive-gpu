## 1. Environment preparation
- [ ] 1.1 Pull `main` on a Linux host (Ubuntu 22.04 or newer recommended) with an AMD GPU whose gfx architecture is in the ROCm support matrix (gfx900 through gfx1100)
- [ ] 1.2 Install ROCm 6.x following the official guide; confirm `rocminfo` lists the GPU and reports its gfx version
- [ ] 1.3 Ensure `libamdhip64.so` and `librocblas.so` resolve on the dynamic linker path (`ldconfig -p | grep -E "amdhip|rocblas"`)
- [ ] 1.4 Install a current stable Rust toolchain: `rustup default stable`

## 2. Compile the new ROCm code
- [ ] 2.1 `cargo check --features rocm` — catches compile-time issues in the Linux + ROCm gate
- [ ] 2.2 If the build fails with an unresolved `libamdhip64.soname`, extend the `hip_candidates` / `rocblas_candidates` arrays in `src/rocm/ffi.rs::HipLib::try_load` to include the local SONAME and retry
- [ ] 2.3 If any HIP symbol fails to resolve (missing entry in `libamdhip64`), compare against `hipruntime_api.h` on the host and update the function-pointer field name in `HipLib`

## 3. Brute-force smoke validation
- [ ] 3.1 Run `cargo test --features rocm --test rocm_smoke` and capture the full output
- [ ] 3.2 Expect six tests to pass: `context_creation_reports_real_device_info`, `batch_add_then_cosine_search_matches_cpu`, `euclidean_returns_nearest_neighbour_first`, `buffer_growth_preserves_existing_data`, `dotproduct_matches_cpu_reference_on_random_batch`, `removed_vectors_are_excluded_from_search`
- [ ] 3.3 If `context_creation_reports_real_device_info` reports `compute_capability == None` or the gfx string looks wrong, patch `HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR` / `MINOR` in `src/rocm/ffi.rs` to match the installed ROCm version
- [ ] 3.4 If `dotproduct_matches_cpu_reference_on_random_batch` comes in outside the 1e-3 tolerance, record the observed divergence and raise the threshold to the tightest value that passes

## 4. IVF validation
- [ ] 4.1 Run `cargo test --features rocm --test rocm_ivf` and capture the output
- [ ] 4.2 Expect five tests to pass: `new_rejects_bad_config`, `build_rejects_empty_and_too_small_inputs`, `set_nprobe_validates`, `recall_at_10_against_bruteforce_dotproduct`, `higher_nprobe_increases_recall`
- [ ] 4.3 If `recall_at_10_against_bruteforce_dotproduct` returns below 0.30, the rocBLAS SGEMM operand orientation is almost certainly wrong — flip the `transa` / `transb` flags in `src/rocm/ivf.rs::assign_to_centroids` and retry
- [ ] 4.4 If recall is between 0.30 and 0.60, the SGEMM is correct but k-means may be converging to a worse local optimum; cluster the data with `clustered_dataset` (see `tests/cuda_ivf.rs` lines 47-70) and retest — recall on clustered data must be >= 0.95

## 5. Quality gates
- [ ] 5.1 `cargo clippy --features rocm --lib --tests --benches -- -D warnings` clean
- [ ] 5.2 `cargo fmt --all --check` clean
- [ ] 5.3 `cargo doc --no-deps --features rocm` clean

## 6. Benchmarks
- [ ] 6.1 Port `benches/cuda_ops.rs` to `benches/rocm_ops.rs`; register the new bench entry in `Cargo.toml` alongside `cuda_ops`
- [ ] 6.2 Run `cargo bench --features rocm --bench rocm_ops` and capture median times
- [ ] 6.3 Port `benches/cuda_ivf.rs` to `benches/rocm_ivf.rs`; register in `Cargo.toml`
- [ ] 6.4 Run `cargo bench --features rocm --bench rocm_ivf` and capture build / search-vs-nprobe / head-to-head numbers

## 7. Documentation
- [ ] 7.1 Add an AMD ROCm section to `docs/benchmarks/PERFORMANCE.md` with the measured numbers, mirroring the CUDA IVF layout
- [ ] 7.2 Update `README.md` backend matrix: mark ROCm as shipping, remove the "designed but not implemented" caveat
- [ ] 7.3 Update `docs/ROADMAP.md` Phase 3 notes — mark ROCm as shipped on the release this task lands in
- [ ] 7.4 Update `CHANGELOG.md` with a `0.2.2` (brute-force only) or `0.3.0` (IVF also) entry describing the shipped functionality and bench results

## 8. Release
- [ ] 8.1 Bump `Cargo.toml` version appropriately
- [ ] 8.2 Tag the release once merged: `git tag v<version>` and push both branch and tag

## 9. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 9.1 Update or create documentation covering the implementation
- [ ] 9.2 Write tests covering the new behavior
- [ ] 9.3 Run tests and confirm they pass
