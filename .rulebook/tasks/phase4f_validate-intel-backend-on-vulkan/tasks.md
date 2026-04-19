## 1. Environment preparation
- [ ] 1.1 Pull `main` on a Linux or Windows host with a Vulkan 1.2-capable GPU — Intel Arc / Battlemage (vendor id `0x8086`) preferred; any other Vulkan GPU under `HIVE_GPU_VULKAN_UNIVERSAL=1` acceptable as secondary target
- [ ] 1.2 Install the latest Intel / vendor graphics driver and verify with `vulkaninfo --summary` that at least one device reports `apiVersion >= 1.2`
- [ ] 1.3 Install the LunarG Vulkan SDK so that `VK_LAYER_KHRONOS_validation` is available at runtime (`vulkaninfo | grep VK_LAYER_KHRONOS_validation`)
- [ ] 1.4 Install a current stable Rust toolchain: `rustup default stable`

## 2. Compile the new Intel code
- [ ] 2.1 `cargo check --features intel` — catches compile-time issues in the `intel`-gated modules and the `build.rs` shader compile step
- [ ] 2.2 If `build.rs` fails during WGSL parse or SPIR-V emit, read the naga error message and patch `src/intel/shaders/sgemv_dot.wgsl` or `sgemm_dot.wgsl` (most likely cause: a naga validator rejects a capability we did not enable in `ValidationFlags::all()`)
- [ ] 2.3 If linking fails with unresolved `vkGetInstanceProcAddr` or other Vulkan loader symbols, confirm the Vulkan loader is reachable (`libvulkan.so.1` on Linux, `vulkan-1.dll` on Windows) and rebuild

## 3. Brute-force smoke validation
- [ ] 3.1 Run `cargo test --features intel --test intel_smoke` and capture the full output (include stderr — the suite prints diagnostic `eprintln!` lines)
- [ ] 3.2 Expect five tests to pass: `context_creation_reports_real_device_info`, `batch_add_then_cosine_search_matches_cpu`, `euclidean_ranks_nearest_first`, `dotproduct_matches_cpu_reference_on_random_batch`, `removed_vectors_are_excluded_from_search`
- [ ] 3.3 If `context_creation_reports_real_device_info` fails because no Intel device is found, rerun with `HIVE_GPU_VULKAN_UNIVERSAL=1 cargo test --features intel --test intel_smoke` and confirm fallback-mode selection works
- [ ] 3.4 If `dotproduct_matches_cpu_reference_on_random_batch` comes in outside the 1e-3 tolerance, record the observed divergence and raise the threshold to the tightest value that passes — Intel iGPUs occasionally use fp16 accumulation paths

## 4. IVF validation
- [ ] 4.1 Run `cargo test --features intel --test intel_ivf` and capture the output
- [ ] 4.2 Expect four tests to pass: `new_rejects_bad_config`, `build_rejects_empty_and_too_small_inputs`, `recall_at_10_against_bruteforce_dotproduct`, `higher_nprobe_increases_recall`
- [ ] 4.3 If `recall_at_10_against_bruteforce_dotproduct` returns below 0.30, the SGEMM operand layout in `src/intel/ivf.rs::assign_to_centroids` is wrong for the row-major WGSL kernel — inspect index formulas and flip operand order
- [ ] 4.4 If recall is between 0.30 and 0.60, the SGEMM is correct but k-means may be converging to a worse local optimum; cluster the data with `clustered_dataset` (see `tests/cuda_ivf.rs` lines 47-70), retest — recall on clustered data must be >= 0.95

## 5. Validation-layer pass
- [ ] 5.1 Enable the Khronos validation layer by setting `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` (Linux) or the equivalent env on Windows, then rerun `cargo test --features intel`
- [ ] 5.2 Fix every validation error surfaced — the most common classes are: missing pipeline barrier between write and read, wrong descriptor binding order, push-constant size mismatch. Each fix belongs in `src/intel/vector_storage.rs` or `src/intel/context.rs`

## 6. Quality gates
- [ ] 6.1 `cargo clippy --features intel --lib --tests --benches -- -D warnings` clean
- [ ] 6.2 `cargo fmt --all --check` clean
- [ ] 6.3 `cargo doc --no-deps --features intel` clean

## 7. Benchmarks
- [ ] 7.1 Port `benches/cuda_ops.rs` to `benches/intel_ops.rs`; register the new bench entry in `Cargo.toml` alongside `cuda_ops`
- [ ] 7.2 Run `cargo bench --features intel --bench intel_ops` and capture median times
- [ ] 7.3 Port `benches/cuda_ivf.rs` to `benches/intel_ivf.rs`; register in `Cargo.toml`
- [ ] 7.4 Run `cargo bench --features intel --bench intel_ivf` and capture build / search-vs-nprobe / head-to-head numbers

## 8. Documentation
- [ ] 8.1 Add an Intel / Vulkan section to `docs/benchmarks/PERFORMANCE.md` with the measured numbers, mirroring the CUDA IVF layout; note whether the host was native Intel Arc or a universal-fallback GPU
- [ ] 8.2 Update `README.md` backend matrix: mark Intel as shipping, remove the "designed but not implemented" caveat
- [ ] 8.3 Update `docs/ROADMAP.md` Phase 3 notes — mark Intel as shipped on the release this task lands in
- [ ] 8.4 Update `CHANGELOG.md` with a `0.2.3` (brute-force only) or `0.3.0` (IVF also) entry describing the shipped functionality and bench results

## 9. Release
- [ ] 9.1 Bump `Cargo.toml` version appropriately
- [ ] 9.2 Tag the release once merged: `git tag v<version>` and push both branch and tag

## 10. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 10.1 Update or create documentation covering the implementation
- [ ] 10.2 Write tests covering the new behavior
- [ ] 10.3 Run tests and confirm they pass
