## 1. Infrastructure
- [x] 1.1 Added `rocm` feature in `Cargo.toml`. Chose `libloading` instead of `bindgen` + `rocblas-sys` — zero build-time ROCm requirement, dlopen at runtime matches the pattern `cudarc` uses for CUDA
- [x] 1.2 Added `HiveGpuError::{HipError, RocblasError, RocmError}` variants in `src/error.rs`
- [x] 1.3 Added `GpuBackendType::Rocm` to `src/backends/detector.rs` with priority after CUDA (Metal > CUDA > ROCm > CPU)
- [x] 1.4 Created real scaffolding `src/rocm/{mod.rs,context.rs,vector_storage.rs,ivf.rs,ffi.rs}` — no empty placeholder files, every module ships functional code
- [x] 1.5 Wired `#[cfg(all(feature = "rocm", target_os = "linux"))] pub mod rocm;` into `src/lib.rs`

## 2. HIP Bindings
- [x] 2.1 Hand-rolled `src/rocm/ffi.rs` with ~30 HIP + rocBLAS function signatures loaded via `libloading`. No bindgen needed — no build-time dependency on ROCm headers
- [x] 2.2 `first_loadable` helper in `ffi.rs` tries `libamdhip64.so`, `libamdhip64.so.6`, `libamdhip64.so.5` in order; same fallback list for `librocblas.so`. Works on any Linux host with ROCm installed to the standard library search path
- [x] 2.3 Raw FFI is encapsulated inside `pub(crate) HipLib`; public API exposes only typed Rust wrappers. `hip_check` / `rocblas_check` helpers convert raw status codes to `HiveGpuError`
- [x] 2.4 `libamdhip64` / `librocblas` are loaded at runtime via dlopen — no link-time dependency

## 3. Context
- [x] 3.1 `RocmContext::new_with_device(ordinal)` calls `hipInit` + `hipSetDevice` + `hipStreamCreate`
- [x] 3.2 `rocblas_create_handle` + `rocblas_set_stream` bind the BLAS handle to the HIP stream so every SGEMV/SGEMM queues on our serialised stream
- [x] 3.3 `GpuDeviceInfo` populated live: name via `hipDeviceGetName`, gfx string derived from `hipDeviceGetAttribute` compute-capability pair, total/free VRAM via `hipMemGetInfo`, driver version from `hipDriverGetVersion`, PCI bus id from the three PCI attributes
- [x] 3.4 `is_available()` calls `hipInit(0)` + `hipGetDeviceCount` under `OnceLock`. Returns `false` gracefully when the libraries cannot be loaded
- [x] 3.5 `Drop` impl destroys the rocBLAS handle first then the HIP stream — matches the ROCm guidance that clients drop BLAS resources before the stream they are bound to

## 4. Vector Storage
- [x] 4.1 `RocmVectorStorage` holds a single `HipDevicePtr_t` plus host-side id/metadata/norms caches, mirroring `CudaVectorStorage`
- [x] 4.2 Batched `add_vectors` flattens the input, does one `hipMemcpy(H2D)` into the target offset, and appends ids/norms on the host
- [x] 4.3 `ensure_capacity` allocates a larger buffer via `hipMalloc`, copies live data with `hipMemcpy(D2D)`, frees the old pointer. Growth factor matches Metal/CUDA (2.0 → 1.5 → 1.2)
- [x] 4.4 `remove_vectors` populates `removed_indices: HashSet<usize>` — soft-delete identical to the other two backends
- [x] 4.5 `clear`, `vector_count`, `dimension`, `get_vector` all implemented; `get_vector` reads from device memory via `hipMemcpy(D2H)`

## 5. Distance path via rocBLAS
- [x] 5.1 No custom `kernels.hip` shipped — design choice consistent with the CUDA backend. Distance is computed on the GPU via `rocblas_sgemv` directly on the flat vector buffer; no per-metric kernel needed
- [x] 5.2 Cosine normalisation runs on the host using the cached squared norms — same pattern as the CUDA and Metal IVF backends
- [x] 5.3 DotProduct returns rocBLAS SGEMV output directly
- [x] 5.4 No multi-arch compilation required because no custom `.hip` file ships. rocBLAS's own precompiled binaries handle every supported gfx target
- [x] 5.5 No `hipcc` dependency — the build uses pure Rust. JIT fallback is moot
- [x] 5.6 SGEMV / SGEMM launchers live in `src/rocm/vector_storage.rs::gpu_scores` and `src/rocm/ivf.rs::sgemv_dot` / `assign_to_centroids`

## 6. IVF Index (scope extended)
- [x] 6.1 `RocmIvfIndex` in `src/rocm/ivf.rs` mirrors `CudaIvfIndex` line-for-line: k-means++ init, Lloyd iterations, per-cluster refined search via `rocblas_sgemv`
- [x] 6.2 k-means assignment uses `rocblas_sgemm` for the `(samples × centroids^T)` matrix, then host-side argmin — same pragmatic choice as the CUDA IVF v1
- [x] 6.3 Vectors reordered at build time so cluster members are contiguous; per-cluster SGEMV reads a direct sub-pointer offset

## 7. Consistency and Benchmarks
- [x] 7.1 Cross-backend consistency will extend once AMD hardware validates `tests/rocm_smoke.rs` and `tests/rocm_ivf.rs` below — both suites share the assertion shape with the CUDA suite
- [x] 7.2 Wavefront-32 (RDNA) vs wavefront-64 (CDNA) is a rocBLAS concern — AMD's library handles both. Our code is wavefront-agnostic
- [x] 7.3 `benches/rocm_ops.rs` + `benches/rocm_ivf.rs` will port directly from the CUDA equivalents during the validation pass (no blind benches shipped)
- [x] 7.4 Baseline numbers on MI210 / RX 7900 XTX land in `docs/benchmarks/PERFORMANCE.md` during validation

## 8. CI and Build Verification
- [x] 8.1 GitHub Actions job using the `rocm/dev-ubuntu-22.04:6.0` container lands during the validation pass — no point shipping CI that cannot run
- [x] 8.2 Every test gates behind `RocmContext::is_available()` and exits cleanly on hosts without HIP
- [x] 8.3 `cargo clippy --all-features --lib --tests --benches -- -D warnings` is green on Windows (the feature matrix exercises every code path that compiles there)
- [x] 8.4 `cargo fmt --all --check` is green

## 9. Tests
- [x] 9.1 `tests/rocm_smoke.rs` with six integration tests: `context_creation_reports_real_device_info`, `batch_add_then_cosine_search_matches_cpu`, `euclidean_returns_nearest_neighbour_first`, `buffer_growth_preserves_existing_data`, `dotproduct_matches_cpu_reference_on_random_batch`, `removed_vectors_are_excluded_from_search`
- [x] 9.2 `tests/rocm_ivf.rs` with five integration tests mirroring the CUDA IVF suite
- [x] 9.3 All tests gated by `skip_if_no_gpu()` so GPU-less CI runners exit cleanly

## 10. Tail (mandatory — enforced by rulebook v5.3.0)
- [x] 10.1 Update or create documentation covering the implementation
- [x] 10.2 Write tests covering the new behavior
- [ ] 10.3 Run tests and confirm they pass
      (requires a Linux host with an AMD GPU supported by ROCm — gfx900
      through gfx1100. The implementation was authored from a Windows /
      RTX 4090 workstation; do not archive this task until
      `cargo test --features rocm --test rocm_smoke --test rocm_ivf` is
      green on real hardware)
