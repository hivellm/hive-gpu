## 1. Infrastructure
- [ ] 1.1 Add `rocm` feature in `Cargo.toml` with `rocblas-sys` and `bindgen` optional deps
- [ ] 1.2 Add `HiveGpuError::{HipError, RocblasError, RocmError}` variants in `src/error.rs`
- [ ] 1.3 Add `GpuBackendType::Rocm` to `src/backends/detector.rs` with priority after CUDA
- [ ] 1.4 Create empty scaffolding `src/rocm/{mod.rs,context.rs,vector_storage.rs,kernels.rs,vram_monitor.rs}`
- [ ] 1.5 Wire `#[cfg(feature = "rocm")] pub mod rocm;` into `src/lib.rs`

## 2. HIP Bindings
- [ ] 2.1 Add bindgen invocation in `build.rs` against `$ROCM_PATH/include/hip/`
- [ ] 2.2 Detect ROCM_PATH / ROCM_HOME with `/opt/rocm` fallback in `build.rs`
- [ ] 2.3 Encapsulate raw FFI in `src/rocm/ffi.rs` so public API never exposes raw pointers
- [ ] 2.4 Link `libamdhip64` and `librocblas` system libraries

## 3. Context
- [ ] 3.1 Implement `RocmContext::new` with `hipSetDevice` + `hipStreamCreate`
- [ ] 3.2 Bind `rocblas_create_handle` to the HIP stream
- [ ] 3.3 Populate `GpuDeviceInfo` from `hipGetDeviceProperties` with real gfx string
- [ ] 3.4 Implement `is_available()` via lazy HIP library loader
- [ ] 3.5 Implement ordered `Drop`: rocBLAS handle, stream, device

## 4. Vector Storage
- [ ] 4.1 Implement `RocmVectorStorage` with `*mut f32` device pointer
- [ ] 4.2 Batched `add_vectors` with single `hipMemcpyAsync` + `hipStreamSynchronize`
- [ ] 4.3 Dynamic buffer expansion via new `hipMalloc` + D2D `hipMemcpyAsync` + `hipFree`
- [ ] 4.4 Soft-delete via `removed_indices: HashSet<usize>` matching Metal/CUDA pattern
- [ ] 4.5 Implement `clear` / `vector_count` / `dimension` / `get_vector`

## 5. HIP Kernels
- [ ] 5.1 Author `src/rocm/kernels.hip` with `hip_l2_distance_kernel` using runtime warpSize
- [ ] 5.2 Route cosine similarity through `rocblas_sgemv` + normalization kernel
- [ ] 5.3 Route dot product through `rocblas_sgemv`
- [ ] 5.4 Compile multi-arch PTX via `hipcc --offload-arch=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100`
- [ ] 5.5 Implement HIP-source-JIT fallback for hosts without `hipcc` in PATH
- [ ] 5.6 Rust launcher in `src/rocm/kernels.rs` with descriptor-set management

## 6. Consistency and Benchmarks
- [ ] 6.1 Extend `tests/cross_backend_consistency.rs` to include ROCm within 1e-4 tolerance
- [ ] 6.2 Validate on both RDNA (wave=32) and CDNA (wave=64) hardware before merge
- [ ] 6.3 Extend `benches/gpu_operations.rs` with ROCm variants gated by `rocm` feature
- [ ] 6.4 Record baseline numbers on reference AMD hardware (MI210 and RX 7900 XTX)

## 7. CI and Build Verification
- [ ] 7.1 Add GitHub Actions job using `rocm/dev-ubuntu-22.04:6.0` container for build
- [ ] 7.2 Gate runtime tests behind `RocmContext::is_available()` for CI hosts without GPU
- [ ] 7.3 Verify `cargo clippy --features rocm -- -D warnings` is clean
- [ ] 7.4 Verify `cargo fmt --all --check` is clean

## 8. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 8.1 Update or create documentation covering the implementation
- [ ] 8.2 Write tests covering the new behavior
- [ ] 8.3 Run tests and confirm they pass
