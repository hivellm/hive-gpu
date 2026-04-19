## 1. Infrastructure
- [x] 1.1 Enable `cudarc` dependency behind `cuda` feature in `Cargo.toml`
- [x] 1.2 Add `HiveGpuError::CudaError(String)` and `CublasError(String)` in `src/error.rs`
- [x] 1.3 Create minimal `build.rs` with `rerun-if-changed` for CUDA kernels
- [x] 1.4 Remove `#![allow(warnings)]` from `src/lib.rs`
- [x] 1.5 Replace env-var detection with `cudarc::driver::CudaDevice::count()` in `src/backends/detector.rs`

## 2. Context
- [ ] 2.1 Rewrite `src/cuda/context.rs` using `cudarc::driver::CudaDevice`
- [ ] 2.2 Populate `GpuDeviceInfo` via `cuDeviceGetAttribute` + `cuDeviceTotalMem`
- [ ] 2.3 Implement real `CudaContext::is_available()`
- [ ] 2.4 Implement ordered `Drop` for context resources

## 3. Vector Storage
- [ ] 3.1 Implement `CudaVectorStorage` with `DeviceSlice<f32>` backing buffer
- [ ] 3.2 Batched `add_vectors` using single `htod_copy` per call
- [ ] 3.3 Dynamic buffer expansion with D2D reallocation mirroring Metal
- [ ] 3.4 Soft-delete via `removed_indices: HashSet<usize>`
- [ ] 3.5 Implement `clear` / `vector_count` / `dimension` / `get_vector`

## 4. Distance Kernels
- [ ] 4.1 Author `src/cuda/kernels.cu` with `l2_distance_kernel`
- [ ] 4.2 Author cosine similarity kernel via cuBLAS SGEMV or fused variant
- [ ] 4.3 Author dot product kernel
- [ ] 4.4 Compile offline to multi-SM PTX (sm_70, sm_75, sm_80, sm_86, sm_89, sm_90)
- [ ] 4.5 Embed PTX via `include_str!` and implement Rust launcher in `src/cuda/kernels.rs`
- [ ] 4.6 Implement CPU-side top-K after score readback

## 5. Consistency and Benchmarks
- [ ] 5.1 Create `tests/cross_backend_consistency.rs` comparing Metal vs CUDA within `1e-4`
- [ ] 5.2 Extend `benches/gpu_operations.rs` with CUDA variants gated by `cuda` feature
- [ ] 5.3 Record baseline numbers on reference NVIDIA hardware

## 6. CI and Build Verification
- [ ] 6.1 Add GitHub Actions job using `nvidia/cuda:12.4-devel-ubuntu22.04` for build
- [ ] 6.2 Gate runtime tests behind `CudaContext::is_available()` so CI hosts without GPU exit cleanly
- [ ] 6.3 Verify `cargo clippy --features cuda -- -D warnings` is clean
- [ ] 6.4 Verify `cargo fmt --all --check` is clean

## 7. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 7.1 Update or create documentation covering the implementation
- [ ] 7.2 Write tests covering the new behavior
- [ ] 7.3 Run tests and confirm they pass
