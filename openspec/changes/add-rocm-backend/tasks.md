# Implementation Tasks

## 1. Project Setup and Dependencies
- [ ] 1.1 Update `Cargo.toml`
  - [ ] Add `hip-runtime-sys = { version = "0.3", optional = true }`
  - [ ] Add `hip-driver-sys = { version = "0.3", optional = true }`
  - [ ] Add `rocblas-sys = { version = "0.3", optional = true }`
  - [ ] Add `rocm` feature flag
  - [ ] Update build-dependencies if needed
- [ ] 1.2 Update `build.rs`
  - [ ] Detect ROCm_PATH or ROCM_HOME environment variable
  - [ ] Configure hipcc compilation flags
  - [ ] Set up multi-architecture compilation (gfx900, gfx906, gfx908, gfx90a, gfx1030, etc.)
  - [ ] Link HIP runtime and rocBLAS libraries
  - [ ] Add rerun-if-changed for kernels.hip

## 2. Error Handling
- [ ] 2.1 Extend `HiveGpuError` in `src/error.rs`
  - [ ] Add `RocmError(String)` variant
  - [ ] Add `RocblasError(String)` variant
  - [ ] Add `HipError(String)` variant
  - [ ] Reuse `NoDevice` and `InvalidDeviceId` variants
- [ ] 2.2 Create helper functions
  - [ ] `hip_check()` - Convert HIP errors to HiveGpuError
  - [ ] `rocblas_check()` - Convert rocBLAS errors to HiveGpuError

## 3. ROCm Context Implementation
- [ ] 3.1 Create `src/rocm/context.rs`
- [ ] 3.2 Define `RocmContext` struct
  - [ ] Add `device_id: i32` field
  - [ ] Add `stream: hipStream_t` field
  - [ ] Add `rocblas_handle: rocblas_handle` field
- [ ] 3.3 Implement constructors
  - [ ] `new()` - Create with default device (0)
  - [ ] `new_with_device(device_id)` - Select specific GPU
  - [ ] `device_count()` - Query number of GPUs via `hipGetDeviceCount`
  - [ ] `is_available()` - Check ROCm availability
- [ ] 3.4 Implement `GpuContext` trait
  - [ ] `device_info()` - Query device properties and VRAM via `hipGetDeviceProperties`
  - [ ] `create_storage()` - Create RocmVectorStorage
- [ ] 3.5 Implement `Drop` trait
  - [ ] Destroy rocBLAS handle
  - [ ] Destroy HIP stream
  - [ ] Clean up resources safely

## 4. ROCm Vector Storage
- [ ] 4.1 Create `src/rocm/storage.rs`
- [ ] 4.2 Define `RocmVectorStorage` struct
  - [ ] Device ID and stream references
  - [ ] rocBLAS handle reference
  - [ ] Dimension and metric
  - [ ] Host vector storage (Vec<GpuVector>)
  - [ ] Device memory pointer (*mut f32)
  - [ ] Capacity tracking
- [ ] 4.3 Implement memory management
  - [ ] `ensure_capacity()` - Dynamic GPU memory growth
  - [ ] GPU memory allocation with `hipMalloc`
  - [ ] GPU memory deallocation with `hipFree`
  - [ ] Data transfer (`hipMemcpyAsync`)
- [ ] 4.4 Implement `GpuVectorStorage` trait
  - [ ] `add_vector()` - Add single vector to GPU
  - [ ] `add_vectors()` - Batch add with flattening
  - [ ] `search()` - Distance computation + top-k
  - [ ] `get_vector()` - Retrieve by ID
  - [ ] `remove_vector()` - Mark for removal
  - [ ] `update_vector()` - Update on GPU
  - [ ] `vector_count()` - Return count
  - [ ] `clear()` - Clear all vectors
- [ ] 4.5 Implement `Drop` trait
  - [ ] Free device memory safely with `hipFree`

## 5. HIP Kernels
- [ ] 5.1 Create `src/rocm/kernels.hip`
- [ ] 5.2 Implement L2 distance kernel
  - [ ] `hip_l2_distance_kernel()` - Compute Euclidean distance
  - [ ] Optimize for AMD wavefront size (64 threads typically)
  - [ ] Use efficient block/thread configuration
  - [ ] Leverage LDS (Local Data Share) if beneficial
- [ ] 5.3 Create C-style wrapper functions
  - [ ] `hip_l2_distance()` - Callable from Rust
  - [ ] Proper error handling
  - [ ] Stream support for async execution
- [ ] 5.4 Implement distance computation strategies
  - [ ] Cosine/DotProduct via rocBLAS SGEMV
  - [ ] Euclidean via custom HIP kernel
  - [ ] Batch processing support

## 6. Module Organization
- [ ] 6.1 Create `src/rocm/mod.rs`
  - [ ] Export `RocmContext`
  - [ ] Export `RocmVectorStorage`
  - [ ] Conditional compilation with `#[cfg(feature = "rocm")]`
- [ ] 6.2 Update `src/lib.rs`
  - [ ] Add `pub mod rocm` with feature gate
  - [ ] Export ROCm types conditionally

## 7. Testing
- [ ] 7.1 Create `tests/rocm_tests.rs`
- [ ] 7.2 Test ROCm availability
  - [ ] Test `RocmContext::is_available()`
  - [ ] Test `RocmContext::device_count()`
  - [ ] Skip tests gracefully if ROCm not available
- [ ] 7.3 Test device info
  - [ ] Verify all fields populated
  - [ ] Check VRAM values
  - [ ] Validate architecture (gfx) string
  - [ ] Verify PCI bus ID format
- [ ] 7.4 Test vector operations
  - [ ] Add single vector
  - [ ] Add multiple vectors
  - [ ] Search with different metrics
  - [ ] Update and remove vectors
  - [ ] Clear storage
- [ ] 7.5 Test batch operations
  - [ ] Batch add 1000+ vectors
  - [ ] Measure performance
  - [ ] Verify correctness
- [ ] 7.6 Test error handling
  - [ ] Invalid device ID
  - [ ] Out of memory
  - [ ] Dimension mismatch
- [ ] 7.7 Integration tests
  - [ ] Cross-backend consistency (Metal vs CUDA vs ROCm)
  - [ ] Large dataset handling
- [ ] 7.8 Run all tests with ROCm feature
  - [ ] `cargo test --features rocm`
  - [ ] Verify ≥95% coverage

## 8. Examples and Documentation
- [ ] 8.1 Create `examples/rocm_basic.rs`
  - [ ] Device detection example
  - [ ] Context creation
  - [ ] Vector add and search
  - [ ] Error handling
  - [ ] Resource cleanup
- [ ] 8.2 Update documentation
  - [ ] `docs/API_REFERENCE.md` - ROCm types and usage
  - [ ] `docs/DEVELOPMENT.md` - ROCm setup instructions
  - [ ] `docs/PERFORMANCE.md` - ROCm benchmarks
  - [ ] `README.md` - ROCm feature flag
  - [ ] AMD GPU compatibility matrix
- [ ] 8.3 Add rustdoc comments
  - [ ] All public types and methods
  - [ ] Code examples in doc comments
  - [ ] Safety notes for unsafe code
  - [ ] Performance characteristics

## 9. Benchmarking
- [ ] 9.1 Create benchmarks in `benches/`
  - [ ] Vector addition benchmark
  - [ ] Search benchmark (various sizes)
  - [ ] Compare Metal vs CUDA vs ROCm performance
  - [ ] Measure throughput (ops/sec)
  - [ ] Measure latency (ms)
  - [ ] Test on different AMD GPU architectures
- [ ] 9.2 Document performance results
  - [ ] Update `docs/PERFORMANCE.md`
  - [ ] Include hardware specs (MI200, RX 7900, etc.)
  - [ ] Include workload descriptions
  - [ ] Compare with CUDA when possible

## 10. AMD-Specific Optimizations
- [ ] 10.1 Wavefront optimization
  - [ ] Tune kernels for 64-thread wavefronts (vs NVIDIA's 32-thread warps)
  - [ ] Use `__builtin_amdgcn_readfirstlane` intrinsics if beneficial
- [ ] 10.2 Memory coalescing
  - [ ] Ensure coalesced memory access patterns
  - [ ] Profile with rocprof to identify issues
- [ ] 10.3 LDS (Local Data Share) usage
  - [ ] Use shared memory efficiently for reductions
  - [ ] Avoid bank conflicts

## 11. Quality Checks
- [ ] 11.1 Code formatting
  - [ ] `cargo fmt --all`
- [ ] 11.2 Linting
  - [ ] `cargo clippy --features rocm --all-targets -- -D warnings`
  - [ ] Fix all warnings
- [ ] 11.3 Testing
  - [ ] `cargo test --features rocm`
  - [ ] `cargo test --all-features`
- [ ] 11.4 Documentation
  - [ ] `cargo doc --no-deps --features rocm`
  - [ ] Verify all docs build correctly
- [ ] 11.5 Build verification
  - [ ] `cargo build --release --features rocm`
  - [ ] Test on Linux with AMD GPU

## 12. CI/CD Integration
- [ ] 12.1 Update GitHub Actions workflows
  - [ ] Add ROCm test job (ubuntu-latest with ROCm Docker image)
  - [ ] Add ROCm build verification
  - [ ] Add conditional ROCm tests (skip if GPU not available)
- [ ] 12.2 Update README badges
  - [ ] Add ROCm support badge

## 13. Final Validation
- [ ] 13.1 OpenSpec validation
  - [ ] `openspec validate add-rocm-backend --strict`
- [ ] 13.2 Manual testing on real AMD hardware
  - [ ] Test on different GPU models (MI series, RX series)
  - [ ] Verify performance meets expectations
  - [ ] Test with large datasets
  - [ ] Compare with CUDA performance on similar workloads
- [ ] 13.3 Update CHANGELOG.md
  - [ ] Document new ROCm backend feature
  - [ ] List supported GPU architectures
  - [ ] Note any limitations or requirements

