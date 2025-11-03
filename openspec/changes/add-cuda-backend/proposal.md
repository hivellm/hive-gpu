# Add CUDA Backend

## Why

Currently, `hive-gpu` only supports Metal on macOS, covering ~5% of production ML/AI servers. **NVIDIA GPUs dominate the market with ~70% share** in production environments, making CUDA support **critical for adoption**.

Without CUDA, the library is limited to Apple Silicon users, excluding the vast majority of:
- Cloud GPU instances (AWS, GCP, Azure)
- Data center deployments
- ML/AI training and inference workloads
- HPC environments

**ROI**: With just 1-2 weeks of development, we unlock 70% of the market, achieving 10-50x speedup over CPU for vector operations.

## What Changes

- Add CUDA backend implementation for NVIDIA GPUs
- Support CUDA 12.0+ with compute capability 7.0+ (Volta and newer)
- Implement `CudaContext` for device and stream management
- Implement `CudaVectorStorage` for GPU memory operations
- Create CUDA kernels for distance computation (L2/Euclidean)
- Integrate cuBLAS for optimized matrix operations (Cosine, Dot Product)
- Add build.rs for automatic CUDA kernel compilation
- Support multiple GPU architectures (sm_70 to sm_90)
- Implement device info API for CUDA
- Provide comprehensive error handling

**Breaking Changes**: None (pure backend addition with feature flag)

## Impact

**Affected specs:**
- New: `cuda-backend` - Complete CUDA implementation spec
- `gpu-context` - CUDA context implementation
- `types` - No changes needed (already defined)
- `error` - Add CUDA-specific errors

**Affected code:**
- NEW: `src/cuda/` - Complete backend module
  - `context.rs` - CUDA context and cuBLAS handle
  - `storage.rs` - Vector storage with GPU memory
  - `kernels.cu` - CUDA C++ kernels
  - `mod.rs` - Module exports
- NEW: `build.rs` - CUDA kernel compilation
- `Cargo.toml` - Add CUDA dependencies and feature flag
- `src/error.rs` - Add CUDA error types
- NEW: `tests/cuda_tests.rs` - CUDA-specific tests
- NEW: `examples/cuda_basic.rs` - CUDA usage example

**Benefits:**
- **70% market coverage** (from 5% to 75%)
- **10-50x performance** vs CPU operations
- **Latency reduction** from 10-30ms to 0.5-3ms
- Production-ready for major cloud providers
- Foundation for future multi-GPU support

**Timeline**: 1-2 weeks implementation + testing
**Priority**: 🔥 CRITICAL (highest ROI)
**Dependencies**: Device Info API (Phase 2)

