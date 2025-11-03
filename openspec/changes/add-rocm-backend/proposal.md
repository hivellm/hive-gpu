# Add ROCm Backend

## Why

With Metal (~5%) and CUDA (~70%) implemented, we have 75% market coverage. **AMD GPUs represent ~15% of the ML/AI server market**, particularly in:
- AMD Instinct data center GPUs (MI200, MI300 series)
- Radeon Pro and RX 6000/7000 series
- Cloud instances (AWS with AMD GPUs)
- Cost-sensitive deployments (AMD often cheaper than NVIDIA)

Adding ROCm support brings us to **~90% total GPU market coverage**, making `hive-gpu` truly universal across all major GPU vendors.

**Strategic Importance**: Many organizations choose AMD for cost reasons or to avoid vendor lock-in. Supporting ROCm positions us as a vendor-neutral solution.

## What Changes

- Add ROCm/HIP backend implementation for AMD GPUs
- Support ROCm 5.0+ with gfx900+ architectures (Vega and newer)
- Implement `RocmContext` for HIP device and stream management
- Implement `RocmVectorStorage` for GPU memory operations
- Create HIP kernels for distance computation
- Integrate rocBLAS for optimized matrix operations
- Add build.rs support for HIP compilation
- Support multiple AMD GPU architectures
- Implement device info API for ROCm
- Provide comprehensive error handling

**Breaking Changes**: None (pure backend addition with feature flag)

## Impact

**Affected specs:**
- New: `rocm-backend` - Complete ROCm implementation spec
- `gpu-context` - ROCm context implementation
- `types` - No changes needed (already defined)
- `error` - Add ROCm-specific errors

**Affected code:**
- NEW: `src/rocm/` - Complete backend module
  - `context.rs` - HIP context and rocBLAS handle
  - `storage.rs` - Vector storage with GPU memory
  - `kernels.hip` - HIP C++ kernels
  - `mod.rs` - Module exports
- UPDATE: `build.rs` - Add HIP kernel compilation
- `Cargo.toml` - Add ROCm dependencies and feature flag
- `src/error.rs` - Add ROCm error types
- NEW: `tests/rocm_tests.rs` - ROCm-specific tests
- NEW: `examples/rocm_basic.rs` - ROCm usage example

**Benefits:**
- **15% additional market coverage** (75% → 90%)
- Support for cost-effective AMD GPUs
- Vendor diversity and reduced lock-in
- Complete coverage of major GPU vendors
- Similar performance to CUDA for many workloads

**Timeline**: 1-2 weeks implementation + testing
**Priority**: ⚡ HIGH (after CUDA stable)
**Dependencies**: CUDA Backend (Phase 3.1) - establishes patterns

