# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-19

### Added

- **Functional CUDA backend** for NVIDIA GPUs (Volta / sm_70+) on Linux and
  Windows, built on `cudarc 0.13` driver API + cuBLAS:
  - `CudaContext` wraps `cudarc::driver::CudaDevice` plus a cuBLAS handle;
    live device info via `cuDeviceGetAttribute`, `cuMemGetInfo`, and
    `cuDriverGetVersion`, plus PCI bus id formatting.
  - `CudaVectorStorage` backed by a single contiguous `CudaSlice<f32>` with
    batched `htod_copy` + `memcpy_dtod_sync` uploads. Adaptive capacity
    growth (2× / 1.5× / 1.2×) matching the Metal backend, soft-delete via
    `HashSet<usize>`, host-cached squared norms.
  - GPU search via cuBLAS SGEMV (trans=T on the column-major view): works
    for DotProduct, Cosine (SGEMV + norm normalisation), and Euclidean
    (L2² derived from dots + norms). Top-K on the CPU after a single
    `dtoh_sync_copy`.
- **CUDA IVF index** (`CudaIvfIndex`, phase4b) — inverted-file index on
  NVIDIA. k-means++ initialisation + Lloyd iterations for training,
  cuBLAS SGEMM for the assignment step (query × centroids), per-list
  residual refinement, and an adjustable `nprobe`. Validated on RTX
  4090: **3.67× faster than the brute-force baseline at 1 M vectors**,
  recall ≥ 0.95 on clustered data. Tests in `tests/cuda_ivf.rs`,
  benchmarks in `benches/cuda_ivf.rs`.
- **CUDA integration tests** (17 brute-force + IVF tests, all passing on
  RTX 4090): `tests/cuda_smoke.rs`, `tests/cuda_device_info.rs`,
  `tests/cuda_vector_ops.rs`, `tests/cuda_ivf.rs`. Each test is a no-op
  on hosts without a reachable driver, keeping CI green on GPU-less
  runners.
- **CUDA benchmarks** `benches/cuda_ops.rs` and `benches/cuda_ivf.rs`
  comparing GPU throughput against a naïve CPU reference.
- **GitHub Actions workflow** `.github/workflows/cuda-build.yml` runs
  check, clippy, fmt, and the test suite against the
  `nvidia/cuda:12.4.1-devel-ubuntu22.04` container image.
- **Metal brute-force via a real compute kernel** (phase4a, *authored
  blind*). Replaces the prior CPU-fallback shim with a
  `sgemv_dot.metal` compute shader driven through
  `MTLComputeCommandEncoder`. Supports DotProduct, Cosine, and
  Euclidean on Apple Silicon.
- **Metal IVF index** (`MetalIvfIndex`, phase4c, *authored blind*) —
  mirrors `CudaIvfIndex`, wired to two custom Metal compute kernels
  (`sgemv_dot.metal`, `sgemm_dot.metal`). Tests in
  `tests/metal_bruteforce.rs` and `tests/metal_ivf.rs`.
- **ROCm / HIP backend** (phase3b, *authored blind*) for AMD GPUs on
  Linux (gfx900–gfx1100). `RocmContext` + `RocmVectorStorage` +
  `RocmIvfIndex`, hand-rolled HIP FFI via `libloading`
  (`libamdhip64.so` + `librocblas.so`). Mirrors the CUDA architecture
  one-for-one. Tests in `tests/rocm_smoke.rs` and
  `tests/rocm_ivf.rs`.
- **Intel / Vulkan Compute backend** (phase3c, *authored blind*) for
  Intel Arc / Battlemage on Linux and Windows, with
  `HIVE_GPU_VULKAN_UNIVERSAL=1` fallback for any Vulkan 1.2 GPU.
  `IntelContext` + `IntelVectorStorage` + `IntelIvfIndex` built on
  `ash 0.38`. WGSL compute shaders (`sgemv_dot.wgsl`,
  `sgemm_dot.wgsl`) compiled to SPIR-V at build time via `naga`
  (pure-Rust, no CMake / C++ toolchain). Tests in
  `tests/intel_smoke.rs` and `tests/intel_ivf.rs`.
- `HiveGpuError` variants for every new backend: `CudaError`,
  `CublasError`, `HipError`, `RocblasError`, `RocmError`,
  `VulkanError`, `IntelError`, `SpirvCompileError`.
- `GpuBackendType::{Rocm, Intel}` in `src/backends/detector.rs`. New
  priority order is `Metal > CUDA > ROCm > Intel > CPU`, each probed
  with a real loader check — `is_rocm_available()` via `libloading`,
  `is_intel_available()` via Vulkan `vkEnumeratePhysicalDevices`.
- `IvfConfig` in `src/types.rs` — shared across all four IVF
  implementations (CUDA / Metal / ROCm / Intel).
- `build.rs` compiles the Intel WGSL shaders when the `intel` feature
  is active and emits rerun hints for CUDA kernel assets.
- Multi-backend analyses under `docs/analysis/{cuda,gcn,intel}/`
  documenting state, gaps, and phased plans.

### Changed

- Detection in `src/backends/detector.rs` now uses
  `cudarc::driver::result::init` + `get_count` instead of env-var
  inspection. Target-gated to Linux / Windows; macOS is unaffected.
- `cuda` Cargo feature now actually pulls in its dependency: `cuda =
  ["dep:cudarc"]` with `cudarc` declared in a target-gated
  `[target.'cfg(any(target_os = "linux", target_os = "windows"))']`
  block carrying `driver`, `cublas`, `cuda-12040`, and
  `dynamic-linking` features.
- `default-features` resolves to nothing on non-macOS hosts — every
  backend dep is target- and feature-gated, so the crate builds
  clean everywhere with default features.
- Removed project-wide `#![allow(warnings)]` from `src/lib.rs`.
  Cleaned up 24 latent warnings (unused imports, underscore-prefixed
  params, scoped `#[allow(dead_code)]` on struct fields still being
  populated by follow-up phases). `cargo clippy --all-features --lib
  --tests --benches -- -D warnings` is now part of the quality gate.
- `docs/benchmarks/PERFORMANCE.md` updated with RTX 4090 baseline
  numbers, CUDA IVF head-to-head vs. brute-force, and the CUDA test
  suite summary.
- `docs/ROADMAP.md` reflects the actual ship order with explicit
  validated-vs-blind status per backend.

### Breaking

- None at the public-API level; existing Metal code paths and trait
  signatures are unchanged. The `cuda` feature behaviour changed from
  a compile-time no-op in 0.1.x to a fully functional backend in
  0.2.0.

### Status notes

Only the CUDA path (brute-force + IVF) has been validated on real
hardware (RTX 4090) in the 0.2.0 release window. Metal (real kernel +
IVF), ROCm, and Intel ship as **authored blind** — the code
cross-compiles, passes `clippy -D warnings`, and has a complete test
suite, but has never executed against the target hardware. Follow-up
validation tasks are live in `.rulebook/tasks/` and will ship a
minor bump each once the corresponding maintainer runs them:

- `phase4d_validate-metal-backend-on-mac` — Metal brute-force + IVF
- `phase4e_validate-rocm-backend-on-amd` — ROCm brute-force + IVF
- `phase4f_validate-intel-backend-on-vulkan` — Intel / Vulkan
  brute-force + IVF

## [0.1.10] - 2025-11-04

### Fixed

- **Clippy Build Error**: Moved `Duration` and `Instant` imports inside conditional compilation block in `tests/gpu_stress_tests.rs`
  - Imports were causing unused import warnings when compiled without `metal-native` feature
  - Now properly scoped within `#[cfg(all(target_os = "macos", feature = "metal-native"))]` module
  - All clippy checks passing with `-D warnings`

## [0.1.9] - 2025-11-04

### Added

- **Comprehensive GPU Test Suite (72 tests total)**
  - GPU Detection Tests (9 tests): Metal device availability, name retrieval, capabilities, multiple contexts, VRAM query, backend detection, performance info
  - Vector Operations Tests (11 tests): Small/medium/large vector addition, cosine similarity, orthogonal vectors, Euclidean distance, batch operations, search accuracy, edge cases
  - Memory Management Tests (10 tests): Small/medium/large buffer allocation, multiple allocations, deallocation, repeated cycles, memory reuse, clear vectors, stress tests
  - VRAM Monitoring Tests (10 tests): Tracking accuracy, percentage calculation, available VRAM checks, usage during allocation, consistency, monitoring over time, pressure detection
  - Integration Tests (9 tests): Basic operations, HNSW construction, error handling, distance metrics, VRAM monitoring, cross-backend compatibility
  - Device Info Tests (4 tests): Metal device info query, VRAM usage percent, availability checks, convenience methods
  - **Performance Benchmarks (10 tests)**: 
    - Vector addition throughput: 3,740 vectors/sec
    - Search latency: 0.92 μs (k=10), 1.08M queries/sec
    - Memory bandwidth: 8+ MB/s
    - Dimension scaling: 64D to 1024D
    - Vector count scaling: 100 to 5000
    - Cold vs warm performance comparison
    - Distance metric performance comparison
    - Concurrent operations testing
    - Performance baseline validation
  - **Stress Tests (9 tests)**:
    - Sustained load: 5000 vectors, 3728 vec/sec
    - Maximum capacity: 10K vectors, 4250 vec/sec
    - Memory pressure scenarios
    - Rapid allocation cycles: 50 cycles
    - Sustained search load: 2000+ QPS
    - Mixed read/write workload
    - Long-running stability (5 seconds)
    - Error recovery validation
    - Concurrent high load: 10 storages

- **Test Infrastructure**
  - `scripts/run-gpu-tests.sh`: Comprehensive test runner with detailed output
  - Enhanced Git hooks with test suite information in pre-commit and pre-push
  - All tests run on real Metal backend (Apple M3 Pro)
  - Tests validate real GPU operations, memory management, VRAM usage, and performance

- **Documentation**
  - `docs/guides/DEVICE_INFO_IMPLEMENTATION.md`: Complete implementation guide for Device Info API
    - Architecture and design decisions
    - Platform-specific details (Metal, CUDA)
    - Real-world usage examples
    - Common pitfalls and best practices
    - Performance considerations
    - Migration guide from old code
    - Test coverage overview
  - `docs/guides/GIT_HOOKS_TESTING.md`: Git hooks configuration and testing guide (created earlier)

### Changed

- Updated Git pre-commit hook to display detailed test suite information
- Updated Git pre-push hook with comprehensive test count (78 tests total)
- Enhanced test output with performance metrics and real GPU data

### Performance

Real-world benchmarks on Apple M3 Pro:
- **Search Performance**: 1.08M queries/sec (k=10), 0.92 μs latency
- **Vector Addition**: 3,740 vectors/sec sustained, 4,250 vectors/sec peak (10K vectors)
- **Memory Bandwidth**: 8+ MB/s effective (including Metal overhead)
- **Scalability**: Linear scaling from 100 to 5000 vectors
- **Sustained Load**: Stable operation at 3,728 vec/sec over 5 seconds
- **Concurrent Load**: 10 storages with 500 vectors each, all successful

### Testing

- Total test count: 78 tests (72 functional + 6 doc tests)
- All tests passing on Apple M3 Pro with Metal backend
- Stress tests validate system stability under extreme load
- Performance benchmarks establish baseline for CI/CD validation

## [0.1.8] - 2025-11-04

### Changed

- **BREAKING: Migrated from discontinued `metal-rs` to `objc2-metal` ecosystem**
  - Replaced `metal 0.27` with `objc2-metal 0.3.2` (actively maintained)
  - Replaced `objc 0.2` with `objc2 0.6.3` (modern, type-safe bindings)
  - Added `objc2-foundation 0.3.2` for Foundation framework support
  - Updated all Metal bindings to use `ProtocolObject<dyn MTLDevice>` pattern
  - Migrated buffer operations to objc2-metal API (camelCase method names)
  - All Metal-specific code now uses objc2-metal traits (MTLDevice, MTLCommandQueue, MTLCommandEncoder, etc.)
  - Complete migration of `src/metal/context.rs`, `src/metal/vector_storage.rs`, `src/metal/buffer_pool.rs`, `src/backends/detector.rs`
  - All 21 tests passing (unit, integration, doc tests)
  - Zero clippy warnings
  - **Security**: Removed dependency on discontinued library with no security updates
  - **Maintenance**: Now using actively maintained crates from objc2 ecosystem
  - **Type Safety**: Improved type safety with modern Objective-C bindings
  - **Future-Proof**: Foundation for continued macOS/Metal development

### Fixed

- Metal device detection now uses `MTLCreateSystemDefaultDevice()` from objc2-metal
- Buffer creation uses proper `MTLResourceOptions` and type-safe methods
- Command buffer and blit encoder creation using objc2-metal patterns

### Internal

- OpenSpec change `migrate-to-objc2-metal` tracking migration progress
- Created rollback tag `pre-objc2-migration` for safety
- Comprehensive migration documentation in `docs/guides/MIGRATION_METAL_OBJC2.md`

## [0.1.7] - 2025-11-03

### Added

- **Device Info API** (Phase 2) - Comprehensive GPU device information API
  - New `GpuDeviceInfo` struct with detailed hardware information:
    - VRAM tracking (total, available, used bytes)
    - Driver version and compute capability
    - Hardware limits (max threads per block, max shared memory)
    - Backend identification (Metal, CUDA, ROCm, wgpu)
    - Device ID and PCI bus ID (where applicable)
  - New `device_info()` method on `GpuContext` trait (returns `Result<GpuDeviceInfo>`)
  - Helper methods:
    - `vram_usage_percent()` - Calculate VRAM usage percentage
    - `has_available_vram(bytes)` - Check if sufficient VRAM available
    - `total_vram_mb()` / `available_vram_mb()` - Convenient MB conversions
  - Full Metal backend implementation with macOS version detection
  - Placeholder implementations for CUDA and wgpu backends
- Comprehensive test suite for Device Info API (5 tests, 100% passing)
- OpenSpec changes for future implementations:
  - `add-device-info-api` - Device Info API specification
  - `add-cuda-backend` - CUDA backend specification (43 tasks)
  - `add-rocm-backend` - ROCm backend specification (46 tasks)
  - `add-memory-pooling` - Memory pooling optimization (33 tasks)
- Complete project documentation:
  - `docs/API_REFERENCE.md` - API documentation
  - `docs/ARCHITECTURE.md` - System architecture
  - `docs/DEVELOPMENT.md` - Development guide
  - `docs/ROADMAP.md` - Project roadmap
  - `docs/DAG.md` - Component dependencies
  - `docs/PERFORMANCE.md` - Performance benchmarks
  - `docs/INTEGRATION_GUIDE.md` - Integration examples
- CI/CD workflows:
  - Rust testing workflow
  - Rust linting workflow
  - Codespell workflow
- Project governance files:
  - `CODE_OF_CONDUCT.md`
  - `CONTRIBUTING.md`
  - `SECURITY.md`
  - `AGENTS.md` - AI assistant rules

### Changed

- Updated `GpuContext` trait to return `Result<GpuDeviceInfo>` instead of `GpuDeviceInfo`
- Improved error handling across all backends
- Enhanced documentation with comprehensive examples

### Fixed

- Fixed unused imports in benchmarks
- Fixed clippy warnings in test files
- Fixed doctest compilation errors

## [0.1.6] - Previous Release

### Added

- Initial Metal Native backend implementation
- Basic CUDA and wgpu placeholder implementations
- Vector storage and HNSW graph operations
- Core traits and types

[0.1.7]: https://github.com/hivellm/hive-gpu/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/hivellm/hive-gpu/releases/tag/v0.1.6

