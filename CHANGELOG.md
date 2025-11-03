# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

