# Add Device Info API

## Why

Currently, `hive-gpu` lacks a standardized way to query GPU device information across backends. Users cannot inspect VRAM availability, driver versions, compute capabilities, or hardware limits before allocating resources. This creates risks of out-of-memory errors and makes debugging difficult.

The Device Info API is **foundational** for CUDA and ROCm backends (Phase 3), as both require exposing backend-specific capabilities. Without this API, we cannot properly implement multi-backend support.

## What Changes

- Add `GpuDeviceInfo` struct with comprehensive device properties
- Add `device_info()` method to `GpuContext` trait
- Add helper methods (`vram_usage_percent()`, `has_available_vram()`)
- Implement for Metal backend first
- Prepare API for CUDA and ROCm future implementations

**Breaking Changes**: None (pure addition)

## Impact

**Affected specs:**
- `gpu-context` - Adding new trait methods
- `types` - Adding new struct

**Affected code:**
- `src/types.rs` - New `GpuDeviceInfo` struct
- `src/traits.rs` - New methods in `GpuContext` trait
- `src/metal/context.rs` - Metal implementation
- `tests/device_info_tests.rs` - New test file

**Benefits:**
- Foundation for CUDA/ROCm implementations
- Better resource planning and error handling
- Improved debugging and monitoring
- No breaking changes to existing APIs

**Timeline**: 1-2 days implementation + testing
**Priority**: 🔥 CRITICAL (blocks Phase 3)

