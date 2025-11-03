# Implementation Tasks

## 1. Define Core Types
- [ ] 1.1 Create `GpuDeviceInfo` struct in `src/types.rs`
  - [ ] Add all fields (name, backend, vram_*, driver_version, etc.)
  - [ ] Implement `Debug` and `Clone` traits
  - [ ] Add helper methods (`vram_usage_percent`, `has_available_vram`)
  - [ ] Add comprehensive doc comments with examples

## 2. Update Trait Definitions
- [ ] 2.1 Add `device_info()` method to `GpuContext` trait
  - [ ] Method signature: `fn device_info(&self) -> Result<GpuDeviceInfo>`
  - [ ] Add default implementations for convenience methods
  - [ ] Update trait documentation

## 3. Implement for Metal Backend
- [ ] 3.1 Implement `device_info()` in `MetalNativeContext`
  - [ ] Query Metal device properties via `device.name()`
  - [ ] Get VRAM info via `recommended_max_working_set_size()` and `current_allocated_size()`
  - [ ] Get macOS version as driver version (via `sw_vers` command)
  - [ ] Report Metal-specific capabilities (max threads, shared memory)
  - [ ] Handle PCI bus ID (set to None for Metal)
- [ ] 3.2 Add helper function `get_macos_version()`
  - [ ] Execute `sw_vers -productVersion` command
  - [ ] Parse output and return version string
  - [ ] Handle errors gracefully

## 4. Testing
- [ ] 4.1 Create `tests/device_info_tests.rs`
- [ ] 4.2 Add Metal device info test
  - [ ] Verify all fields are populated
  - [ ] Check VRAM values are positive and consistent
  - [ ] Validate backend name is "Metal"
  - [ ] Test helper methods (`vram_usage_percent`, `has_available_vram`)
- [ ] 4.3 Add edge case tests
  - [ ] Test with low VRAM conditions
  - [ ] Test VRAM percentage calculation
  - [ ] Test `has_available_vram()` with various thresholds
- [ ] 4.4 Run all existing tests to ensure no regressions

## 5. Documentation
- [ ] 5.1 Update `docs/API_REFERENCE.md`
  - [ ] Document `GpuDeviceInfo` struct
  - [ ] Document new trait methods
  - [ ] Add usage examples
- [ ] 5.2 Update `docs/DEVELOPMENT.md`
  - [ ] Add section on device info API
  - [ ] Explain how to query device properties
- [ ] 5.3 Add example code in doc comments
- [ ] 5.4 Update CHANGELOG.md with new features

## 6. Quality Checks
- [ ] 6.1 Run `cargo fmt --all`
- [ ] 6.2 Run `cargo clippy --all-targets -- -D warnings`
- [ ] 6.3 Run `cargo test --all-features`
- [ ] 6.4 Verify test coverage ≥95%
- [ ] 6.5 Run `cargo doc --no-deps` and verify documentation builds

## 7. Validation
- [ ] 7.1 Verify API works on macOS with Metal
- [ ] 7.2 Confirm no breaking changes to existing code
- [ ] 7.3 Validate that API design supports future CUDA/ROCm implementations
- [ ] 7.4 Run OpenSpec validation: `openspec validate add-device-info-api --strict`

