# GPU Context Specification - Device Info API Changes

## ADDED Requirements

### Requirement: Device Information Query
The system SHALL provide comprehensive GPU device information through the `GpuContext` trait.

#### Scenario: Query device information successfully
- **WHEN** a user calls `device_info()` on a valid GPU context
- **THEN** the system returns a `GpuDeviceInfo` struct containing:
  - Device name (e.g., "Apple M1 Pro", "NVIDIA RTX 4090")
  - Backend type (e.g., "Metal", "CUDA", "ROCm")
  - Total VRAM in bytes
  - Available VRAM in bytes
  - Currently used VRAM in bytes
  - Driver version string
  - Compute capability (if applicable)
  - Maximum threads per block (if applicable)
  - Maximum shared memory per block (if applicable)
  - Device ID for multi-GPU systems
  - PCI bus ID (if applicable)

#### Scenario: Check VRAM availability before allocation
- **WHEN** a user needs to allocate N bytes of GPU memory
- **THEN** the system provides `has_available_vram(N)` method
- **AND** the method returns true if N bytes are available
- **AND** the method returns false otherwise

#### Scenario: Monitor VRAM usage percentage
- **WHEN** a user calls `vram_usage_percent()` on device info
- **THEN** the system calculates and returns percentage as `(used / total) * 100.0`
- **AND** the percentage is between 0.0 and 100.0

### Requirement: Metal Backend Device Info
The Metal backend SHALL implement device information query using Metal framework APIs.

#### Scenario: Metal device properties retrieval
- **WHEN** `device_info()` is called on a Metal context
- **THEN** the system queries device name via `MTLDevice.name`
- **AND** queries total VRAM via `recommendedMaxWorkingSetSize`
- **AND** queries current allocation via `currentAllocatedSize`
- **AND** calculates available VRAM as `total - used`
- **AND** retrieves macOS version as driver version

#### Scenario: Metal hardware limits reporting
- **WHEN** Metal device info is requested
- **THEN** the system reports typical Metal values:
  - Max threads per block: 1024
  - Max shared memory: 32768 bytes (32 KB)
  - Compute capability: None (Metal doesn't use this)
  - PCI bus ID: None (not exposed by Metal)

### Requirement: Type Safety and Error Handling
All device info operations SHALL be type-safe and handle errors gracefully.

#### Scenario: Handle Metal framework errors
- **WHEN** Metal device queries fail
- **THEN** the system wraps errors in `HiveGpuError`
- **AND** provides meaningful error messages
- **AND** does not panic

#### Scenario: Validate VRAM calculations
- **WHEN** VRAM calculations result in negative or invalid values
- **THEN** the system uses `saturating_sub()` to prevent underflow
- **AND** ensures all returned values are non-negative
- **AND** ensures available ≤ total

## MODIFIED Requirements

### Requirement: GpuContext Trait Extension
The `GpuContext` trait SHALL be extended with device information methods while maintaining backward compatibility.

#### Scenario: Existing code continues to work
- **WHEN** code using old `GpuContext` methods is compiled
- **THEN** all existing methods remain functional
- **AND** no breaking changes are introduced
- **AND** new methods are available for opt-in usage

#### Scenario: Convenient helper methods
- **WHEN** `GpuContext` provides helper methods
- **THEN** `vram_usage()` returns current VRAM usage in bytes
- **AND** `has_available_vram(bytes)` checks VRAM availability
- **AND** both methods delegate to `device_info()` internally

## Implementation Notes

**Core Type Definition**:
```rust
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: String,
    pub vram_total: usize,
    pub vram_available: usize,
    pub vram_used: usize,
    pub driver_version: String,
    pub compute_capability: Option<String>,
    pub max_threads_per_block: Option<usize>,
    pub max_shared_memory: Option<usize>,
    pub device_id: usize,
    pub pci_bus_id: Option<String>,
}
```

**Trait Extension**:
```rust
pub trait GpuContext: Send + Sync {
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError>;
    
    fn vram_usage(&self) -> Result<usize, HiveGpuError> {
        Ok(self.device_info()?.vram_used)
    }
    
    fn has_available_vram(&self, required_bytes: usize) -> Result<bool, HiveGpuError> {
        Ok(self.device_info()?.has_available_vram(required_bytes))
    }
    
    // ... existing methods unchanged
}
```

**Testing Requirements**:
- Unit tests for `GpuDeviceInfo` helper methods
- Integration tests for Metal implementation
- Edge case tests (low VRAM, calculations)
- Regression tests for existing functionality

**Documentation Requirements**:
- Comprehensive rustdoc comments
- Usage examples in doc comments
- Update API reference documentation
- Add troubleshooting guide

