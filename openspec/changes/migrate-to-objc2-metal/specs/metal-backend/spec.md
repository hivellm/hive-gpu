# Metal Backend Specification

This specification defines the requirements for the Metal GPU backend implementation using modern objc2-metal bindings.

## ADDED Requirements

### Requirement: Modern Metal Bindings

The system SHALL use objc2-metal (v0.2+) and objc2 (v0.5+) for all Metal framework interactions, replacing the deprecated metal-rs and objc 0.2 dependencies.

#### Scenario: Dependency declaration

- **WHEN** the project dependencies are declared in Cargo.toml
- **THEN** it SHALL include objc2-metal version 0.2 or higher
- **AND** it SHALL include objc2-foundation version 0.2 or higher  
- **AND** it SHALL include objc2 version 0.5 or higher
- **AND** it SHALL NOT include the deprecated metal crate
- **AND** it SHALL NOT include the deprecated objc 0.2 crate

#### Scenario: Feature flag configuration

- **WHEN** the metal-native feature is enabled
- **THEN** it SHALL enable objc2-metal as an optional dependency
- **AND** it SHALL enable objc2-foundation as an optional dependency
- **AND** it SHALL enable objc2 as an optional dependency
- **AND** all Metal-related code SHALL compile successfully

### Requirement: Type-Safe Metal Device Access

The system SHALL provide type-safe access to Metal GPU devices using objc2-metal's MTLDevice bindings with proper lifetime management and memory safety guarantees.

#### Scenario: Device initialization

- **WHEN** creating a Metal context
- **THEN** it SHALL use MTLCreateSystemDefaultDevice from objc2-metal
- **AND** it SHALL return a properly retained MTLDevice instance
- **AND** it SHALL handle the absence of Metal-capable devices gracefully
- **AND** device lifetime SHALL be managed through Arc for shared ownership

#### Scenario: Device capability queries

- **WHEN** querying device capabilities
- **THEN** it SHALL use MTLDevice methods from objc2-metal bindings
- **AND** it SHALL support GPU family checks (MTLGPUFamily)
- **AND** it SHALL query max threadgroup sizes correctly
- **AND** it SHALL retrieve device name as a String
- **AND** all capability information SHALL be accurate

### Requirement: Safe Buffer Management

The system SHALL manage Metal buffers using objc2-metal's safe buffer abstractions with proper resource options and storage modes.

#### Scenario: Buffer creation

- **WHEN** allocating GPU memory for vectors
- **THEN** it SHALL use MTLDevice::newBuffer from objc2-metal
- **AND** it SHALL specify appropriate MTLResourceOptions
- **AND** it SHALL use MTLStorageMode::Private for VRAM-only buffers
- **AND** buffer size SHALL be validated before allocation
- **AND** allocation failures SHALL be handled with Result types

#### Scenario: Buffer content updates

- **WHEN** writing data to Metal buffers
- **THEN** it SHALL use objc2-metal's safe buffer APIs
- **AND** it SHALL use staging buffers for Private storage mode
- **AND** it SHALL synchronize buffer operations correctly
- **AND** memory safety SHALL be enforced by the type system
- **AND** no unsafe code SHALL be required for standard buffer operations

### Requirement: Command Queue and Execution

The system SHALL manage command queue creation and command buffer execution using objc2-metal's safe abstractions.

#### Scenario: Command queue creation

- **WHEN** initializing the Metal context
- **THEN** it SHALL create a command queue using MTLDevice::newCommandQueue
- **AND** the command queue SHALL be retained for the context lifetime
- **AND** command queue creation SHALL be validated
- **AND** the queue SHALL be shared across operations via Arc

#### Scenario: Command buffer execution

- **WHEN** executing GPU operations
- **THEN** it SHALL create command buffers from the queue
- **AND** it SHALL encode compute commands using objc2 bindings
- **AND** it SHALL commit and wait for completion safely
- **AND** execution errors SHALL be propagated as Result types
- **AND** resource dependencies SHALL be managed correctly

### Requirement: Shader Library Compilation

The system SHALL compile Metal shader sources and load shader functions using objc2-metal's library APIs with proper error handling.

#### Scenario: Library creation from source

- **WHEN** loading Metal shaders
- **THEN** it SHALL use MTLDevice::newLibraryWithSource from objc2-metal
- **AND** it SHALL provide compile options if needed
- **AND** compilation errors SHALL be captured and reported
- **AND** the compiled library SHALL be cached in the context
- **AND** shader functions SHALL be loadable by name

#### Scenario: Compute pipeline creation

- **WHEN** creating compute pipelines
- **THEN** it SHALL use MTLComputePipelineDescriptor from objc2-metal
- **AND** it SHALL set the compute function from the library
- **AND** pipeline creation SHALL be validated
- **AND** pipeline state objects SHALL be cached for reuse
- **AND** pipeline errors SHALL provide actionable error messages

### Requirement: VRAM Monitoring

The system SHALL query and monitor VRAM usage using objc2-metal's device memory APIs.

#### Scenario: VRAM capacity queries

- **WHEN** retrieving GPU memory information
- **THEN** it SHALL use recommendedMaxWorkingSetSize from objc2-metal
- **AND** it SHALL query currentAllocatedSize for usage tracking
- **AND** it SHALL calculate available VRAM as recommended minus allocated
- **AND** memory values SHALL be returned in bytes
- **AND** queries SHALL be efficient and non-blocking

#### Scenario: Memory statistics reporting

- **WHEN** generating device info
- **THEN** GpuDeviceInfo SHALL include total_vram_bytes from recommendedMaxWorkingSetSize
- **AND** it SHALL include used_vram_bytes from currentAllocatedSize
- **AND** it SHALL calculate available_vram_bytes correctly
- **AND** statistics SHALL be accurate and up-to-date
- **AND** no platform-specific unsafe code SHALL be required

### Requirement: Backward Compatible Public API

The system SHALL maintain backward compatibility with the existing GpuContext and GpuVectorStorage public APIs despite internal Metal binding changes.

#### Scenario: Trait implementation preservation

- **WHEN** migrating to objc2-metal internally
- **THEN** all GpuBackend trait methods SHALL remain unchanged
- **AND** all GpuContext trait methods SHALL remain unchanged  
- **AND** all GpuVectorStorage trait methods SHALL remain unchanged
- **AND** external consumers SHALL NOT require code changes
- **AND** behavior SHALL be identical to metal-rs implementation

#### Scenario: Error handling consistency

- **WHEN** operations fail
- **THEN** error types SHALL remain HiveGpuError enum
- **AND** error messages SHALL be consistent with previous versions
- **AND** error handling patterns SHALL not change
- **AND** Result types SHALL be used consistently
- **AND** no new error variants SHALL be required for migration

### Requirement: Performance Parity

The system SHALL maintain equivalent or better performance compared to the metal-rs implementation for all GPU operations.

#### Scenario: Operation latency targets

- **WHEN** executing vector operations
- **THEN** search latency SHALL remain under 3ms
- **AND** batch operations SHALL maintain throughput targets
- **AND** buffer allocation overhead SHALL be negligible
- **AND** no performance regression SHALL be introduced
- **AND** benchmarks SHALL validate performance parity

#### Scenario: Memory efficiency

- **WHEN** managing GPU memory
- **THEN** buffer allocation patterns SHALL be optimal
- **AND** VRAM usage SHALL not increase vs metal-rs
- **AND** memory fragmentation SHALL be minimized
- **AND** buffer pooling SHALL work efficiently
- **AND** no memory leaks SHALL occur

### Requirement: Safe Rust Patterns

The system SHALL follow objc2's safety model and minimize unsafe code blocks while maintaining performance.

#### Scenario: Safe abstraction usage

- **WHEN** interacting with Metal APIs
- **THEN** it SHALL prefer objc2's safe wrapper methods
- **AND** unsafe blocks SHALL be minimized and documented
- **AND** lifetime management SHALL be enforced by the type system
- **AND** memory safety SHALL be guaranteed by Rust's borrow checker
- **AND** all unsafe code SHALL have safety comments explaining invariants

#### Scenario: Resource cleanup

- **WHEN** resources are no longer needed
- **THEN** Drop implementations SHALL ensure proper cleanup
- **AND** reference counting SHALL prevent premature deallocation
- **AND** no manual memory management SHALL be required
- **AND** resources SHALL be released deterministically
- **AND** cleanup SHALL be automatic and reliable

## MODIFIED Requirements

N/A - This is a new specification introducing Metal backend with objc2-metal.

## REMOVED Requirements

N/A - No requirements are being removed, only implementation details are changing.

## RENAMED Requirements

N/A - No requirements are being renamed.

