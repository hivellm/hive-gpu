# CUDA Backend Specification

## ADDED Requirements

### Requirement: CUDA Backend Support
The system SHALL provide a complete CUDA backend implementation for NVIDIA GPUs with compute capability 7.0 or higher.

#### Scenario: CUDA availability detection
- **WHEN** the application starts on a system with NVIDIA GPUs
- **THEN** `CudaContext::is_available()` returns true
- **AND** `CudaContext::device_count()` returns the number of available GPUs
- **AND** the system can enumerate all CUDA devices

#### Scenario: CUDA availability on non-NVIDIA system
- **WHEN** the application runs on a system without NVIDIA GPUs
- **THEN** `CudaContext::is_available()` returns false
- **AND** `CudaContext::device_count()` returns 0
- **AND** attempting to create a context returns `HiveGpuError::NoDevice`

### Requirement: CUDA Context Management
The system SHALL manage CUDA device contexts, streams, and cuBLAS handles safely and efficiently.

#### Scenario: Create CUDA context with default device
- **WHEN** a user calls `CudaContext::new()`
- **THEN** the system selects GPU device 0
- **AND** creates a CUDA stream for async operations
- **AND** initializes a cuBLAS handle bound to the stream
- **AND** returns a valid context or an error

#### Scenario: Create CUDA context with specific device
- **WHEN** a user calls `CudaContext::new_with_device(device_id)`
- **THEN** the system validates the device ID exists
- **AND** sets the active CUDA device
- **AND** creates stream and cuBLAS handle for that device
- **AND** returns error if device_id is invalid

#### Scenario: Clean up CUDA resources on drop
- **WHEN** a `CudaContext` goes out of scope
- **THEN** the system destroys the cuBLAS handle
- **AND** destroys the CUDA stream
- **AND** does not leak GPU resources

### Requirement: CUDA Device Information
The CUDA backend SHALL implement the device info API to expose GPU properties and capabilities.

#### Scenario: Query CUDA device properties
- **WHEN** `device_info()` is called on a CUDA context
- **THEN** the system queries device properties via `cudaGetDeviceProperties()`
- **AND** queries memory info via `cudaMemGetInfo()`
- **AND** returns `GpuDeviceInfo` with:
  - Device name (e.g., "NVIDIA RTX 4090")
  - Backend: "CUDA"
  - Total VRAM in bytes
  - Available VRAM in bytes
  - Used VRAM (calculated)
  - Driver version (e.g., "12.0")
  - Compute capability (e.g., "8.9")
  - Max threads per block
  - Max shared memory per block
  - Device ID
  - PCI bus ID (formatted as domain:bus:device.0)

### Requirement: CUDA Vector Storage
The system SHALL implement GPU vector storage using CUDA memory management with dynamic capacity.

#### Scenario: Add single vector to GPU
- **WHEN** a user calls `add_vector(vector)`
- **THEN** the system validates vector dimension matches storage dimension
- **AND** ensures GPU capacity is sufficient (grows if needed)
- **AND** copies vector data to GPU via `cudaMemcpyAsync`
- **AND** stores vector metadata in host memory
- **AND** returns the vector's index

#### Scenario: Batch add multiple vectors
- **WHEN** a user calls `add_vectors(&[vector1, vector2, ...])`
- **THEN** the system validates all vectors have correct dimension
- **AND** ensures GPU capacity for all vectors
- **AND** flattens all vectors into a single buffer
- **AND** performs a single batch `cudaMemcpyAsync` operation
- **AND** waits for transfer completion via `cudaStreamSynchronize`
- **AND** stores all metadata

#### Scenario: Dynamic GPU memory growth
- **WHEN** adding vectors exceeds current GPU capacity
- **THEN** the system allocates new larger GPU buffer (next power of 2)
- **AND** copies existing data to new buffer
- **AND** frees old buffer
- **AND** updates capacity tracking
- **AND** continues with the add operation

### Requirement: CUDA Distance Computation
The system SHALL compute vector distances efficiently using CUDA kernels and cuBLAS operations.

#### Scenario: Cosine similarity via cuBLAS
- **WHEN** searching with Cosine or DotProduct metric
- **THEN** the system uses cuBLAS `SGEMV` for matrix-vector multiply
- **AND** computes dot products between query and all stored vectors
- **AND** returns results sorted by highest similarity first

#### Scenario: Euclidean distance via CUDA kernel
- **WHEN** searching with Euclidean metric
- **THEN** the system calls custom `cuda_l2_distance` kernel
- **AND** computes L2 distances in parallel on GPU
- **AND** returns results sorted by lowest distance first

#### Scenario: Top-K search results
- **WHEN** a user searches for top-K vectors
- **THEN** the system computes distances for all vectors on GPU
- **AND** transfers results back to host memory
- **AND** sorts results by distance (metric-dependent order)
- **AND** returns exactly K results (or fewer if less than K vectors exist)

### Requirement: CUDA Error Handling
All CUDA operations SHALL include comprehensive error checking and safe error propagation.

#### Scenario: CUDA API error handling
- **WHEN** any CUDA API call fails
- **THEN** the system checks return code via `cuda_check()`
- **AND** converts CUDA error to `HiveGpuError::CudaError` with message
- **AND** propagates error up the call stack
- **AND** does not panic

#### Scenario: cuBLAS error handling
- **WHEN** any cuBLAS operation fails
- **THEN** the system checks status via `cublas_check()`
- **AND** converts to `HiveGpuError::CublasError`
- **AND** provides meaningful error description

#### Scenario: Out of memory handling
- **WHEN** GPU memory allocation fails
- **THEN** the system returns `HiveGpuError::OutOfMemory`
- **AND** does not leak partial allocations
- **AND** allows graceful recovery

### Requirement: CUDA Kernel Implementation
Custom CUDA kernels SHALL be efficient, safe, and support multiple GPU architectures.

#### Scenario: L2 distance kernel execution
- **WHEN** the L2 kernel is launched
- **THEN** each thread computes distance for one vector
- **AND** uses efficient block/thread configuration (256 threads/block)
- **AND** performs vectorized operations where possible
- **AND** computes sqrt() for final distance

#### Scenario: Multi-architecture support
- **WHEN** CUDA kernels are compiled
- **THEN** build.rs compiles for architectures sm_70 through sm_90
- **AND** includes Volta, Turing, Ampere, Ada, and Hopper support
- **AND** NVIDIA driver selects optimal binary at runtime

### Requirement: CUDA Build System
The build system SHALL automatically compile CUDA kernels during cargo build.

#### Scenario: Detect CUDA installation
- **WHEN** building with `--features cuda`
- **THEN** build.rs checks for CUDA_PATH or CUDA_HOME
- **AND** falls back to /usr/local/cuda if not set
- **AND** fails gracefully with helpful message if CUDA not found

#### Scenario: Compile CUDA kernels
- **WHEN** build.rs executes
- **THEN** it compiles src/cuda/kernels.cu with nvcc
- **AND** generates code for multiple sm_* architectures
- **AND** links cudart and cublas libraries
- **AND** triggers recompilation if kernels.cu changes

### Requirement: Thread Safety and Async Support
CUDA operations SHALL be thread-safe and support asynchronous execution.

#### Scenario: Concurrent operations on same context
- **WHEN** multiple threads access the same CUDA context
- **THEN** all CUDA operations use the same stream (serialized)
- **AND** operations complete in order
- **AND** no race conditions occur

#### Scenario: Async memory transfers
- **WHEN** copying data to/from GPU
- **THEN** the system uses `cudaMemcpyAsync` with stream
- **AND** operations overlap with computation when possible
- **AND** synchronizes only when results are needed

## Implementation Notes

**Dependencies**:
```toml
[dependencies]
cuda-runtime-sys = { version = "0.3", optional = true }
cuda-driver-sys = { version = "0.3", optional = true }
cublas-sys = { version = "0.3", optional = true }

[build-dependencies]
cc = "1.0"

[features]
cuda = ["cuda-runtime-sys", "cuda-driver-sys", "cublas-sys"]
```

**Safety Requirements**:
- All raw pointer operations in `unsafe` blocks
- Validate all device pointers before use
- Ensure proper memory cleanup in Drop
- Check CUDA error codes for every API call

**Performance Targets**:
- Vector add: <0.1ms for 128-dim vector
- Search (1K vectors): <2ms
- Search (10K vectors): <5ms
- Search (100K vectors): <50ms
- Batch add: >10K vectors/second

**Testing Requirements**:
- Unit tests for all public methods
- Integration tests with Metal backend for consistency
- Performance benchmarks vs CPU and Metal
- Error path testing
- Memory leak testing
- Multi-GPU testing (if available)

**Documentation Requirements**:
- Setup guide (CUDA Toolkit installation)
- API documentation with examples
- Performance tuning guide
- Troubleshooting guide
- Architecture-specific optimizations

