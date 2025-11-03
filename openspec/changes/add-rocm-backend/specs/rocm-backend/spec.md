# ROCm Backend Specification

## ADDED Requirements

### Requirement: ROCm/HIP Backend Support
The system SHALL provide a complete ROCm backend implementation for AMD GPUs with gfx900 or newer architecture (Vega and later).

#### Scenario: ROCm availability detection
- **WHEN** the application starts on a system with AMD GPUs and ROCm installed
- **THEN** `RocmContext::is_available()` returns true
- **AND** `RocmContext::device_count()` returns the number of available AMD GPUs
- **AND** the system can enumerate all ROCm devices via `hipGetDeviceCount`

#### Scenario: ROCm availability on non-AMD system
- **WHEN** the application runs on a system without AMD GPUs or ROCm
- **THEN** `RocmContext::is_available()` returns false
- **AND** `RocmContext::device_count()` returns 0
- **AND** attempting to create a context returns `HiveGpuError::NoDevice`

### Requirement: HIP Context Management
The system SHALL manage HIP device contexts, streams, and rocBLAS handles safely and efficiently.

#### Scenario: Create ROCm context with default device
- **WHEN** a user calls `RocmContext::new()`
- **THEN** the system selects GPU device 0
- **AND** creates a HIP stream for async operations via `hipStreamCreate`
- **AND** initializes a rocBLAS handle bound to the stream
- **AND** returns a valid context or an error

#### Scenario: Create ROCm context with specific device
- **WHEN** a user calls `RocmContext::new_with_device(device_id)`
- **THEN** the system validates the device ID exists
- **AND** sets the active HIP device via `hipSetDevice`
- **AND** creates stream and rocBLAS handle for that device
- **AND** returns error if device_id is invalid

#### Scenario: Clean up ROCm resources on drop
- **WHEN** a `RocmContext` goes out of scope
- **THEN** the system destroys the rocBLAS handle
- **AND** destroys the HIP stream
- **AND** does not leak GPU resources

### Requirement: ROCm Device Information
The ROCm backend SHALL implement the device info API to expose AMD GPU properties and capabilities.

#### Scenario: Query ROCm device properties
- **WHEN** `device_info()` is called on a ROCm context
- **THEN** the system queries device properties via `hipGetDeviceProperties()`
- **AND** queries memory info via `hipMemGetInfo()`
- **AND** returns `GpuDeviceInfo` with:
  - Device name (e.g., "AMD Radeon RX 7900 XTX", "AMD Instinct MI210")
  - Backend: "ROCm"
  - Total VRAM in bytes
  - Available VRAM in bytes
  - Used VRAM (calculated)
  - Driver version from ROCm
  - Compute capability: gfx architecture string (e.g., "gfx1030", "gfx90a")
  - Max threads per block (wavefront size * max waves)
  - Max shared memory per block (LDS size)
  - Device ID
  - PCI bus ID

### Requirement: HIP Vector Storage
The system SHALL implement GPU vector storage using HIP memory management with dynamic capacity.

#### Scenario: Add single vector to GPU via HIP
- **WHEN** a user calls `add_vector(vector)`
- **THEN** the system validates vector dimension matches storage dimension
- **AND** ensures GPU capacity is sufficient (grows if needed)
- **AND** copies vector data to GPU via `hipMemcpyAsync`
- **AND** stores vector metadata in host memory
- **AND** returns the vector's index

#### Scenario: Batch add multiple vectors with HIP
- **WHEN** a user calls `add_vectors(&[vector1, vector2, ...])`
- **THEN** the system validates all vectors have correct dimension
- **AND** ensures GPU capacity for all vectors
- **AND** flattens all vectors into a single buffer
- **AND** performs a single batch `hipMemcpyAsync` operation
- **AND** waits for transfer completion via `hipStreamSynchronize`
- **AND** stores all metadata

#### Scenario: Dynamic GPU memory growth with HIP
- **WHEN** adding vectors exceeds current GPU capacity
- **THEN** the system allocates new larger GPU buffer via `hipMalloc`
- **AND** copies existing data to new buffer with `hipMemcpyAsync`
- **AND** frees old buffer with `hipFree`
- **AND** updates capacity tracking
- **AND** continues with the add operation

### Requirement: HIP Distance Computation
The system SHALL compute vector distances efficiently using HIP kernels and rocBLAS operations.

#### Scenario: Cosine similarity via rocBLAS
- **WHEN** searching with Cosine or DotProduct metric
- **THEN** the system uses rocBLAS `SGEMV` for matrix-vector multiply
- **AND** computes dot products between query and all stored vectors
- **AND** returns results sorted by highest similarity first

#### Scenario: Euclidean distance via HIP kernel
- **WHEN** searching with Euclidean metric
- **THEN** the system calls custom `hip_l2_distance` kernel
- **AND** computes L2 distances in parallel on GPU
- **AND** optimizes for AMD wavefront size (64 threads)
- **AND** returns results sorted by lowest distance first

#### Scenario: AMD wavefront-optimized kernels
- **WHEN** HIP kernels execute on AMD GPUs
- **THEN** kernels are optimized for 64-thread wavefronts (not 32-thread warps)
- **AND** use LDS (Local Data Share) efficiently
- **AND** avoid divergence within wavefronts
- **AND** achieve coalesced memory access

### Requirement: ROCm Error Handling
All HIP and rocBLAS operations SHALL include comprehensive error checking and safe error propagation.

#### Scenario: HIP API error handling
- **WHEN** any HIP API call fails
- **THEN** the system checks return code via `hip_check()`
- **AND** converts HIP error to `HiveGpuError::RocmError` with message
- **AND** propagates error up the call stack
- **AND** does not panic

#### Scenario: rocBLAS error handling
- **WHEN** any rocBLAS operation fails
- **THEN** the system checks status via `rocblas_check()`
- **AND** converts to `HiveGpuError::RocblasError`
- **AND** provides meaningful error description

#### Scenario: Out of memory handling with HIP
- **WHEN** GPU memory allocation via `hipMalloc` fails
- **THEN** the system returns `HiveGpuError::OutOfMemory`
- **AND** does not leak partial allocations
- **AND** allows graceful recovery

### Requirement: HIP Kernel Implementation
Custom HIP kernels SHALL be efficient, safe, and optimized for AMD GPU architectures.

#### Scenario: L2 distance HIP kernel execution
- **WHEN** the L2 kernel is launched
- **THEN** each thread computes distance for one vector
- **AND** uses efficient block/thread configuration (256 threads/block typical)
- **AND** optimizes for 64-thread wavefronts
- **AND** performs vectorized operations where possible
- **AND** computes sqrt() for final distance

#### Scenario: Multi-architecture support for AMD
- **WHEN** HIP kernels are compiled
- **THEN** build.rs compiles for architectures gfx900 through gfx1100+
- **AND** includes Vega, RDNA, RDNA2, RDNA3, and CDNA support
- **AND** ROCm runtime selects optimal binary at runtime

### Requirement: ROCm Build System
The build system SHALL automatically compile HIP kernels during cargo build with ROCm feature.

#### Scenario: Detect ROCm installation
- **WHEN** building with `--features rocm`
- **THEN** build.rs checks for ROCM_PATH or ROCM_HOME
- **AND** falls back to /opt/rocm if not set
- **AND** fails gracefully with helpful message if ROCm not found

#### Scenario: Compile HIP kernels
- **WHEN** build.rs executes for ROCm
- **THEN** it compiles src/rocm/kernels.hip with hipcc
- **AND** generates code for multiple gfx* architectures
- **AND** links HIP runtime and rocBLAS libraries
- **AND** triggers recompilation if kernels.hip changes

### Requirement: Thread Safety and Async Support
HIP operations SHALL be thread-safe and support asynchronous execution.

#### Scenario: Concurrent operations on same ROCm context
- **WHEN** multiple threads access the same ROCm context
- **THEN** all HIP operations use the same stream (serialized)
- **AND** operations complete in order
- **AND** no race conditions occur

#### Scenario: Async memory transfers with HIP
- **WHEN** copying data to/from GPU
- **THEN** the system uses `hipMemcpyAsync` with stream
- **AND** operations overlap with computation when possible
- **AND** synchronizes only when results are needed via `hipStreamSynchronize`

### Requirement: Cross-Backend Consistency
ROCm backend SHALL produce identical results to CUDA and Metal backends for the same operations.

#### Scenario: Consistent search results across backends
- **WHEN** the same vectors are searched with the same query on ROCm, CUDA, and Metal
- **THEN** all backends return the same top-K vectors (order may vary for ties)
- **AND** distances are within floating-point error tolerance
- **AND** behavior is consistent across backends

## Implementation Notes

**Dependencies**:
```toml
[dependencies]
hip-runtime-sys = { version = "0.3", optional = true }
rocblas-sys = { version = "0.3", optional = true }

[features]
rocm = ["hip-runtime-sys", "rocblas-sys"]
```

**AMD GPU Architecture Support**:
- gfx900 (Vega 10, MI25)
- gfx906 (Vega 20, MI50/MI60)
- gfx908 (CDNA, MI100)
- gfx90a (CDNA2, MI200 series)
- gfx940 (CDNA3, MI300 series)
- gfx1030 (RDNA2, RX 6000 series)
- gfx1100 (RDNA3, RX 7000 series)

**Performance Targets**:
- Similar to CUDA performance on equivalent hardware
- Vector add: <0.1ms for 128-dim vector
- Search (1K vectors): <3ms
- Search (10K vectors): <8ms
- Batch add: >8K vectors/second

**Testing Requirements**:
- Unit tests for all public methods
- Integration tests with Metal and CUDA for consistency
- Performance benchmarks vs CUDA
- Error path testing
- Memory leak testing
- AMD-specific optimization validation

**Documentation Requirements**:
- ROCm setup guide (installation on Ubuntu/RHEL)
- AMD GPU compatibility matrix
- API documentation with examples
- Performance tuning guide for AMD GPUs
- Troubleshooting guide (rocm-smi, rocprof)
- Comparison with CUDA backend

