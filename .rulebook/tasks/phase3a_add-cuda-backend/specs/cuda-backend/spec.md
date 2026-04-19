# CUDA Backend Specification

## ADDED Requirements

### Requirement: CUDA Runtime Detection

The system SHALL detect NVIDIA CUDA-capable devices at runtime via the
CUDA driver API rather than environment variables.

#### Scenario: CUDA device present

Given a host with an NVIDIA GPU and a compatible CUDA driver installed
When `CudaContext::is_available()` is called
Then it returns `true`
And `detect_available_backends()` includes `GpuBackendType::Cuda`

#### Scenario: CUDA driver absent

Given a host without an NVIDIA driver
When `CudaContext::is_available()` is called
Then it returns `false`
And `detect_available_backends()` does not include `GpuBackendType::Cuda`

### Requirement: CUDA Device Information

The CUDA backend MUST populate `GpuDeviceInfo` from live driver queries.

#### Scenario: Query device properties

Given a `CudaContext` created on device 0
When `GpuContext::device_info()` is called
Then the returned `GpuDeviceInfo` has `backend == "CUDA"`
And `compute_capability` reflects the real sm_* string from `cuDeviceGetAttribute`
And `total_vram_bytes` matches `cuDeviceTotalMem`
And `available_vram_bytes` is computed from `cuMemGetInfo`

### Requirement: VRAM-Only Vector Storage

The CUDA backend SHALL store vector payloads in device-local GPU memory
with host staging for uploads.

#### Scenario: Add batch of vectors

Given a `CudaVectorStorage` with dimension 128 and metric Cosine
When `add_vectors` is called with 1000 vectors
Then all vectors reside in a single `DeviceSlice<f32>` buffer
And the upload uses a single `htod_copy` call
And `vector_count()` returns 1000

#### Scenario: Buffer expansion on overflow

Given a `CudaVectorStorage` at full capacity
When `add_vectors` is called with additional vectors
Then the backend allocates a larger `DeviceSlice<f32>` via `cudaMalloc`
And copies existing data device-to-device via `cudaMemcpyAsync`
And frees the old buffer via `cudaFree`
And the growth factor matches the Metal backend pattern (2.0/1.5/1.2)

### Requirement: GPU Distance Computation

The CUDA backend MUST compute vector distances on the GPU for all three
supported metrics.

#### Scenario: L2 distance search

Given a populated `CudaVectorStorage` with metric Euclidean
When `search(query, k)` is called
Then `l2_distance_kernel` runs on the GPU for every stored vector
And the kernel uses 256 threads per block
And top-K results are produced on the CPU after score readback

#### Scenario: Cosine similarity search

Given a populated `CudaVectorStorage` with metric Cosine
When `search(query, k)` is called
Then the backend computes dot products via cuBLAS SGEMV or a fused kernel
And normalizes the results to cosine similarity
And top-K results are produced on the CPU after score readback

### Requirement: Cross-Backend Numerical Consistency

The CUDA backend SHALL produce results numerically consistent with the
Metal backend for identical inputs.

#### Scenario: Metal and CUDA agree on top-K

Given the same 1000 random vectors in 128 dimensions indexed on both
Metal and CUDA backends
When the same query is searched on both backends
Then the top-10 result sets are identical as sets
And per-element distance values agree within 1e-4 absolute tolerance

### Requirement: Error Propagation

All CUDA and cuBLAS failures MUST be propagated as typed errors without
panicking.

#### Scenario: Out-of-memory during vector add

Given a `CudaVectorStorage` near the VRAM limit
When `add_vectors` triggers a `cudaMalloc` that returns `cudaErrorMemoryAllocation`
Then the call returns `Err(HiveGpuError::CudaError(_))`
And no partial allocation is retained
And subsequent calls continue to work

#### Scenario: cuBLAS failure during search

Given a `CudaVectorStorage` using cuBLAS SGEMV
When `cublasSgemv` returns a non-success status
Then the call returns `Err(HiveGpuError::CublasError(_))`
And the error message includes the cuBLAS status code

### Requirement: PTX Kernel Distribution

The CUDA backend SHALL ship compute kernels as embedded PTX so end-user
builds do not require `nvcc` on the host.

#### Scenario: Kernel load without nvcc

Given a user with only the CUDA driver installed (no toolkit / nvcc)
When the user builds `hive-gpu` with `--features cuda`
Then the build succeeds without invoking `nvcc`
And the PTX is loaded from embedded `include_str!` data at runtime

### Requirement: Feature Flag Isolation

The CUDA backend MUST be gated behind the `cuda` Cargo feature without
affecting users on other backends.

#### Scenario: Build without CUDA feature

Given a project depending on `hive-gpu` with default features only
When the project is built on Linux or Windows
Then no CUDA dependencies are pulled
And the binary contains no CUDA symbols
And Metal and CPU paths remain unchanged
