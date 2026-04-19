# ROCm Backend Specification

## ADDED Requirements

### Requirement: ROCm Runtime Detection

The system SHALL detect AMD GPUs via HIP runtime queries rather than
environment variables.

#### Scenario: ROCm device present

Given a Linux or Windows host with an AMD GPU and ROCm installed
When `RocmContext::is_available()` is called
Then it returns `true`
And `detect_available_backends()` includes `GpuBackendType::Rocm`

#### Scenario: ROCm absent

Given a host without ROCm or without an AMD GPU
When `RocmContext::is_available()` is called
Then it returns `false`
And no HIP library load failure surfaces to the caller

### Requirement: ROCm Device Information

The ROCm backend MUST populate `GpuDeviceInfo` from `hipGetDeviceProperties`.

#### Scenario: Query AMD device properties

Given a `RocmContext` created on a Radeon RX 7900 XTX
When `GpuContext::device_info()` is called
Then the returned `GpuDeviceInfo` has `backend == "ROCm"`
And `compute_capability` contains the gfx string (for example "gfx1100")
And `total_vram_bytes` matches the value reported by `hipMemGetInfo`
And `driver_version` reflects the installed ROCm version string

### Requirement: VRAM-Only Vector Storage

The ROCm backend SHALL store vector payloads in device-local GPU memory
via HIP allocations.

#### Scenario: Batch upload via HIP

Given a `RocmVectorStorage` with dimension 128 and metric Cosine
When `add_vectors` is called with 1000 vectors
Then all vectors reside in a single device pointer allocated by `hipMalloc`
And the upload uses a single `hipMemcpyAsync` call on the bound stream
And the call returns only after `hipStreamSynchronize`

#### Scenario: D2D reallocation on growth

Given a `RocmVectorStorage` at full capacity
When `add_vectors` is called with additional vectors
Then the backend allocates a larger buffer via `hipMalloc`
And copies existing data device-to-device via `hipMemcpyAsync`
And frees the old buffer via `hipFree`
And the growth factor matches the Metal backend pattern

### Requirement: Wavefront-Agnostic Kernels

HIP kernels MUST run correctly on both 32-thread wavefront hardware
(RDNA/RDNA2/RDNA3) and 64-thread wavefront hardware (Vega/CDNA).

#### Scenario: Reduction on RDNA hardware

Given a kernel running on gfx1030 (wavefront size 32)
When the kernel performs a subgroup reduction
Then it uses runtime `warpSize` for the reduction stride
And produces the same result as the equivalent CDNA execution

#### Scenario: Reduction on CDNA hardware

Given a kernel running on gfx90a (wavefront size 64)
When the kernel performs a subgroup reduction
Then it uses runtime `warpSize` for the reduction stride
And produces the same result as the equivalent RDNA execution

### Requirement: GPU Distance Computation

The ROCm backend SHALL compute vector distances on the GPU for all three
supported metrics, using rocBLAS when available.

#### Scenario: L2 distance via custom HIP kernel

Given a populated `RocmVectorStorage` with metric Euclidean
When `search(query, k)` is called
Then `hip_l2_distance_kernel` runs on the GPU for every stored vector
And the kernel uses shared LDS memory for reduction
And top-K results are produced on the CPU after score readback

#### Scenario: Cosine similarity via rocBLAS

Given a populated `RocmVectorStorage` with metric Cosine
When `search(query, k)` is called
Then the backend invokes `rocblas_sgemv` to compute dot products
And a separate normalization kernel converts dot products to cosine
And top-K results are produced on the CPU after score readback

### Requirement: Cross-Backend Numerical Consistency

The ROCm backend MUST agree with Metal and CUDA on top-K results within
floating-point tolerance.

#### Scenario: Metal, CUDA, and ROCm agree on top-K

Given the same 1000 random vectors in 128 dimensions indexed on Metal,
CUDA, and ROCm backends
When the same query is searched on all three backends
Then the top-10 result sets are identical as sets across all backends
And per-element distance values agree within 1e-4 absolute tolerance

### Requirement: Error Propagation

All HIP and rocBLAS failures MUST be propagated as typed errors without
panicking.

#### Scenario: hipMalloc failure during vector add

Given a `RocmVectorStorage` near the VRAM limit
When `add_vectors` triggers a `hipMalloc` that returns `hipErrorOutOfMemory`
Then the call returns `Err(HiveGpuError::HipError(_))`
And no partial allocation is retained
And subsequent calls continue to work

#### Scenario: rocBLAS failure during search

Given a `RocmVectorStorage` using rocBLAS SGEMV
When `rocblas_sgemv` returns a non-success status
Then the call returns `Err(HiveGpuError::RocblasError(_))`
And the error message includes the rocBLAS status code

### Requirement: Build System for HIP Kernels

The `build.rs` MUST locate the ROCm installation and compile HIP kernels
into multi-architecture PTX at build time.

#### Scenario: ROCm installation detected

Given a host with `ROCM_PATH` set or `/opt/rocm` present
When `cargo build --features rocm` runs
Then `build.rs` locates `hipcc` from the ROCm install
And invokes `hipcc --offload-arch=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100`
And links `libamdhip64` and `librocblas`

#### Scenario: hipcc missing, source-JIT fallback

Given a host without `hipcc` in PATH but with the HIP runtime installed
When `cargo build --features rocm` runs
Then `build.rs` proceeds without compiling kernels
And the backend loads HIP source at runtime via `hipModuleLoadData` and JIT-compiles it

### Requirement: Feature Flag Isolation

The ROCm backend MUST be gated behind the `rocm` Cargo feature and only
compile on Linux and Windows.

#### Scenario: Build without ROCm feature

Given a project depending on `hive-gpu` with default features only
When the project is built on any OS
Then no ROCm dependencies are pulled
And Metal, CUDA, and CPU paths remain unchanged

#### Scenario: ROCm feature on macOS

Given a macOS host
When a user attempts to build `hive-gpu` with `--features rocm`
Then the ROCm module is excluded via `#[cfg(any(target_os = "linux", target_os = "windows"))]`
And the build succeeds without attempting to load HIP libraries
