# Intel Backend Specification

## ADDED Requirements

### Requirement: Intel GPU Detection via Vulkan

The system SHALL detect Intel GPUs by enumerating Vulkan physical
devices and filtering by Intel vendor ID.

#### Scenario: Intel Arc discrete GPU present

Given a Linux or Windows host with an Intel Arc GPU and Vulkan driver
When `IntelContext::is_available()` is called
Then Vulkan is loaded lazily via `ash::Entry`
And at least one physical device with `vendorID == 0x8086` is found
And the function returns `true`

#### Scenario: No Intel GPU, strict mode

Given a host without an Intel GPU and `HIVE_GPU_VULKAN_UNIVERSAL` unset
When `IntelContext::is_available()` is called
Then it returns `false`
And no other Vulkan device is selected

#### Scenario: Universal fallback mode

Given a host without an Intel GPU but with a Vulkan-capable NVIDIA or AMD GPU
When `HIVE_GPU_VULKAN_UNIVERSAL=1` is set
And `IntelContext::is_available()` is called
Then the vendor filter is relaxed
And the function returns `true`
And a non-Intel Vulkan device is accepted as the compute target

### Requirement: Discrete GPU Preference

When multiple Vulkan devices are available, the backend MUST prefer
discrete GPUs over integrated GPUs.

#### Scenario: Integrated Xe plus discrete Arc

Given a laptop with both Iris Xe integrated and Arc discrete GPUs
When `IntelContext::new()` is called
Then the discrete device (`VkPhysicalDeviceType::DiscreteGpu`) is selected
And the integrated device is ignored

### Requirement: Intel Device Information

The Intel backend MUST populate `GpuDeviceInfo` from Vulkan physical
device properties.

#### Scenario: Query Arc properties

Given an `IntelContext` created on an Arc Pro B70
When `GpuContext::device_info()` is called
Then the returned `GpuDeviceInfo` has `backend == "Intel"`
And `total_vram_bytes` matches the device-local heap size
And `driver_version` is populated from `VK_KHR_driver_properties`
And `max_threads_per_block` reflects `maxComputeWorkGroupInvocations`

### Requirement: VRAM-Only Vector Storage

The Intel backend SHALL store vector payloads in Vulkan device-local
memory using `VkBuffer` allocations.

#### Scenario: Batch upload via staging buffer

Given an `IntelVectorStorage` with dimension 128 and metric Cosine
When `add_vectors` is called with 1000 vectors
Then all vectors reside in a single device-local `VkBuffer`
And the upload uses a reusable host-visible staging buffer
And the transfer is synchronized via `VkFence` before the call returns

#### Scenario: D2D reallocation on growth

Given an `IntelVectorStorage` at full capacity
When `add_vectors` is called with additional vectors
Then the backend allocates a larger `VkBuffer` in device-local memory
And copies existing data via `vkCmdCopyBuffer` on the transfer queue
And frees the old buffer and its memory allocation
And the growth factor matches the Metal / CUDA / ROCm pattern

### Requirement: Descriptor-Set Caching

The backend MUST pool descriptor sets to avoid per-search reallocation.

#### Scenario: Repeated searches reuse descriptors

Given an `IntelVectorStorage` with existing descriptor sets
When `search(query, k)` is called repeatedly
Then the same descriptor sets are rebound rather than reallocated
And the descriptor pool does not grow unbounded

### Requirement: GPU Distance Computation

The Intel backend SHALL compute vector distances using SPIR-V compute
shaders for all three supported metrics.

#### Scenario: L2 distance via GLSL kernel

Given a populated `IntelVectorStorage` with metric Euclidean
When `search(query, k)` is called
Then the `l2_distance.comp` compute pipeline runs on the GPU
And top-K results are produced on the CPU after score readback

#### Scenario: Cosine similarity without BLAS

Given a populated `IntelVectorStorage` with metric Cosine
When `search(query, k)` is called
Then the `cosine_similarity.comp` compute pipeline runs on the GPU
And the kernel implements SGEMV + normalization by hand
And top-K results are produced on the CPU after score readback

### Requirement: Cross-Backend Numerical Consistency

The Intel backend MUST agree with Metal, CUDA, and ROCm on top-K results
within floating-point tolerance.

#### Scenario: All backends agree on top-K

Given the same 1000 random vectors in 128 dimensions indexed on Metal,
CUDA, ROCm, and Intel backends
When the same query is searched on all backends
Then the top-10 result sets are identical as sets across all backends
And per-element distance values agree within 1e-4 absolute tolerance
And the result holds on both Arc B580 (consumer) and Arc Pro B70 (workstation)

### Requirement: Error Propagation

All Vulkan failures MUST be propagated as typed errors without panicking.

#### Scenario: Vulkan allocation failure

Given an `IntelVectorStorage` near the VRAM limit
When `add_vectors` triggers a buffer allocation that fails with
`VK_ERROR_OUT_OF_DEVICE_MEMORY`
Then the call returns `Err(HiveGpuError::VulkanError(_))`
And no partial allocation is retained
And subsequent calls continue to work

#### Scenario: SPIR-V compile failure

Given an Intel backend where an embedded SPIR-V blob is corrupted
When `IntelContext::new()` attempts to create a compute pipeline
Then the call returns `Err(HiveGpuError::SpirvCompileError(_))`
And the error includes the Vulkan validation layer message

### Requirement: SPIR-V Build Pipeline

The `build.rs` MUST compile GLSL compute shaders to SPIR-V at build
time and embed them in the binary.

#### Scenario: Build with shaderc available

Given a host with the Vulkan SDK or `shaderc-sys` bundled binaries
When `cargo build --features intel` runs
Then `build.rs` compiles each `src/intel/shaders/*.comp` to SPIR-V
And the output is embedded via `include_bytes!`

#### Scenario: Build without shaderc available

Given a host where `shaderc` fails to build
When `cargo build --features intel` runs
Then `build.rs` falls back to checked-in SPIR-V blobs in the repo
And the build succeeds without recompiling shaders

### Requirement: Feature Flag and Platform Isolation

The Intel backend MUST be gated behind the `intel` Cargo feature and
only compile on Linux and Windows.

#### Scenario: Build without Intel feature

Given a project depending on `hive-gpu` with default features only
When the project is built on any OS
Then no Vulkan dependencies are pulled
And Metal, CUDA, ROCm, and CPU paths remain unchanged

#### Scenario: Intel feature on macOS

Given a macOS host
When a user attempts to build `hive-gpu` with `--features intel`
Then the Intel module is excluded via `#[cfg(any(target_os = "linux", target_os = "windows"))]`
And the build succeeds without pulling Vulkan dependencies
