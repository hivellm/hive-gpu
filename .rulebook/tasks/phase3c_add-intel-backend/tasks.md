## 1. Infrastructure
- [ ] 1.1 Add `intel` feature in `Cargo.toml` with `ash` and build-dep `shaderc`
- [ ] 1.2 Add `HiveGpuError::{VulkanError, IntelError, SpirvCompileError}` variants in `src/error.rs`
- [ ] 1.3 Add `GpuBackendType::Intel` to `src/backends/detector.rs` with priority after ROCm
- [ ] 1.4 Create empty scaffolding `src/intel/{mod.rs,context.rs,vector_storage.rs,buffer_pool.rs,kernels.rs,vram_monitor.rs}`
- [ ] 1.5 Wire `#[cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))] pub mod intel;` into `src/lib.rs`
- [ ] 1.6 Decide and document the kernel source path (GLSL + shaderc vs. existing WGSL + naga)

## 2. Vulkan Context
- [ ] 2.1 Implement `IntelContext::new` with Vulkan instance + logical device + compute queue
- [ ] 2.2 Filter physical devices by `vendorID == 0x8086` (override via `HIVE_GPU_VULKAN_UNIVERSAL=1`)
- [ ] 2.3 Prefer `VkPhysicalDeviceType::DiscreteGpu` over integrated when both are present
- [ ] 2.4 Enable `VK_KHR_driver_properties` extension for driver version string
- [ ] 2.5 Populate `GpuDeviceInfo` from `VkPhysicalDeviceProperties` + memory heap info
- [ ] 2.6 Implement `is_available()` via lazy Vulkan loader (ash `Entry::load`)
- [ ] 2.7 Enable validation layers in debug builds, disable in release

## 3. Vector Storage
- [ ] 3.1 Implement `IntelVectorStorage` with device-local `VkBuffer` backing
- [ ] 3.2 Maintain a reusable host-visible staging buffer for uploads
- [ ] 3.3 Batched `add_vectors` via `vkCmdCopyBuffer` + `vkQueueSubmit` + `VkFence`
- [ ] 3.4 Dynamic buffer expansion with D2D `vkCmdCopyBuffer` mirroring Metal/CUDA/ROCm
- [ ] 3.5 Soft-delete via `removed_indices: HashSet<usize>`
- [ ] 3.6 Descriptor-set cache to avoid reallocating on every search

## 4. Compute Kernels
- [ ] 4.1 Author `src/intel/shaders/l2_distance.comp` (GLSL compute)
- [ ] 4.2 Author `src/intel/shaders/cosine_similarity.comp` with hand-written SGEMV
- [ ] 4.3 Author `src/intel/shaders/dot_product.comp`
- [ ] 4.4 Compile GLSL to SPIR-V in `build.rs` using `shaderc` and embed via `include_bytes!`
- [ ] 4.5 Provide prebuilt SPIR-V blobs in the repo as a fallback when `shaderc` is unavailable
- [ ] 4.6 Rust launcher in `src/intel/kernels.rs`: pipeline creation, descriptor binding, dispatch
- [ ] 4.7 Implement CPU-side top-K after score readback

## 5. Consistency and Benchmarks
- [ ] 5.1 Extend `tests/cross_backend_consistency.rs` to include Intel within 1e-4 tolerance
- [ ] 5.2 Validate on Arc B580 (Battlemage consumer) and Arc Pro B70 (workstation) before merge
- [ ] 5.3 Extend `benches/gpu_operations.rs` with Intel variants gated by `intel` feature
- [ ] 5.4 Record baseline numbers honestly, documenting the 40-60% native-parity expectation

## 6. CI and Build Verification
- [ ] 6.1 Add GitHub Actions job with `lavapipe` (software Vulkan) for build verification
- [ ] 6.2 Gate runtime tests behind `IntelContext::is_available()` for CI hosts without GPU
- [ ] 6.3 Verify `cargo clippy --features intel -- -D warnings` is clean
- [ ] 6.4 Verify `cargo fmt --all --check` is clean
- [ ] 6.5 Confirm Vulkan validation layers produce zero warnings in debug runs

## 7. Universal Vulkan Fallback Mode
- [ ] 7.1 Document `HIVE_GPU_VULKAN_UNIVERSAL=1` in `docs/guides/BACKEND_SELECTION.md`
- [ ] 7.2 Verify the backend runs on NVIDIA and AMD Vulkan drivers when filter is relaxed
- [ ] 7.3 Add a test case covering the universal mode path

## 8. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 8.1 Update or create documentation covering the implementation
- [ ] 8.2 Write tests covering the new behavior
- [ ] 8.3 Run tests and confirm they pass
