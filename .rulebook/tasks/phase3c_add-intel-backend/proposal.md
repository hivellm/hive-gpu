# Proposal: phase3c_add-intel-backend

Source: docs/analysis/intel/

## Why

After CUDA (~70%) and ROCm (~15%) ship, the largest remaining GPU vendor
is Intel. Intel Arc Pro B70 (32 GB GDDR6, 22.9 TFLOPS) shipped in March
2026 for the workstation segment, and Crescent Island (Xe3P, 160 GB
LPDDR5X) samples in Q3 2026 for the datacenter. Market share in ML / AI
vector search is <1% today, so this is a differentiator investment, not
a coverage investment — Intel support closes the vendor matrix and
unlocks customers with Arc Pro fleets that have no other path.

The task must start only after phase3a (CUDA) and phase3b (ROCm) are
merged. It inherits the `build.rs` and cross-backend consistency harness
those tasks create.

A valuable side effect: the Vulkan Compute path chosen here also runs on
NVIDIA and AMD Vulkan drivers, providing a universal fallback for
environments without the native toolchains installed (Docker images
without CUDA / ROCm, CI runners without vendor SDKs).

## What Changes

- Add `src/intel/` module using **Vulkan Compute via `ash`** as the API
  layer. Level Zero is the native Intel path but has no maintained Rust
  binding in 2026; DPC++/SYCL is C++ only; OpenCL was ruled out.
- Author compute kernels in GLSL and compile to SPIR-V at build time via
  `shaderc`. Kernel sources live in `src/intel/shaders/*.comp` and SPIR-V
  is embedded via `include_bytes!`. Alternative path (reuse existing
  `src/shaders/*.wgsl` via `naga`) is a Phase 1 audit decision.
- Implement `IntelContext` creating a Vulkan instance + compute-capable
  physical device filtered by `vendorID == 0x8086` (with an
  `HIVE_GPU_VULKAN_UNIVERSAL` env var to relax the filter for the
  universal-fallback mode).
- Populate `GpuDeviceInfo` from `VkPhysicalDeviceProperties` +
  `VK_KHR_driver_properties`. `compute_capability` receives the Vulkan
  device name / driver ID combination.
- Implement `IntelVectorStorage` with device-local `VkBuffer` + a
  reusable host-visible staging buffer. Uploads via `vkCmdCopyBuffer`
  submitted to the transfer queue with `VkFence` synchronization.
- Ship three compute kernels: `l2_distance.comp`, `cosine_similarity.comp`,
  `dot_product.comp`. No rocBLAS / cuBLAS equivalent is reachable from
  Rust on Intel, so SGEMV is hand-written — budget extra time vs. ROCm.
- Add `GpuBackendType::Intel` with priority Metal > CUDA > ROCm > Intel
  > CPU. Expose `HIVE_GPU_BACKEND` env var for explicit override.
- Add `HiveGpuError::{VulkanError, IntelError, SpirvCompileError}`.
- Extend `tests/cross_backend_consistency.rs` to include Intel within
  the `1e-4` tolerance envelope. Validate on Arc B580 (consumer) and
  Arc Pro B70 (workstation) before merge.
- CI using a Linux runner with `lavapipe` (software Vulkan) for build
  verification; real GPU tests require a self-hosted runner with an
  Intel discrete card.

## Impact

- Affected specs: new `intel-backend` spec.
- Affected code: new `src/intel/*`, updates to `src/lib.rs`,
  `src/error.rs`, `src/backends/detector.rs`, `Cargo.toml`, `build.rs`
  (add `shaderc` invocation), new `tests/intel_*.rs`, new
  `examples/intel_basic.rs`, new `docs/guides/INTEL_SETUP.md`,
  new `docs/guides/BACKEND_SELECTION.md`.
- Breaking change: NO. Feature-gated behind `intel`.
- User benefit: production-ready Intel Arc acceleration on Xe-HPG
  (Alchemist) and Xe2 (Battlemage / Arc Pro) hardware, plus a universal
  Vulkan fallback that runs on NVIDIA / AMD when native toolchains are
  unavailable. Covers Intel-specific workstation customers and future
  Crescent Island datacenter deployments.
- Performance expectation: 40–60% of native CUDA/ROCm on equivalent
  silicon price. Document honestly — this is not a replacement for
  vendor-native paths on their own hardware.
- HNSW is not part of v1. XMX matrix engines (Battlemage, Arc Pro) are
  not used in v1 and remain on the v2 watch list.
