# 05 — Architectural Decisions

## 5.1 Binding / API layer choice

Four candidate paths, ranked by viability for a pure-Rust crate in 2026:

| # | Path | Rust support | Pros | Cons |
|---|---|---|---|---|
| 1 | **Vulkan Compute via `ash`** | Mature (`ash 0.37+`) | Single codebase for Intel + NVIDIA + AMD; well-documented; validation layers; huge install base | 20–40% slower than native on NVIDIA/AMD; SPIR-V kernel pipeline required |
| 2 | **Level Zero via `bindgen`** | Immature (no maintained crate) | Native Intel low-level API; same API DPC++ uses underneath; lower overhead than Vulkan | Must hand-maintain bindings; Intel-only; smaller community |
| 3 | **DPC++/SYCL FFI shim** | None (C++ only) | Access to oneMKL GPU, joint_matrix, full Intel toolchain | Requires C++ build; two languages to maintain; undermines pure-Rust story |
| 4 | **OpenCL via `ocl`** | Mature (`ocl 0.19`) | Drop-in vendor-neutral; Intel runtime implements OpenCL 3.0 | Legacy; Apple deprecated; NVIDIA frozen at subset; poor BLAS story — already rejected in prior discussion |

**Recommendation:** **Vulkan Compute via `ash`** is the only path that combines mature Rust bindings with practical Intel coverage in 2026. The 20–40% performance penalty on NVIDIA/AMD is irrelevant because those vendors already have their own backends in this project.

## 5.2 Kernel language and build pipeline

Three candidates for authoring compute kernels:

1. **GLSL compute shaders + `shaderc` crate** — compile at build time, embed SPIR-V.
   - Pros: GLSL is widely known; `shaderc` is mature; debugging with `spirv-cross` is easy.
   - Cons: introduces a second language into the repo; `shaderc` pulls `shaderc-sys` which requires the Vulkan SDK on the build host for some configurations (provide prebuilt alternatives).
2. **`rust-gpu` (Rust → SPIR-V)** — write kernels as Rust functions.
   - Pros: single language across the entire crate; type safety extends to shaders.
   - Cons: `rust-gpu` is still 0.x; SPIR-T IR is relatively new; production usage is growing but not yet mainstream.
3. **Hand-written SPIR-V assembly (`spirv-asm`)** — expert-only.
   - Pros: no external compiler.
   - Cons: unmaintainable for non-trivial kernels.

**Recommendation:** **start with GLSL + `shaderc`** for kernel authoring because the algorithmic reference (the WGSL files in [src/shaders/](../../../src/shaders/)) is easy to translate line-by-line to GLSL. **Evaluate `rust-gpu` in v2** once the kernels stabilize; the migration is incremental (one kernel at a time).

Alternative worth flagging: use **Naga as a library** (not as part of `wgpu` runtime) to compile the existing WGSL directly to SPIR-V at build time. This avoids rewriting kernels and honors the project's "no web/wgpu runtime" constraint — Naga is a standalone Rust crate. **This is the most leveraged option if the WGSL files are production-quality.** They need to be audited first (see [10-next-steps.md](10-next-steps.md)).

## 5.3 Memory model mapping

Same contract as Metal ([vector_storage.rs:50-102](../../../src/metal/vector_storage.rs#L50)), CUDA, and ROCm:

| Metal operation | Vulkan Compute equivalent |
|---|---|
| `MTLResourceOptions::StorageModePrivate` | `VkBuffer` with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` in device-local memory (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) |
| `StorageModeShared` staging | `VkBuffer` in host-visible memory (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT \| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`) |
| `blit_encoder.copyFromBuffer` | `vkCmdCopyBuffer` inside a command buffer submitted to the transfer queue |
| `command_buffer.waitUntilCompleted()` | `VkFence` signaled at submit + `vkWaitForFences` |
| `expand_buffer` (reallocation) | New `VkBuffer` → `vkCmdCopyBuffer` → destroy old + free memory |

The Vulkan path has **one extra concern**: descriptor sets. Every search has to bind the current `vectors_buffer` + `query_buffer` + `results_buffer` into a descriptor set before launching the compute pipeline. This can be pooled, but the code is heavier than Metal's "set argument, dispatch" flow. Plan for a `DescriptorCache` helper.

## 5.4 Distance computation

| Metric | Recommended path |
|---|---|
| `Euclidean` (L2) | Custom GLSL compute kernel `l2_distance.comp`, workgroup = 256, subgroup reductions |
| `Cosine` | Custom GLSL kernel combining SGEMV + normalization (no rocBLAS/cuBLAS equivalent available from Rust) |
| `DotProduct` | Custom GLSL SGEMV kernel |

**Critical implication:** unlike CUDA and ROCm, we cannot lean on a vendor BLAS library. All distance kernels are hand-written. This adds roughly 3–5 dev-days to the plan versus the ROCm estimate.

## 5.5 Runtime detection

```rust
// src/backends/detector.rs
fn is_intel_available() -> bool {
    // 1. Enumerate Vulkan physical devices via ash
    // 2. Filter by VkPhysicalDeviceProperties::vendorID == 0x8086 (Intel)
    //    OR fall back to any device if "universal Vulkan" mode is enabled
    // 3. Cache the result
}
```

The `vendorID == 0x8086` check selects Intel specifically. If the project later broadens to "universal Vulkan", drop the vendor filter.

## 5.6 Error-type design

Extend [error.rs](../../../src/error.rs) with:

```rust
#[error("Vulkan error: {0}")]
VulkanError(String),

#[error("Intel GPU error: {0}")]
IntelError(String),

#[error("SPIR-V compilation failed: {0}")]
SpirvCompileError(String),
```

Implement `From<ash::vk::Result>` for `VulkanError`.

## 5.7 Backend selection

With four backends present, the detector priority needs an explicit rule:

**Default priority:** `Metal > CUDA > ROCm > Intel > CPU`.

**Override via env var:** `HIVE_GPU_BACKEND=cuda|rocm|intel|metal|cpu`.

**Special case for "universal Vulkan" mode:** a second env var `HIVE_GPU_VULKAN_UNIVERSAL=1` makes the Intel backend accept any Vulkan-capable device, not just Intel ones. Useful for Docker containers without CUDA/ROCm.

Document all of this in a new `docs/guides/BACKEND_SELECTION.md` once implemented.

## 5.8 Naming

**Feature flag name:** `intel` (not `xe`, not `oneapi`, not `vulkan`).

Reasoning: the target audience talks about "Intel GPUs". Xe is internal architecture branding that may be retired. OneAPI is the software brand but we are not using DPC++. Vulkan is correct but misleading — users would expect it to be the primary backend on all platforms, which it is not.

**Module path:** `hive_gpu::intel::IntelContext`, `hive_gpu::intel::IntelVectorStorage`.

**Backend string in `GpuDeviceInfo::backend`:** `"Intel"`.
