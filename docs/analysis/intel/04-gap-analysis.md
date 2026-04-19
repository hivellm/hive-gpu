# 04 — Gap Analysis vs. Requirements

There is no prior OpenSpec proposal for Intel. The comparable artifact exists for [CUDA](../../../openspec/changes/add-cuda-backend/) and [ROCm](../../../openspec/changes/add-rocm-backend/); an `add-intel-backend` proposal would be the first deliverable of this stream.

## 4.1 What would an equivalent task list look like

Modeled on [openspec/changes/add-rocm-backend/tasks.md](../../../openspec/changes/add-rocm-backend/tasks.md), an Intel task list has 11 sections. All are ❌ today.

| # | Section | Status | Note |
|---|---|---|---|
| 1 | Project setup + `intel` feature in `Cargo.toml` | ❌ | Needs `ash`, `shaderc` (or `naga`), optional `rust-gpu-builder` |
| 2 | Error handling (`IntelError`, `VulkanError`) in `HiveGpuError` | ❌ | Absent from [error.rs](../../../src/error.rs) |
| 3 | `IntelContext` (Vulkan instance + device + queue) | ❌ | Module does not exist |
| 4 | `IntelVectorStorage` (Vulkan buffers) | ❌ | Module does not exist |
| 5 | Compute kernels (SPIR-V) | ❌ | No `.glsl`/`.spv`/`.rs` kernels; WGSL in [src/shaders/](../../../src/shaders/) is not wired into any backend |
| 6 | Module organization + `pub mod intel` | ❌ | Not in [lib.rs](../../../src/lib.rs) |
| 7 | Tests `tests/intel_*.rs` | ❌ | Absent |
| 8 | Examples (`examples/intel_basic.rs`) + docs | ❌ | Absent |
| 9 | Benchmarks | ❌ | [benches/](../../../benches/) Metal-only |
| 10 | CI job with Intel Vulkan driver | ❌ | No workflow |
| 11 | Final validation + CHANGELOG | ❌ | Pending everything above |

## 4.2 Deltas vs. the CUDA/ROCm plan

Compared to the ROCm plan in [../gcn/](../gcn/), the Intel plan adds work in these areas:

| Concern | CUDA / ROCm | Intel (via Vulkan) |
|---|---|---|
| Binding layer | `cudarc` (mature) / `bindgen` on HIP headers | **`ash`** (mature, actively maintained) |
| Kernel language | CUDA C++ / HIP C++ | **GLSL compute or Rust-gpu** |
| Kernel compiler | `nvcc` / `hipcc` | **`shaderc` (GLSL→SPIR-V) or `rust-gpu`/`naga` at build time** |
| BLAS | cuBLAS / rocBLAS | **None usable from Rust** — must write SGEMV kernel by hand |
| VRAM query | `cuDeviceTotalMem` / `hipMemGetInfo` | `vkGetPhysicalDeviceMemoryProperties` (well supported) |
| Device info strings | `"sm_89"` / `"gfx1030"` | `VkPhysicalDeviceProperties::deviceName` + `driverID` |

**Implication:** the hardest part of Intel support is not the FFI layer (Vulkan via `ash` is well-trodden) but the **BLAS gap**. Cosine and DotProduct on CUDA/ROCm are one-line SGEMV calls; on Vulkan Compute they require a hand-written matrix-vector kernel. Good news: the `batch_construction.wgsl` and `similarity.wgsl` files already have algorithmic building blocks.

## 4.3 Prerequisites already met

- Traits in [src/traits.rs](../../../src/traits.rs) are backend-agnostic. No trait change required.
- `GpuDeviceInfo` in [src/types.rs](../../../src/types.rs) has optional `compute_capability` and `pci_bus_id` — both fit naturally.
- Error trait uses `thiserror` — adding `IntelError` variants is additive.

## 4.4 Prerequisites not met

- **No generic `build.rs` exists yet** in the repo root. CUDA and ROCm plans require creating one. Intel will add a third consumer (`shaderc` invocation or `rust-gpu-builder`). Design the `build.rs` with all three in mind when CUDA lands.
- **No shader-language infrastructure.** The WGSL files are currently compile-time untested — nothing consumes them. Shipping Intel means introducing the first real shader pipeline in the project (SPIR-V compile, validate, embed).
- **No Vulkan dependency anywhere.** Adding `ash` is a meaningful dependency-graph addition (see [08-project-impact.md](08-project-impact.md)).

## 4.5 Scope boundary: Intel backend vs. "universal Vulkan backend"

A tempting re-scoping: call this the **Vulkan Compute backend**, not the **Intel backend**, since the same code runs on NVIDIA and AMD Vulkan drivers.

**Arguments for the Intel-scoped framing (what this analysis assumes):**

- Priority is Intel discrete hardware; NVIDIA/AMD will use their native backend anyway for performance.
- Keeps the feature name short (`intel`) and the detection logic clean.
- Positions the backend honestly — users do not expect Vulkan to beat CUDA on NVIDIA.

**Arguments for the universal-Vulkan framing:**

- Single backend covers every major vendor (Intel primary, NVIDIA/AMD as fallback when native init fails).
- Useful for Docker environments where CUDA/ROCm toolchains are missing but Vulkan drivers are present.
- The performance hit (20–40%) is acceptable when the alternative is CPU.

**Recommendation:** ship as `intel` feature initially, document that it *also* works on NVIDIA/AMD as a fallback. If later a fleet operator asks for it, rename to `vulkan` without breaking users who enabled `intel` (re-export via a feature alias).
