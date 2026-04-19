# 02 — Current State of the Code

## 2.1 What exists

Nothing. The Intel backend has no representation anywhere in the repository:

| Check | Result |
|---|---|
| `grep -ri intel` in [src/](../../../src/) | 0 matches |
| `grep -ri "oneapi\|level.zero\|xe\|arc"` in [src/](../../../src/) | 0 matches (after filtering the unrelated `arch` substring) |
| `Intel` in [src/types.rs](../../../src/types.rs) doc strings | Absent |
| OpenSpec proposal for Intel | None ([openspec/changes/](../../../openspec/changes/) contains only `add-cuda-backend`, `add-rocm-backend`, etc.) |
| Feature flag `intel` / `xe` / `oneapi` | Not in [Cargo.toml](../../../Cargo.toml) |
| `GpuBackendType::Intel` | Not in [src/backends/detector.rs](../../../src/backends/detector.rs) |
| Intel-specific error variant | Not in [src/error.rs](../../../src/error.rs) |
| `.hlsl` / `.glsl` / `.spv` compute kernels | None |
| Vulkan-related dependency (`ash`, `vulkano`, `wgpu`) | None |

## 2.2 What exists that *could* be repurposed

The [src/shaders/](../../../src/shaders/) directory contains WGSL shaders originally authored for a (cancelled?) WebGPU backend:

- [batch_construction.wgsl](../../../src/shaders/batch_construction.wgsl)
- [distance.wgsl](../../../src/shaders/distance.wgsl)
- [dot_product.wgsl](../../../src/shaders/dot_product.wgsl)
- [hnsw_construction.wgsl](../../../src/shaders/hnsw_construction.wgsl)
- [hnsw_navigation.wgsl](../../../src/shaders/hnsw_navigation.wgsl)
- [similarity.wgsl](../../../src/shaders/similarity.wgsl)

**Why this is relevant:** WGSL, GLSL, and HLSL all compile to SPIR-V. The Vulkan Compute path recommended here could in principle reuse the algorithmic intent of these WGSL kernels by translating them to GLSL (or via `wgpu`'s Naga to compile WGSL → SPIR-V directly). This is **not free** — WGSL has subtly different memory and barrier semantics — but it means the project is not starting from zero on the algorithmic side.

Current user decision: **no web/WebGPU**, so `wgpu` is out as a runtime dependency. But **Naga the library** (which `wgpu` uses) can still be used at build time to compile WGSL → SPIR-V without pulling the full wgpu runtime.

## 2.3 What the public API already commits to

Unlike ROCm, the `GpuDeviceInfo` docstrings in [src/types.rs](../../../src/types.rs) do not mention Intel, Xe, or oneAPI. This means:

- **No broken promise** if Intel support is never delivered.
- **Full freedom** to design the backend name, error taxonomy, and feature flag without breaking existing doc examples.
- The string `"Intel"` (or `"IntelXe"`, `"OneAPI"` — a naming decision in [05-architecture-decisions.md](05-architecture-decisions.md)) can be freshly introduced.

## 2.4 Platform gating implications

Intel GPUs are present on **Linux and Windows**, never on macOS (Apple hardware does not ship Intel discrete GPUs anymore, and Apple Intel-integrated graphics work through Metal). The `#[cfg(...)]` gating should mirror ROCm:

```rust
#[cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))]
pub mod intel;
```

No `cfg(target_os = "macos")` code path — Apple-silicon Macs use the Metal backend, and Intel Macs are a shrinking legacy base that the project explicitly does not target via an Intel backend.
