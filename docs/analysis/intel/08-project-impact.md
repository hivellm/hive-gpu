# 08 — Impact on the Rest of the Project

## 8.1 Files that change

- [src/lib.rs](../../../src/lib.rs): add `#[cfg(feature = "intel")] pub mod intel;`. The existing `#![allow(warnings)]` at [lib.rs:6](../../../src/lib.rs#L6) **must** be removed before merging a fourth backend — otherwise `cargo check` cannot police the new `unsafe` Vulkan surface.
- [src/traits.rs](../../../src/traits.rs): **no changes** — backend-agnostic.
- [src/types.rs](../../../src/types.rs): `backend: String = "Intel"` — introducing a new canonical value. Consider whether existing tests pattern-match on known strings.
- [src/backends/detector.rs](../../../src/backends/detector.rs): add `Intel` variant + `is_intel_available()` + prioritization rules.
- [Cargo.toml](../../../Cargo.toml): new `intel` feature + `[target.'cfg(any(target_os = "linux", target_os = "windows"))'.dependencies]` entries for `ash` (and `shaderc` as a build dep).
- [tests/gpu_detection_tests.rs](../../../tests/gpu_detection_tests.rs): add an Intel path with graceful skip.

## 8.2 New files

```
src/intel/
├── mod.rs
├── context.rs            # Vulkan instance + device + queue
├── vector_storage.rs     # VkBuffer + staging + descriptor cache
├── buffer_pool.rs        # Optional: memory sub-allocator (vk-mem wrapper)
├── vram_monitor.rs
├── kernels.rs            # Pipeline creation + dispatch
└── shaders/
    ├── l2_distance.comp
    ├── cosine_similarity.comp
    └── dot_product.comp

tests/
├── intel_device_info.rs
├── intel_vector_ops.rs
├── intel_stress.rs
└── cross_backend_consistency.rs  # updated to include Intel

examples/
└── intel_basic.rs

docs/guides/
└── INTEL_SETUP.md
```

## 8.3 Dependency-graph additions

Adding `ash` pulls:

- `ash` itself (well-maintained, small)
- `libloading` (already pulled if CUDA/cudarc is in; otherwise new)

Adding `shaderc`:

- `shaderc` crate + `shaderc-sys`
- At build time: the system `glslc` binary OR the Vulkan SDK (depending on how `shaderc-sys` is configured)

Alternative if the project prefers fewer build-time dependencies:

- Ship **pre-compiled SPIR-V blobs** in the crate under `src/intel/shaders/*.spv`.
- Make `shaderc` a `[dev-dependencies]` only, used to regenerate the SPIR-V when kernels change.
- Document the regeneration command in `CONTRIBUTING.md`.

This keeps end-user builds lean and matches how many Vulkan crates ship today.

## 8.4 Build-time cost

`shaderc` + `ash` together add ~15–30 seconds to a clean `cargo build --features intel` depending on the host. Incremental builds are unaffected once the SPIR-V is cached. Acceptable.

Comparison: the CUDA `build.rs` with `nvcc` is expected to add 30–60 seconds; the ROCm `build.rs` with `hipcc` 30–60 seconds. Intel's cost is the lowest of the three.

## 8.5 Backend-selection impact

With four backends, the priority logic in [src/backends/detector.rs:55-68](../../../src/backends/detector.rs#L55) becomes:

```text
Metal > CUDA > ROCm > Intel > CPU
```

This is a **fallback chain**, not an exclusivity list. On a Linux workstation with both an NVIDIA card and an Arc Pro, CUDA wins by default. Overrides via env var let the user pick.

**Edge case to document:** on Linux hosts with an Intel iGPU + an Arc discrete GPU, the Intel detector must pick the discrete one. Ordering physical devices by `VkPhysicalDeviceType::DiscreteGpu` first, then `IntegratedGpu`, handles this.

## 8.6 Documentation surface

New or updated documents:

- [README.md](../../../README.md) — new "Intel backend" section, updated feature-flag table, updated performance table, backend matrix at the bottom.
- [docs/API_REFERENCE.md](../../../docs/API_REFERENCE.md) — mirror the Metal/CUDA/ROCm subsections.
- [docs/DEVELOPMENT.md](../../../docs/DEVELOPMENT.md) — "Building with Intel / Vulkan" section: required driver, Vulkan SDK, how to run the suite on a local Arc GPU.
- [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) — Intel column in benchmark tables.
- `docs/guides/INTEL_SETUP.md` (new) — Ubuntu 24.04 / Windows 11 install, `vulkaninfo` verification, `i915` vs `xe` kernel module, common driver bugs.
- `docs/guides/BACKEND_SELECTION.md` (new) — explains the priority chain and env-var overrides now that four backends exist.

## 8.7 Release sequencing

Assuming CUDA is shipped in `0.2.0` and ROCm in `0.3.0`, Intel fits naturally as:

- `0.3.1` → Phases 1 + 2 + OpenSpec proposal (infrastructure + context, experimental flag).
- `0.3.2` → Phases 3 + 4 (storage + kernels, marked beta).
- `0.4.0` → Phases 5 + 6 (consistency + CI), Intel declared production-ready.

All feature-gated and additive. No breaking changes for Metal / CUDA / ROCm users.

## 8.8 Four-backend coverage matrix (post-v0.4.0)

| OS | Metal | CUDA | ROCm | Intel | CPU |
|---|---|---|---|---|---|
| macOS (Apple Silicon) | ✅ Primary | ❌ | ❌ | ❌ | ✅ |
| Linux x86_64 + NVIDIA | ❌ | ✅ Primary | ❌ | 🟡 via Vulkan | ✅ |
| Linux x86_64 + AMD | ❌ | ❌ | ✅ Primary | 🟡 via Vulkan | ✅ |
| Linux x86_64 + Intel | ❌ | ❌ | ❌ | ✅ Primary | ✅ |
| Linux x86_64, no GPU | ❌ | ❌ | ❌ | ❌ | ✅ |
| Windows x86_64 + NVIDIA | ❌ | ✅ Primary | ❌ | 🟡 via Vulkan | ✅ |
| Windows x86_64 + AMD | ❌ | ❌ | 🟡 Experimental | 🟡 via Vulkan | ✅ |
| Windows x86_64 + Intel | ❌ | ❌ | ❌ | ✅ Primary | ✅ |

The 🟡 cells are the "universal Vulkan" bonus that comes for free with the Intel backend.

**Total market coverage after v0.4.0:** Metal (~5%) + CUDA (~70%) + ROCm (~15%) + Intel (~1%) + universal Vulkan fallback ≈ **92–93%**.

The Intel backend moves coverage from 90% to ~92% and adds a universal fallback. That ~2% is the honest answer to "what does this backend buy us?".
