# 06 — Implementation Plan

Phased plan assuming **CUDA and ROCm are already merged** (establishes the `build.rs` pattern, the error-type conventions, and the cross-backend consistency harness).

## Phase 0 — Prerequisites (out of scope here)

- Functional CUDA backend (see [../cuda/](../cuda/)).
- Functional ROCm backend (see [../gcn/](../gcn/)).
- Generic `build.rs` in place.
- `#![allow(warnings)]` removed from [src/lib.rs:6](../../../src/lib.rs#L6).
- An OpenSpec proposal `add-intel-backend` drafted and accepted.

## Phase 1 — Infrastructure (2 days)

1. Add the `intel` feature to [Cargo.toml](../../../Cargo.toml) with optional deps `ash` and `shaderc` (or `naga` if WGSL is kept).
2. Add `GpuBackendType::Intel` to [src/backends/detector.rs:10](../../../src/backends/detector.rs#L10). Update `detect_available_backends` / `select_best_backend` (priority after ROCm, before CPU).
3. Add `HiveGpuError::VulkanError`, `IntelError`, `SpirvCompileError` to [error.rs](../../../src/error.rs).
4. Create empty scaffolding `src/intel/{mod.rs,context.rs,vector_storage.rs,buffer_pool.rs,vram_monitor.rs,kernels.rs}` mirroring [src/cuda/](../../../src/cuda/).
5. Wire `#[cfg(feature = "intel")] pub mod intel;` into [src/lib.rs](../../../src/lib.rs).
6. Extend `build.rs` to compile `src/intel/shaders/*.comp` → SPIR-V and embed via `OUT_DIR` + `include_bytes!`.

**Exit criterion:** `cargo check --features intel` passes on Linux and Windows.

## Phase 2 — Context + device info (3–4 days)

1. Implement `IntelContext` with:
   - Vulkan instance creation with validation layers in debug builds.
   - Physical-device selection filtered by `vendorID == 0x8086` (or any Vulkan device if `HIVE_GPU_VULKAN_UNIVERSAL=1`).
   - Logical device with a dedicated compute queue.
   - Command pool + descriptor pool for workload submission.
2. Populate `GpuDeviceInfo` from `VkPhysicalDeviceProperties`, `VkPhysicalDeviceMemoryProperties`, and Vulkan extension `VK_KHR_driver_properties` for a usable `driver_version` string.
3. Implement `is_available()` via lazy Vulkan loader (ash loads the library on demand).
4. Test: `tests/intel_device_info.rs` with graceful skip when no Intel GPU is present.

**Exit criterion:** device info returned matches the output of `vulkaninfo --summary` on an Arc card.

## Phase 3 — Vector Storage (4–5 days)

1. `IntelVectorStorage` with device-local `VkBuffer` + a reusable host-visible staging buffer.
2. `add_vectors` → write to staging, `vkCmdCopyBuffer` to device buffer, `vkQueueSubmit` with fence.
3. `ensure_capacity` with D2D `vkCmdCopyBuffer` + `vkDestroyBuffer` + `vkFreeMemory` for the old buffer.
4. `remove_vectors` via a `removed_indices: HashSet<usize>` mask (same pattern as Metal / CUDA / ROCm).
5. A small descriptor-set cache to avoid re-allocating on every search.
6. Parity tests with Metal / CUDA / ROCm.

**Exit criterion:** storing 10k × 128-dim vectors succeeds on Arc B580 and Arc Pro B70; grow/shrink cycles stable over 5-minute stress.

## Phase 4 — Compute kernels (5–6 days)

1. Decision gate: **GLSL + shaderc** or **WGSL + naga**. Pick one; do not ship both.
2. Write the three distance kernels:
   - `l2_distance.comp` — brute force L2 with per-workgroup reduction.
   - `cosine_similarity.comp` — dot product + norms fused (no BLAS available).
   - `dot_product.comp` — SGEMV by hand.
3. Rust launcher in `src/intel/kernels.rs`: pipeline creation, descriptor binding, dispatch, readback.
4. Top-K initially on CPU (copy scores back + sort). Defer GPU radix sort.

**Exit criterion:** numerical agreement with Metal, CUDA, and ROCm within `1e-4` over 1000 random queries.

## Phase 5 — Cross-backend consistency + benchmarks (2–3 days)

1. Extend the cross-backend consistency test from ROCm Phase 5 to include Intel.
2. Benchmark on Arc B580 (consumer), Arc Pro B70 (workstation), and integrated Iris Xe (smoke only).
3. Document performance honestly: expect 40–60% of CUDA at equivalent silicon price point.

**Exit criterion:** consistency test green; benchmark numbers recorded in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md).

## Phase 6 — Tests, CI, and docs (2–3 days)

1. `tests/intel_integration.rs` — scenarios mirroring [tests/integration_tests.rs](../../../tests/integration_tests.rs).
2. GitHub Actions workflow with a Linux runner + `swiftshader` or `lavapipe` (software Vulkan) to at least exercise the validation layers in CI. Real hardware testing requires a self-hosted runner with an Arc card.
3. New guide `docs/guides/INTEL_SETUP.md` — driver install on Ubuntu 24.04 / Windows 11, how to verify with `vulkaninfo`, common pitfalls (wrong driver version, secure boot, kernel module `xe` vs `i915`).
4. Update [README.md](../../../README.md) backend matrix with Intel column.
5. Update [docs/API_REFERENCE.md](../../../docs/API_REFERENCE.md).

**Exit criterion:** `cargo build --features intel` green on Linux and Windows; docs reviewed by one external reader.

## Total effort

- **Functional parity with Metal/CUDA/ROCm:** 18–23 dev-days once the prerequisites are in place.
- **Plus the OpenSpec proposal, kernel audits, and CI runner setup:** add 4–5 dev-days of overhead.
- **Grand total:** 22–28 dev-days.

## Sequencing constraints

- Do **not** start Phase 1 before CUDA's `build.rs` pattern is frozen.
- Do **not** merge Phase 4 without having tested on at least one discrete Arc GPU (B580 minimum); integrated Iris Xe alone is insufficient.
- Do **not** advertise the backend in the README until Phase 5's consistency test is green — users will assume it matches CUDA/ROCm performance, and that is not true.
