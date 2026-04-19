# 10 — Next Immediate Steps

## Decision point first

Before any code is written, the project owner needs to answer three questions:

1. **Do we have — or plan to acquire — at least one Intel Arc discrete GPU for dev + CI?** Without hardware access, the backend is unimplementable. Minimum: Arc B580 or A770 on a developer workstation. Ideal: Arc Pro B70 in a self-hosted CI runner.
2. **Are we targeting Arc Pro (workstation) customers specifically, or is this a differentiator ticked off the roadmap?** The answer changes the tiering in [03-hardware-targets.md](03-hardware-targets.md) and the benchmark priorities in [06-implementation-plan.md](06-implementation-plan.md).
3. **Who maintains this backend after v1 ships?** A Vulkan backend running on four vendors (Intel + NVIDIA/AMD/software Vulkan) has a wide surface area. If no one owns it, it will rot within two minor releases.

If any of these three answers is "no" or "we do not know", the realistic action is **to shelve this workstream and revisit in v0.4+**.

## If "yes" across the board

1. **Audit the existing WGSL shaders** in [src/shaders/](../../../src/shaders/). Determine whether they are production-quality and can be compiled to SPIR-V via `naga`, or whether new GLSL kernels are the cleaner path. This 4-hour audit saves ~3 dev-days downstream.
2. **Draft an OpenSpec proposal** `add-intel-backend` mirroring the shape of [openspec/changes/add-rocm-backend/proposal.md](../../../openspec/changes/add-rocm-backend/proposal.md). Get it reviewed before any implementation.
3. **Validate `ash` + `shaderc` build on all three target OSes** (Linux Ubuntu 24.04, Windows 11, macOS — last one only to confirm the feature gate excludes it cleanly). 1-day spike.
4. **Provision hardware** — buy or borrow an Arc B580 + Arc Pro B70. Set up one as a self-hosted runner or reserve budget for cloud instances (few cloud providers offer Intel Arc yet — this may force on-prem).
5. Open branch `feat/intel-backend` and start Phase 1.

## Handoff checklist for the next engineer

- [ ] Read [01-executive-summary.md](01-executive-summary.md) to understand the ROI context and why this is deferred work.
- [ ] Read [03-hardware-targets.md](03-hardware-targets.md) to understand which Intel SKUs matter.
- [ ] Read [05-architecture-decisions.md](05-architecture-decisions.md) — the Vulkan + GLSL + shaderc vs. Naga+WGSL choice is the highest-leverage call.
- [ ] Read [../cuda/](../cuda/) and [../gcn/](../gcn/) in full — Intel inherits the scaffolding patterns they establish.
- [ ] Confirm `vulkaninfo` lists the Intel GPU on the dev host before starting Phase 2.
- [ ] Have a reviewer with prior Vulkan experience pre-agreed for Phase 3 (buffer lifetimes + descriptor-set management is the main failure mode).

## Open decisions to resolve before coding

- **Kernel pipeline:** GLSL + `shaderc` vs. WGSL + `naga` vs. `rust-gpu`. Pick one.
- **Pre-compiled vs. build-time SPIR-V:** ship blobs or recompile on each build? Affects the `build.rs` design.
- **Feature-flag name:** confirm `intel` (recommended) vs. `vulkan` (broader scope). Renames after v0.3.1 will break users.
- **Universal Vulkan mode:** yes or no? Affects scope by roughly 2 dev-days but expands test surface significantly.
- **Minimum Vulkan version:** 1.2 (widely available) or 1.3 (better subgroup features)? Locks out Arc Alchemist on older drivers if 1.3.

## Escalation triggers

Pause and re-evaluate this plan if any of the following is true at the end of Phase 2:

- `ash`'s physical-device enumeration is unreliable across Linux + Windows with current Intel drivers.
- The `shaderc` or `naga` build pipeline adds more than 60 seconds to clean builds or requires a non-trivial installation step on contributor machines.
- Arc B580 benchmark numbers (when Phase 5 arrives) are below 30% of CUDA on equivalent silicon — suggests a fundamental Vulkan-driver limitation we cannot break.
- The cross-backend consistency test exceeds the `1e-3` divergence envelope even after careful FMA flag tuning.

In any of those cases, revisit [05-architecture-decisions.md](05-architecture-decisions.md) before continuing.

## Once done

When the Intel backend reaches "production-ready" per [07-risks-and-mitigations.md](07-risks-and-mitigations.md):

- Announce in [CHANGELOG.md](../../../CHANGELOG.md) with the supported Arc SKU list, minimum Vulkan version, minimum driver version.
- Update [README.md](../../../README.md) backend matrix and coverage claims — honestly, not optimistically.
- Merge the OpenSpec change at `openspec/changes/add-intel-backend/` into the main specs tree.
- Record benchmark numbers in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) for at least Arc B580 + Arc Pro B70, comparing honestly against CUDA/ROCm at similar price points.
- Consider whether the "universal Vulkan" mode deserves its own top-level feature flag `vulkan` in a later minor version.
