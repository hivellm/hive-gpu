# 01 — Executive Summary

**There is no Intel code in the project and no mention of Intel in the public API.** Unlike ROCm (which is name-checked in [src/types.rs:110,128](../../../src/types.rs#L110)), Intel has zero surface area in v0.1.10 — no `intel`/`xe`/`oneapi` feature flag in [Cargo.toml](../../../Cargo.toml), no module in [src/](../../../src/), no doc example mentioning Xe compute capability strings.

Starting Intel support is therefore a **purely additive** decision with no legacy debt, but the ecosystem cost is meaningfully higher than CUDA or ROCm because the Rust side of Intel's compute stack is immature.

**Maturity rating:** 🔴 **Non-existent** (0% implemented, not even specified in OpenSpec).

## The core decision

Intel compute from a pure-Rust crate in 2026 has no first-class native path. The options are, in descending order of practicality:

1. **Vulkan Compute via `ash` + kernels in `rust-gpu` / GLSL.** Works on Arc and Arc Pro via the official Intel Vulkan driver; also works incidentally on NVIDIA and AMD (single codebase for three vendors). **Recommended.**
2. **Level Zero (L0) via hand-written `bindgen`.** The "native Intel" low-level API, open-source, used by DPC++ underneath. Rust bindings are sparse; maintaining them is non-trivial.
3. **OpenCL 3.0** via `ocl` crate. Works, but already rejected in [../opencl-analysis](../).
4. **DPC++ / SYCL FFI shim.** C++ only; requires maintaining a C++ wrapper crate. Adds a second build system and offers no clear upside over Vulkan Compute.

## Why this matters — or does not

**Reasons to do it:**

- Only realistic path to Intel Arc Pro B70 (32 GB GDDR6, 22.9 TFLOPS, workstation-class, shipping March 2026) and future Crescent Island (Xe3P, 160 GB LPDDR5X, Q3 2026) — the only Intel parts with enough VRAM/bandwidth for serious vector search.
- Side benefit: a Vulkan-Compute backend **also runs on NVIDIA and AMD**, giving a universal fallback that covers edge cases where the native backend (CUDA/ROCm) fails to initialize.
- Differentiator vs. competing vector search libraries — none of them support Intel today.

**Reasons not to:**

- Intel discrete GPU share in ML/AI / vector search in 2026 is **<1%**. No production deployments to point at.
- Ponte Vecchio (the only serious datacenter Intel GPU to date) is **end-of-life Jan 2026**; Crescent Island is not shipping yet.
- Rust ecosystem gap: no maintained `level-zero-sys`, no Rust oneMKL-GPU binding. Bootstrapping costs more than the other three backends.
- Every hour spent here is an hour not spent polishing CUDA/ROCm/Metal, which together cover ~90% of the market.

## Honest ROI ranking

After CUDA (70% coverage) and ROCm (15%), Intel is the smallest pie slice among the serious GPU vendors. Below it are:

- **Apple Neural Engine (ANE)** — tied into CoreML, no direct GPU API.
- **Qualcomm Adreno / ARM Mali** — mobile, out of scope.
- **TPUs** — Google Cloud only, proprietary API.

Intel is the **largest remaining opportunity**, but still a distant fourth.

**Recommendation:** deliver Metal → CUDA → ROCm first. Revisit Intel **in v0.3.x**, either when (a) Crescent Island ships and a customer with that hardware appears, or (b) a team member volunteers Arc hardware and bandwidth to maintain the backend.

If the project owner wants to proceed immediately anyway, the plan in [06-implementation-plan.md](06-implementation-plan.md) is viable — just be clear-eyed that it buys single-digit percent coverage for 20–28 dev-days of work, not 70% like CUDA.
