# Intel GPU Backend Analysis — Index

**Date:** 2026-04-19
**Analyzed version:** `hive-gpu 0.1.10`
**Scope:** Viability, architectural options, and implementation plan for an Intel GPU backend (Arc, Arc Pro, Battlemage, Data Center GPU Max) as a fourth backend after Metal / CUDA / ROCm.

## Terminology note

"Intel GPU" in this document covers the modern Xe family: **Xe-LP** (integrated in Tiger Lake → Arrow Lake), **Xe-HPG** (Arc Alchemist — A380/A580/A750/A770), **Xe2 / Battlemage** (Arc B-series consumer and Pro B70/B65), and **Xe-HPC** (Ponte Vecchio — Data Center GPU Max, being sunset in 2026 with Crescent Island as successor).

The feature flag should be named `intel` (not `xe`, not `oneapi`) so it survives multiple Intel generations without churn.

## Documents

1. [Executive Summary](01-executive-summary.md)
2. [Current State of the Code](02-current-state.md)
3. [Hardware Targets and Architectures](03-hardware-targets.md)
4. [Gap Analysis vs. Requirements](04-gap-analysis.md)
5. [Architectural Decisions](05-architecture-decisions.md)
6. [Implementation Plan](06-implementation-plan.md)
7. [Risks and Mitigations](07-risks-and-mitigations.md)
8. [Impact on the Rest of the Project](08-project-impact.md)
9. [Feature Parity Matrix](09-feature-parity.md)
10. [Next Immediate Steps](10-next-steps.md)

## At a Glance

- **Maturity in the repo:** Non-existent. No feature flag, no module, no kernel, no mention in [src/types.rs](../../../src/types.rs) — unlike ROCm, Intel is not even promised in the public API.
- **Realistic market share (2026):** <1% of ML/AI / vector-search deployments. This is a **differentiator play**, not a mass-coverage play.
- **Recommended stack:** **Vulkan Compute via `ash` + kernels via `rust-gpu` or GLSL + `shaderc`**. Level Zero has no mature Rust binding; DPC++/SYCL is C++-only; oneMKL's GPU path is not reachable from Rust without a C++ shim.
- **Estimated effort** for functional parity with Metal/CUDA: **20–28 dev-days** after CUDA is working. Higher than ROCm because the Rust ecosystem for Intel is immature and kernel authoring goes through Vulkan/SPIR-V instead of a HIP-like layer.
- **Priority recommendation:** **defer until after CUDA + ROCm ship.** The honest ROI is the lowest of the four backends. Revisit when Crescent Island ships (Q3 2026) or when a concrete customer with Arc Pro B70 fleet appears.

## TL;DR: should the project do this?

Yes, **but as a late v0.3.x investment**, not an urgent workstream. The rationale, tradeoffs, and escape hatches are in [01-executive-summary.md](01-executive-summary.md).
