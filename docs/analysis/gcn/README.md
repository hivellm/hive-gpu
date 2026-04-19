# GCN / ROCm Backend Analysis — Index

**Date:** 2026-04-19
**Analyzed version:** `hive-gpu 0.1.10`
**Scope:** Current state of AMD GPU support (GCN / RDNA / CDNA via ROCm / HIP), gaps, and implementation plan.

## Terminology note

"GCN" (Graphics Core Next) was the AMD architecture through gfx9 (Vega). From gfx10 onward AMD uses **RDNA** (consumer) and **CDNA** (data center). In this document we use **GCN as a shorthand for "AMD backend via ROCm / HIP"**, covering gfx900 and onward (Vega, RDNA, RDNA2, RDNA3, CDNA, CDNA2, CDNA3). The feature flag to create should be named `rocm`, not `gcn`, to stay aligned with the upstream ecosystem.

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

- **Maturity:** Non-existent (0% implemented, 100% specified via OpenSpec).
- **Feature flag `rocm`:** does **not** exist in [Cargo.toml](../../../Cargo.toml); no `src/rocm/`, no `GpuBackendType::Rocm`.
- **Prerequisite:** the [CUDA backend](../cuda/README.md) should reach Phase 3 before ROCm starts, so ROCm can mirror the stabilized pattern.
- **Estimated effort** for functional parity with Metal/CUDA: **16–21 dev-days** after CUDA is working.
- **Recommended binding strategy:** `bindgen` against `$ROCM_PATH/include/hip/` + `rocblas-sys` for BLAS.
- **Kernel strategy:** `.hip` compiled by `hipcc` with `--offload-arch` for gfx900, gfx906, gfx908, gfx90a, gfx1030, gfx1100.
