# 01 — Executive Summary

**There is no AMD code in the project.** Nothing in [src/](../../../src/), no `rocm/`, no `hip/`, no reference to `HIP_PATH`. What exists is a well-formed **OpenSpec proposal** in [openspec/changes/add-rocm-backend/](../../../openspec/changes/add-rocm-backend/) describing the work, but it was never implemented.

Compared to CUDA (which at least has structural stubs), ROCm is in a **greenfield** state. On the positive side, this allows the implementation to follow the already-stabilized shape of the Metal backend and mirror the CUDA backend once CUDA is functional, reusing 80–90% of the algorithmic design.

**Maturity rating:** 🔴 **Non-existent** (0% implemented, 100% specified).

## Key findings

| Finding | Location | Severity |
|---|---|---|
| No feature flag `rocm` in Cargo | [Cargo.toml](../../../Cargo.toml) | 🔴 Blocking |
| No `src/rocm/` module | [src/](../../../src/) | 🔴 Blocking |
| No `GpuBackendType::Rocm` variant | [src/backends/detector.rs:10](../../../src/backends/detector.rs#L10) | 🔴 Blocking |
| No `HiveGpuError::RocmError` variant | [src/error.rs](../../../src/error.rs) | 🔴 Blocking |
| No HIP kernels | repo-wide | 🔴 Blocking |
| No examples / tests for ROCm | [examples/](../../../examples/), [tests/](../../../tests/) | 🔴 Blocking |
| No `build.rs` for `hipcc` | (repo root) | 🔴 Blocking |
| No AMD CI workflow | [.github/workflows/](../../../.github/workflows/) | 🟡 Major |
| Public API already references `"ROCm"` and `"gfx*"` | [src/types.rs:110,128](../../../src/types.rs#L110) | ✅ Positive (contract is ready) |

## Why this matters

The project already advertises ROCm as a target in [src/types.rs:110](../../../src/types.rs#L110) (documentation string) and [src/types.rs:128](../../../src/types.rs#L128) (compute capability `"gfx1030"`). This commits the public API to a future AMD backend — so delivering it is not optional, it is a pending promise.

According to the existing [proposal](../../../openspec/changes/add-rocm-backend/proposal.md), ROCm adds ~15% market coverage (AMD Instinct MI-series in HPC, RX 6000/7000 in prosumer/workstation). Combined with CUDA and Metal, total coverage reaches ~90%.

## Why it should come *after* CUDA, not before

1. **Reusable patterns:** HIP's API is intentionally CUDA-like (`hipMalloc`↔`cudaMalloc`, `hipMemcpyAsync`↔`cudaMemcpyAsync`, `rocblas_sgemv`↔`cublasSgemv`). Writing CUDA first lets the ROCm author copy the scaffolding almost literally.
2. **Tooling maturity:** `cudarc` is a mature Rust crate; the HIP equivalents are not. Having a working CUDA build makes it easier to bisect whether an AMD bug is in the binding layer or the algorithm.
3. **Cross-backend tests:** the consistency harness (`Metal × CUDA × ROCm`) is trivial to extend once two backends already agree.

This ordering is reflected in the [roadmap](../../ROADMAP.md) as Phase 3.1 (CUDA) preceding Phase 3.2 (ROCm).
