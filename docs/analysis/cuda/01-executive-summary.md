# 01 — Executive Summary

The CUDA backend exists only as a **skeleton of stubs** — no real call to the CUDA runtime/driver, no kernels, no real VRAM management. The `cuda` feature flag in [Cargo.toml](../../../Cargo.toml) is **empty** (it does not pull `cudarc` or any sys-crate), so enabling it today merely recompiles the `src/cuda/` module without linking anything.

Compared to the Metal backend ([src/metal/vector_storage.rs](../../../src/metal/vector_storage.rs)), which already performs real VRAM uploads via blit encoders, owns an expandable buffer pool, and offers functional storage, the CUDA module is approximately **5% complete**.

**Maturity rating:** 🔴 **Stub** (compiles nothing GPU-related, returns `"not implemented yet"` on nearly every path).

## Key findings

| Finding | Location | Severity |
|---|---|---|
| Feature `cuda = []` has no dependencies | [Cargo.toml:39](../../../Cargo.toml#L39) | 🔴 Blocking |
| `compute_capability` hard-coded to `(7, 5)` | [src/cuda/context.rs:29](../../../src/cuda/context.rs#L29) | 🔴 Blocking |
| `total_memory` hard-coded to 1 GB | [src/cuda/context.rs:30](../../../src/cuda/context.rs#L30) | 🔴 Blocking |
| `is_available()` always returns `false` | [src/cuda/context.rs:72](../../../src/cuda/context.rs#L72) | 🔴 Blocking |
| Detection uses only env vars | [src/backends/detector.rs:84](../../../src/backends/detector.rs#L84) | 🟡 Major |
| `#![allow(warnings)]` globally masks issues | [src/lib.rs:6](../../../src/lib.rs#L6) | 🟡 Major |
| Example panics at runtime | [examples/cuda_basic.rs](../../../examples/cuda_basic.rs) | 🟡 Major |
| No CUDA tests exist | [tests/](../../../tests/) | 🟡 Major |
| No kernels (`.cu`/`.ptx`) in repo | [src/shaders/](../../../src/shaders/) | 🔴 Blocking |
| No `build.rs` for `nvcc` | (repo root) | 🔴 Blocking |

## Why this matters

The [README](../../../README.md) advertises CUDA support and a `cuda` feature flag, but any user enabling it on Linux/Windows will get a compiling crate that **falls back to error paths** on every real operation. The project is effectively Metal-only in v0.1.10 regardless of the feature flag.

Closing this gap unlocks ~70% of the GPU market according to the existing [proposal](../../../openspec/changes/add-cuda-backend/proposal.md) and is the single highest-ROI workstream on the roadmap.
