# 02 — Current State of the Code

## 2.1 What already exists

| Artifact | Where | State |
|---|---|---|
| High-level proposal | [openspec/changes/add-rocm-backend/proposal.md](../../../openspec/changes/add-rocm-backend/proposal.md) | ✅ Complete |
| Task list (13 sections) | [openspec/changes/add-rocm-backend/tasks.md](../../../openspec/changes/add-rocm-backend/tasks.md) | ✅ Complete |
| Spec with requirements / scenarios | [openspec/changes/add-rocm-backend/specs/rocm-backend/spec.md](../../../openspec/changes/add-rocm-backend/specs/rocm-backend/spec.md) | ✅ Complete |
| Mentions in public types | [src/types.rs:110, 122, 128](../../../src/types.rs#L110) | `backend: "ROCm"`, `compute_capability: "gfx1030"` documented but never produced |
| Mentions in tests | [tests/device_info_tests.rs](../../../tests/device_info_tests.rs), [tests/gpu_detection_tests.rs](../../../tests/gpu_detection_tests.rs) | Only strings in assertions; no AMD code path |

## 2.2 What does NOT exist

- ❌ `src/rocm/` — no folder, no module.
- ❌ `rocm` feature flag in [Cargo.toml](../../../Cargo.toml).
- ❌ Dependencies `hip-runtime-sys` / `rocblas-sys` / `hip-sys` / any AMD crate.
- ❌ `GpuBackendType::Rocm` in [src/backends/detector.rs:10](../../../src/backends/detector.rs#L10) — the enum only holds `Metal`, `Cuda`, `Cpu`.
- ❌ `HiveGpuError::RocmError` in [src/error.rs](../../../src/error.rs).
- ❌ `.hip` / `.cpp` kernels for HIP.
- ❌ `examples/rocm_basic.rs`.
- ❌ `tests/rocm_*.rs`.
- ❌ CI workflow for ROCm.
- ❌ `build.rs` (needed to compile HIP kernels with `hipcc`).

## 2.3 Cross-references in current code

- [src/types.rs:110](../../../src/types.rs#L110) already documents `backend: String` with `"ROCm"` as a valid value.
- [src/types.rs:128](../../../src/types.rs#L128) documents `compute_capability: "gfx1030"` — the contract with the backend when it exists.
- [src/backends/detector.rs:55-68](../../../src/backends/detector.rs#L55) has the priority order `Metal > CUDA > CPU` — **ROCm must be inserted between CUDA and CPU** (or ranked equal to CUDA on Linux).

## 2.4 What the Metal and (future) CUDA backends provide as reference

Once CUDA reaches Phase 3 of its plan, the ROCm implementer has **two working reference backends** to cross-check against. This is the main reason to sequence ROCm after CUDA rather than in parallel.

The contracts the ROCm backend must satisfy are defined in [src/traits.rs](../../../src/traits.rs):

- `GpuBackend` → describe the device.
- `GpuContext` → construct storages.
- `GpuVectorStorage` → add/search/remove/clear.
- `GpuBufferManager` (optional, currently only Metal) → pooling.
- `GpuMonitor` (optional) → VRAM accounting.

None of these traits need changes to accept a ROCm implementation.

## 2.5 Noise in public API already committing to ROCm

The following lines already expose ROCm in the public surface, making the implementation a binding promise:

- [src/types.rs:110](../../../src/types.rs#L110): doc example lists `"ROCm"` as a valid `backend` string.
- [src/types.rs:122](../../../src/types.rs#L122): `driver_version` doc example mentions `"ROCm 5.4"`.
- [src/types.rs:128](../../../src/types.rs#L128): `compute_capability` doc explicitly names `"gfx1030"`.

Any caller that pattern-matches on these strings would be permitted to expect ROCm support, even though it is absent. The remedy is to ship the backend — not to remove the documentation.
