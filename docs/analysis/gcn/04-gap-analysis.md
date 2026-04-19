# 04 — Gap Analysis vs. Requirements

The [OpenSpec proposal](../../../openspec/changes/add-rocm-backend/proposal.md) and [tasks.md](../../../openspec/changes/add-rocm-backend/tasks.md) define 13 sections of work. Mapping them to the current codebase:

| # | Section | Status | Notes |
|---|---|---|---|
| 1 | Project setup and dependencies | ❌ | No `rocm` feature, no AMD crate references |
| 2 | Error handling (`RocmError`, `RocblasError`, `HipError`) | ❌ | Not present in [error.rs](../../../src/error.rs) |
| 3 | `RocmContext` | ❌ | No module exists |
| 4 | `RocmVectorStorage` | ❌ | No module exists |
| 5 | HIP kernels | ❌ | No `.hip` files |
| 6 | Module organization / `pub mod rocm` | ❌ | Not wired in [lib.rs](../../../src/lib.rs) |
| 7 | Tests | ❌ | `tests/rocm_*.rs` absent |
| 8 | Examples and documentation | ❌ | `examples/rocm_basic.rs` absent |
| 9 | Benchmarking | ❌ | [benches/](../../../benches/) has no ROCm path |
| 10 | AMD-specific optimizations (wavefront, LDS, coalescing) | ❌ | N/A until kernels exist |
| 11 | Quality checks (`cargo fmt`, clippy, docs) | ❌ | Cannot run without code |
| 12 | CI / CD integration | ❌ | No AMD workflow |
| 13 | Final validation (OpenSpec, manual testing, changelog) | ❌ | Pending everything above |

## Prerequisites already met

- [Device Info API](../../ROADMAP.md) — shipped in v0.1.7, exposed via [src/types.rs](../../../src/types.rs). ROCm can plug into `GpuDeviceInfo` without extending the struct.
- Error trait ergonomics — [error.rs](../../../src/error.rs) uses `thiserror`, so adding ROCm variants is additive.
- Public API contract — `GpuVectorStorage` / `GpuContext` are stable.
- Benchmark harness — [benches/gpu_operations.rs](../../../benches/gpu_operations.rs) is `required-features = ["metal-native"]`; extend with a mirror using `required-features = ["rocm"]`.

## What is *not* obvious from tasks.md

Reading the code, these additional gaps are worth flagging before implementation begins:

1. **No abstraction over PCIe enumeration.** Metal sets `pci_bus_id: None`. CUDA will use `cuDeviceGetPCIBusId`. ROCm via `hipDeviceGetPCIBusId` returns a string of the form `"XXXX:XX:XX.0"` — align the format across all backends or document the divergence.
2. **Payload storage is per-backend today** ([metal/vector_storage.rs:44](../../../src/metal/vector_storage.rs#L44)). ROCm will also need its own `HashMap<String, ...>`. Consider extracting the payload map into a shared helper once two backends exist.
3. **`#![allow(warnings)]`** at [src/lib.rs:6](../../../src/lib.rs#L6) must be removed before a new backend is merged — otherwise clippy cannot police ROCm's `unsafe` usage.
4. **Tests assume `target_os = "macos"` for anything real.** Suites like [tests/gpu_vector_ops_tests.rs](../../../tests/gpu_vector_ops_tests.rs) and [tests/gpu_stress_tests.rs](../../../tests/gpu_stress_tests.rs) skip aggressively outside macOS. The ROCm suite must introduce a `target_os = "linux"` + `feature = "rocm"` equivalent.

## Dependency on CUDA backend

Strictly speaking, ROCm could land before CUDA. But the following artifacts are better authored against CUDA first:

- `build.rs` skeleton (detect external toolchains, link system libraries).
- Cross-backend numerical tolerance envelope.
- Example format (`examples/cuda_basic.rs` → `examples/rocm_basic.rs`).

Delaying ROCm until CUDA Phase 3 is merged reduces duplicated design cost by roughly 30% and avoids churn in these shared artifacts.
