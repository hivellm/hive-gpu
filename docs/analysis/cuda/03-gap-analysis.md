# 03 — Gap Analysis vs. Requirements

The existing proposal in [openspec/changes/add-cuda-backend/](../../../openspec/changes/add-cuda-backend/) lists 12 task sections. Mapped against the current code:

| # | Area | Status | Observation |
|---|---|---|---|
| 1 | `Cargo.toml` + `build.rs` | ❌ | Empty feature, no build script |
| 2 | Errors (`CudaError`, `CublasError`) in `HiveGpuError` | ❌ | [error.rs](../../../src/error.rs) only carries generic variants |
| 3 | `CudaContext` with stream + cuBLAS handle | ❌ | Struct exists, no real CUDA fields |
| 4 | `CudaVectorStorage` with `cudaMalloc` / `cudaMemcpyAsync` | ❌ | CPU-only bookkeeping |
| 5 | `.cu` kernels (L2, Cosine via SGEMV) | ❌ | Non-existent |
| 6 | Module organization | 🟡 | Facade OK, implementation empty |
| 7 | Tests `tests/cuda_tests.rs` | ❌ | File does not exist |
| 8 | Example + docs | 🟡 | Example exists but breaks |
| 9 | Benchmarks | ❌ | [benches/gpu_operations.rs](../../../benches/) is Metal-only |
| 10 | `clippy` / `fmt` with feature `cuda` | ❌ | Not verified in CI |
| 11 | CI job for `nvidia/cuda:12.x` | ❌ | No workflow |
| 12 | `CHANGELOG.md` + versioning | N/A | To be done after implementation |

## Cross-reference with `docs/ROADMAP.md`

The [roadmap](../../ROADMAP.md) places CUDA as **Phase 3.1** (highest priority after Device Info API). Phase 2 (Device Info API) is already **implemented** for Metal in [src/metal/context.rs:138-171](../../../src/metal/context.rs#L138-L171), so the prerequisite is met. No blockers remain from a sequencing standpoint.

## What the Metal backend already provides (and CUDA must replicate)

Reading [src/metal/vector_storage.rs](../../../src/metal/vector_storage.rs) as the reference implementation:

1. **VRAM-only storage** via `MTLResourceOptions::StorageModePrivate` → CUDA uses `cudaMalloc` (VRAM is default).
2. **Staging buffer** with `StorageModeShared` + blit copy → CUDA uses pinned host memory via `cudaMallocHost` + `cudaMemcpyAsync`.
3. **Adaptive growth** with factors 2.0 → 1.5 → 1.2 and a 1 GB cap ([vector_storage.rs:231-256](../../../src/metal/vector_storage.rs#L231-L256)).
4. **Soft delete** via `removed_indices: HashSet<usize>` ([vector_storage.rs:43](../../../src/metal/vector_storage.rs#L43)).
5. **Payload kept on CPU** in `HashMap<String, Option<HashMap<...>>>` ([vector_storage.rs:44](../../../src/metal/vector_storage.rs#L44)).
6. **Validation**: dimension, NaN/Inf rejection, ID length, uniqueness ([vector_storage.rs:106-138](../../../src/metal/vector_storage.rs#L106-L138)).

All six points are missing in the CUDA module. The CUDA implementation should **port these invariants line by line** to guarantee cross-backend consistency tests pass trivially.
