# 06 — Implementation Plan

Recommended sequence, assuming the **CUDA backend is already functional** (reduces risk and accelerates development).

## Phase 0 — Prerequisites (out of scope here)

- Functional CUDA backend (see [../cuda/05-implementation-plan.md](../cuda/05-implementation-plan.md)).
- Generic `build.rs` introduced in the project.

## Phase 1 — Infrastructure (2 days)

1. Add the `rocm` feature to [Cargo.toml](../../../Cargo.toml) with `hip-sys` / `rocblas-sys` optional.
2. Add `GpuBackendType::Rocm` to [src/backends/detector.rs:10](../../../src/backends/detector.rs#L10) and update `detect_available_backends` / `select_best_backend` (prioritize ROCm after CUDA, before CPU).
3. Add `HiveGpuError::RocmError(String)`, `RocblasError(String)`, `HipError(String)` to [error.rs](../../../src/error.rs).
4. Create empty scaffolding `src/rocm/{mod.rs,context.rs,vector_storage.rs,buffer_pool.rs,vram_monitor.rs}` mirroring [src/cuda/](../../../src/cuda/).
5. Update [src/lib.rs](../../../src/lib.rs) with `#[cfg(feature = "rocm")] pub mod rocm;`.

**Exit criterion:** `cargo check --features rocm` passes on Linux; `GpuBackendType::Rocm` is reachable through the detector.

## Phase 2 — Bindings + context (3–4 days)

1. Generate HIP bindings with `bindgen` in `build.rs`.
2. Implement `RocmContext` with:
   - `hipGetDeviceCount` / `hipSetDevice` / `hipStreamCreate`.
   - `rocblas_create_handle` bound to the stream.
   - `hipGetDeviceProperties` → populate `GpuDeviceInfo` for real (including `compute_capability: "gfx1030"` etc.).
3. Ordered `Drop`: rocBLAS handle → stream → optional device reset.
4. Test: `tests/rocm_device_info.rs` with graceful skip.

**Exit criterion:** `GpuDeviceInfo` returned matches `rocm-smi` output; compute capability reflects the actual gfx architecture.

## Phase 3 — Vector Storage (3–4 days)

1. `RocmVectorStorage` with `*mut f32` + capacity tracking.
2. `add_vectors` with batched `hipMemcpyAsync` + `hipStreamSynchronize`.
3. `ensure_capacity` with D2D reallocation.
4. `clear` / `remove_vectors` (masking, no free).
5. Parity test with Metal / CUDA.

**Exit criterion:** storing 10k × 128-dim vectors succeeds on gfx90a and gfx1030; grow/shrink cycles stable over 5-minute stress.

## Phase 4 — HIP kernels (4–5 days)

1. Create `src/rocm/kernels.hip` with:
   ```cpp
   extern "C" __global__ void hip_l2_distance_kernel(
       const float* __restrict__ query,
       const float* __restrict__ vectors,
       float* __restrict__ out,
       int n, int d);
   ```
2. Compile multi-arch via `build.rs` + `hipcc`.
3. Launcher in `src/rocm/kernels.rs`.
4. Cosine via `rocblas_sgemv` + a normalization kernel.

**Exit criterion:** numerical agreement with Metal and CUDA within `1e-4` on 1000 random queries, on both a wave=32 and a wave=64 device.

## Phase 5 — Cross-backend consistency (2 days)

1. New test `tests/cross_backend_consistency.rs` — same vectors, same query, compare top-K between Metal / CUDA / ROCm within tolerance.
2. Document acceptable numerical divergences.

**Exit criterion:** consistency test green on at least two of the three backends on every CI run; all three on nightly with self-hosted runners.

## Phase 6 — CI + Docs (2 days)

1. GitHub Actions workflow with container `rocm/dev-ubuntu-22.04:6.0` (build-only; GPU tests require a self-hosted runner or AMD CI — can be deferred).
2. Benchmarks in [benches/gpu_operations.rs](../../../benches/) with the `rocm` feature.
3. Update [README.md](../../../README.md) removing the ROCm-absent note.
4. Update [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) with numbers.

**Exit criterion:** `cargo build --features rocm` green on a clean ROCm 6.0 container; benchmark table populated with at least MI210 and RX 7900 XTX numbers.

## Total effort

- **Functional parity with Metal / CUDA:** 16–21 dev-days once CUDA is ready.
- **If done in parallel with CUDA:** expect 25–30 days due to the friction of maintaining two backends simultaneously.

## Sequencing constraints

- Do **not** start Phase 1 until the CUDA `build.rs` pattern (if any) is frozen.
- Do **not** merge Phase 4 without access to both a wave=32 (RDNA) and a wave=64 (CDNA) GPU for testing.
- Do **not** open Phase 6's public documentation until the numerical agreement from Phase 5 is captured — advertising a backend that is not measurably consistent is worse than no advertisement at all.
