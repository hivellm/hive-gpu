# Proposal: phase3b_add-rocm-backend

Source: docs/analysis/gcn/

## Why

There is zero AMD code in the project, yet `src/types.rs` already documents
`backend: "ROCm"` and `compute_capability: "gfx1030"` as valid values in the
public API — making ROCm support a pending promise rather than an optional
feature. AMD GPUs represent ~15% of the ML/AI and HPC market (Instinct
MI200/MI300 series in datacenter, RX 6000/7000 in prosumer), and combined
with CUDA + Metal this closes coverage to ~90% of production GPU silicon.

This task must start only after the CUDA backend (phase3a) is merged so
ROCm can mirror the stabilized shape — HIP's API is deliberately
CUDA-like, so the algorithmic work collapses into a translation effort
rather than a ground-up design.

## What Changes

- Add `src/rocm/` module mirroring the CUDA structure after phase3a ships.
- Bind to HIP via `bindgen` against `$ROCM_PATH/include/hip/` inside
  `build.rs` for a focused subset (~30 functions) plus `rocblas-sys` for
  BLAS routines. No third-party Rust HIP crate is mature enough to depend
  on as a transitive dep.
- Implement `RocmContext` with `hipGetDeviceCount`, `hipSetDevice`,
  `hipStreamCreate`, and a `rocblas_create_handle` bound to the stream.
- Populate `GpuDeviceInfo` from `hipGetDeviceProperties` with the real
  gfx architecture string (gfx900 through gfx1100+).
- Implement `RocmVectorStorage` with `hipMalloc` / `hipMemcpyAsync`,
  D2D reallocation on growth, and soft-delete via `removed_indices`
  matching the Metal / CUDA pattern.
- Author `src/rocm/kernels.hip` for L2 distance, with Cosine / DotProduct
  routed through `rocblas_sgemv` where possible.
- Compile kernels via `hipcc --offload-arch=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100`
  triggered from `build.rs`. Allow a HIP-source-JIT fallback for hosts
  without `hipcc` in the PATH.
- Kernels MUST use `warpSize` at runtime (not hard-coded 32 or 64) because
  AMD wavefronts are 64 on CDNA/Vega and 32 on RDNA/RDNA2/RDNA3.
- Add `GpuBackendType::Rocm` to `src/backends/detector.rs` with
  `is_rocm_available()` via lazy HIP loader; priority order becomes
  Metal > CUDA > ROCm > CPU.
- Add `HiveGpuError::{HipError, RocblasError, RocmError}` variants.
- Extend `tests/cross_backend_consistency.rs` (introduced in phase3a) to
  include ROCm within the `1e-4` tolerance envelope.
- CI using the `rocm/dev-ubuntu-22.04:6.0` container for build
  verification; real GPU tests require a self-hosted runner.

## Impact

- Affected specs: new `rocm-backend` spec.
- Affected code: new `src/rocm/*`, updates to `src/lib.rs`, `src/error.rs`,
  `src/backends/detector.rs`, `Cargo.toml`, `build.rs` (extend phase3a
  pattern), new `tests/rocm_*.rs`, new `examples/rocm_basic.rs`,
  new `docs/guides/ROCM_SETUP.md`.
- Breaking change: NO. Feature-gated behind `rocm`.
- User benefit: production-ready ROCm acceleration on gfx900+ (Vega and
  later) covering AMD Instinct MI series (MI50/MI100/MI210/MI250/MI300)
  and Radeon RX 6000/7000 series. Unlocks HPC / cost-sensitive cloud
  deployments where AMD is chosen over NVIDIA.
- HNSW is not part of v1 (brute-force search only, matching the CUDA
  scope baseline).
