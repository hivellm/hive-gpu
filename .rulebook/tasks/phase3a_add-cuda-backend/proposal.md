# Proposal: phase3a_add-cuda-backend

Source: docs/analysis/cuda/

## Why

The `cuda` feature in `Cargo.toml` is empty (`cuda = []`) and the `src/cuda/`
module is a stub — every real path returns `"not implemented yet"`. The
README advertises CUDA support, creating a broken promise for ~70% of the
GPU market (NVIDIA on Linux/Windows — cloud servers, ML/AI infrastructure).

Delivering a working CUDA backend is the single highest-ROI workstream on
the roadmap: 15–20 dev-days for 10–50× speedup over CPU and production
readiness on cloud providers that dominate the market. The Metal backend
already defines the contract (`GpuBackend`, `GpuContext`, `GpuVectorStorage`
in `src/traits.rs`), so CUDA only has to mirror proven invariants rather
than invent new ones.

## What Changes

- Replace CUDA stubs in `src/cuda/` with a real backend using the `cudarc`
  driver-API crate (pure Rust bindings).
- Populate `GpuDeviceInfo` from `cuDeviceGetAttribute` + `cuDeviceTotalMem`.
- Implement `CudaVectorStorage` with real `cudaMalloc` / `cudaMemcpyAsync`,
  dynamic buffer expansion, and soft-delete tracking mirroring the Metal
  backend shape.
- Ship compute kernels for L2, Cosine, and DotProduct distances. Kernels
  authored in `.cu` and distributed as embedded PTX compiled offline with
  `nvcc`. Top-K sort stays on CPU in v1.
- Real backend detection in `src/backends/detector.rs` via
  `cudarc::driver::CudaDevice::count()` rather than env-var inspection.
- Add `HiveGpuError::CudaError` / `CublasError` variants.
- Remove `#![allow(warnings)]` from `src/lib.rs:6` so clippy can police the
  new unsafe surface.
- Add cross-backend consistency tests (Metal ↔ CUDA within `1e-4`
  tolerance).
- CI: `nvidia/cuda:12.4-devel-ubuntu22.04` for build verification;
  self-hosted runner optional for real GPU tests.

## Impact

- Affected specs: new `cuda-backend` spec in `specs/cuda-backend/spec.md`.
- Affected code: `src/cuda/*` (rewrite), `src/lib.rs`, `src/error.rs`,
  `src/backends/detector.rs`, `Cargo.toml`, new `build.rs` (minimal),
  new `tests/cuda_*.rs`, updated `examples/cuda_basic.rs`.
- Breaking change: NO. All CUDA work is feature-gated behind `cuda` feature.
  Existing Metal users are untouched.
- User benefit: production-ready CUDA acceleration on NVIDIA (Volta+ / sm_70+)
  with VRAM-only storage, batch uploads, and GPU-accelerated distance
  computation. Unlocks cloud and on-prem ML/AI deployments.
- HNSW implementation is deferred to a follow-up task; v1 ships brute-force
  search on GPU plus CPU top-K.
