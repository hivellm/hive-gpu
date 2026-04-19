# 05 — Implementation Plan

Phased plan, each phase independently mergeable.

## Phase 1 — Infrastructure (2–3 days)

1. Enable `cudarc` in [Cargo.toml](../../../Cargo.toml) behind the `cuda` feature.
2. Add variants `HiveGpuError::CudaError(String)` and `CublasError(String)` in [error.rs](../../../src/error.rs).
3. Create a minimal `build.rs` (only `rerun-if-changed` for now).
4. Update [backends/detector.rs](../../../src/backends/detector.rs) to detect via `cudarc`.
5. Remove `#![allow(warnings)]` from [src/lib.rs:6](../../../src/lib.rs#L6).

**Exit criterion:** `cargo check --features cuda` passes; `detect_available_backends()` returns `Cuda` on a host with a driver installed.

## Phase 2 — Real context (2 days)

1. Rewrite [cuda/context.rs](../../../src/cuda/context.rs) using `cudarc::driver::CudaDevice`.
2. Populate `GpuDeviceInfo` with real data (`cuDeviceGetAttribute`, `cuDeviceTotalMem`, optionally `nvmlDeviceGetMemoryInfo`).
3. Implement `is_available()` for real.
4. Test: `tests/cuda_device_info.rs` — skip gracefully when no GPU is present.

**Exit criterion:** `GpuDeviceInfo` returned matches `nvidia-smi` output within tolerance; compute capability reflects actual hardware.

## Phase 3 — Vector Storage (3–4 days)

1. Implement `CudaVectorStorage` using `DeviceSlice<f32>` (cudarc).
2. `add_vectors` → batch `htod_copy` (single `cudaMemcpyAsync`).
3. `expand_buffer` mirroring the Metal pattern.
4. `remove_vectors` via masking (`removed_indices: HashSet<usize>`).
5. Tests: add/search/remove/clear parity with [tests/gpu_vector_ops_tests.rs](../../../tests/gpu_vector_ops_tests.rs).

**Exit criterion:** storing 10k × 128-dim vectors succeeds without leaks; repeated expand/shrink cycles are stable under 5-minute stress test.

## Phase 4 — Distance kernels (3 days)

1. Write `src/cuda/kernels.cu` with:
   - `l2_distance_kernel(query, vectors, out, n, d)`
   - `cosine_distance_kernel` (or SGEMV + norm).
2. Compile offline to multi-arch PTX; embed via `include_str!`.
3. Rust launcher in `src/cuda/kernels.rs`.
4. Top-K initially on CPU (simplicity).

**Exit criterion:** numerical agreement with Metal within `1e-4` across 1000 random queries.

## Phase 5 — HNSW (5–7 days, optional v2)

Port the logic from [metal_hnsw.metal](../../../src/shaders/metal_hnsw.metal) to CUDA. This can be deferred — the CUDA path can ship as brute-force search only and still deliver major speedups over CPU.

**Exit criterion:** recall@10 ≥ 0.95 on a 100k × 128-dim sift1m subset; search latency within 2× of a well-tuned CPU HNSW (hnswlib) baseline.

## Phase 6 — Tests and CI (2 days)

1. `tests/cuda_integration.rs` replicating the scenarios in [tests/integration_tests.rs](../../../tests/integration_tests.rs).
2. Cross-backend consistency test (Metal × CUDA with 1e-5 tolerance).
3. GitHub Actions workflow using container `nvidia/cuda:12.4-devel-ubuntu22.04` (build-only; GPU tests need a self-hosted runner or skip).

**Exit criterion:** CI green on Linux + Windows; self-hosted runner (optional) executes the full suite on real hardware at least weekly.

## Phase 7 — Docs / bench / release (1 day)

1. Update [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) with real numbers.
2. Adjust the "CUDA backend is currently in development" section in [README.md](../../../README.md).
3. Run benchmarks in [benches/gpu_operations.rs](../../../benches/) with feature `cuda`.
4. Draft `CHANGELOG.md` entry: supported SMs, known limitations, upgrade notes.

**Exit criterion:** `cargo publish --dry-run --features cuda` succeeds; documented performance numbers reproducible on an RTX 4090 reference host.

## Total effort

- **Functional parity with Metal:** 15–20 dev-days.
- **Plus HNSW:** +7 days.

## Suggested sequencing vs. team

If the team has only one Rust + CUDA engineer, do the phases strictly in order. If two engineers can pair, Phase 4 (kernels) can overlap Phase 3 (storage) once the buffer allocation API is frozen at the end of Phase 3 day 1.
