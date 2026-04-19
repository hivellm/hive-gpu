# Proposal: phase4e_validate-rocm-backend-on-amd

## Why

`phase3b_add-rocm-backend` shipped a complete ROCm backend (context +
brute-force search + IVF) authored from a Windows / RTX 4090 workstation
without access to AMD hardware. The code is cross-platform-clean
(`cargo check` and `clippy -D warnings` green on every feature
combination) but the Linux + ROCm-gated paths — every HIP or rocBLAS
call — have never executed.

The two test suites that ship with the backend (`tests/rocm_smoke.rs`
and `tests/rocm_ivf.rs`, 11 tests total) need to run on an AMD GPU
supported by ROCm (gfx900 through gfx1100). When they pass, the ROCm
backend reaches functional parity with the CUDA backend that shipped in
0.2.0. When they fail, the failures will point at concrete HIP /
rocBLAS / libloading fixes.

Same shape as `phase4d_validate-metal-backend-on-mac`: a one-turn task
for a maintainer with the right hardware, not a new feature.

## What Changes

- Run the two existing test suites on a Linux + AMD host:
  - `cargo test --features rocm --test rocm_smoke`
  - `cargo test --features rocm --test rocm_ivf`
- Fix any compilation / FFI / runtime errors that surface. The most
  likely failure modes are listed below and in the companion spec.
  None of them require architectural changes.
- Run the quality gate:
  - `cargo clippy --features rocm --lib --tests --benches -- -D warnings`
  - `cargo fmt --all --check`
  - `cargo doc --no-deps --features rocm`
- Port `benches/cuda_ops.rs` → `benches/rocm_ops.rs` and
  `benches/cuda_ivf.rs` → `benches/rocm_ivf.rs` (register both in
  `Cargo.toml`).
- Capture real numbers on at least one AMD GPU (MI210, RX 7900 XTX, or
  whatever hardware the maintainer has access to) and land them in
  `docs/benchmarks/PERFORMANCE.md` alongside the existing CUDA IVF
  section.
- Update `README.md` backend matrix: mark ROCm as shipping, remove the
  "ROCm designed but not implemented" caveat.
- Update `CHANGELOG.md`:
  - `0.2.2` if only brute-force passes
  - `0.3.0` (or `0.4.0` if phase4d Metal lands concurrently) if IVF
    also passes
- Tag the release once merged.

## Prerequisite reading for the AMD maintainer

- [`src/rocm/ffi.rs`](../../../src/rocm/ffi.rs) — the FFI layer. Start
  here. The `first_loadable` helper tries `libamdhip64.so` →
  `libamdhip64.so.6` → `libamdhip64.so.5`; add more if the local ROCm
  install uses a different SONAME.
- [`src/rocm/context.rs`](../../../src/rocm/context.rs) — context
  construction, device info queries, ordered Drop.
- [`src/rocm/vector_storage.rs`](../../../src/rocm/vector_storage.rs) —
  brute-force search path.
- [`src/rocm/ivf.rs`](../../../src/rocm/ivf.rs) — IVF index.
- [`src/cuda/ivf.rs`](../../../src/cuda/ivf.rs) and
  [`benches/cuda_ivf.rs`](../../../benches/cuda_ivf.rs) — the
  already-validated CUDA reference the ROCm code was ported from.

## Expected failure modes

Ranked by probability. Treat them as hypotheses — if the tool reports
something else, trust the tool.

1. **Library SONAME mismatch.** `first_loadable` tries three common
   candidates per library. If the distro packages `libamdhip64.so.7`
   (ROCm 7.x) add it to the list.
2. **Device-attribute enum values.** The integer constants for
   `HIP_DEVICE_ATTR_*` in `src/rocm/ffi.rs` match ROCm 6.x. If the
   attribute values differ in the installed ROCm version,
   `hipDeviceGetAttribute` returns success with nonsense values —
   diagnose with a side-by-side `rocminfo` / `hipinfo` comparison and
   patch the constants.
3. **rocBLAS SGEMM operand orientation.** The IVF assignment SGEMM is
   transcribed from cuBLAS convention (column-major with swapped
   operands to get row-major output). If recall tests fail with very
   low numbers (< 0.3) the culprit is almost certainly a wrong
   `trans` flag — flip `ROCBLAS_OP_T` / `ROCBLAS_OP_N` and retry.
4. **`hipMemcpy` kind values.** We hard-code `HIP_MEMCPY_HOST_TO_DEVICE
   = 1`, `DEVICE_TO_HOST = 2`, `DEVICE_TO_DEVICE = 3`. Verify against
   `hipruntime_api.h`.
5. **Recall threshold too tight.** `tests/rocm_ivf.rs` asserts
   `>= 0.65` random recall at `nprobe = n_list / 4` — conservatively
   below what the CUDA suite achieves (0.76). If the measured number
   comes in at 0.60 after inspection confirms everything is correct,
   document and tune the threshold.

## Impact

- Affected specs: none new — consumes the specs delivered by phase3b.
- Affected code:
  - likely small edits inside `src/rocm/ffi.rs` (SONAME list, attribute
    constants)
  - potentially `src/rocm/ivf.rs` or `src/rocm/vector_storage.rs` if
    rocBLAS operand orientation needs flipping
  - new `benches/rocm_ops.rs` and `benches/rocm_ivf.rs` ported from the
    CUDA equivalents
- Breaking change: NO — all work is behind the `rocm` feature and
  cfg-gated to Linux.
- User benefit: ROCm backend stops being "written but unverified".
  AMD users get brute-force + IVF parity with the CUDA 0.2.0 release,
  bringing total market coverage to ~90 % (~5 % Metal + ~70 % CUDA +
  ~15 % ROCm).

Budget estimate: **half a day on a Linux + AMD host**, plus benchmark
capture and docs.
