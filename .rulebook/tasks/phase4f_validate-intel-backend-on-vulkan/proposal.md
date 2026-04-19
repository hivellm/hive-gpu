# Proposal: phase4f_validate-intel-backend-on-vulkan

## Why

`phase3c_add-intel-backend` shipped a complete Intel Vulkan Compute
backend (context + brute-force search + IVF) authored from a Windows /
RTX 4090 workstation without access to Intel Arc / Battlemage hardware
and without a Vulkan-capable GPU under `HIVE_GPU_VULKAN_UNIVERSAL=1`
fallback. The code is cross-platform-clean (`cargo check` and
`clippy -D warnings` green on every feature combination, `cargo fmt`
green) but the `intel`-gated paths — every `ash` entry point, every
SPIR-V dispatch, every push-constant upload — have never executed.

The two test suites that ship with the backend (`tests/intel_smoke.rs`
and `tests/intel_ivf.rs`, 9 tests total) need to run on a Vulkan 1.2+
device. The preferred target is Intel Arc / Battlemage (vendor id
`0x8086`) because that is what the backend was designed for; any
Vulkan-capable GPU in universal-fallback mode is acceptable as a
secondary target. When the suites pass, the Intel backend reaches
functional parity with the CUDA backend that shipped in 0.2.0. When
they fail, the failures will point at concrete `ash` / SPIR-V / memory
type fixes.

Same shape as `phase4d_validate-metal-backend-on-mac` and
`phase4e_validate-rocm-backend-on-amd`: a one-turn task for a
maintainer with the right hardware, not a new feature.

## What Changes

- Run the two existing test suites on a Linux or Windows host with a
  Vulkan-capable GPU (Intel Arc preferred, anything else under
  `HIVE_GPU_VULKAN_UNIVERSAL=1` acceptable):
  - `cargo test --features intel --test intel_smoke`
  - `cargo test --features intel --test intel_ivf`
- Fix any compilation / validation-layer / runtime errors that
  surface. The most likely failure modes are listed below. None of
  them require architectural changes.
- Run the quality gate:
  - `cargo clippy --features intel --lib --tests --benches -- -D warnings`
  - `cargo fmt --all --check`
  - `cargo doc --no-deps --features intel`
- Port `benches/cuda_ops.rs` → `benches/intel_ops.rs` and
  `benches/cuda_ivf.rs` → `benches/intel_ivf.rs` (register both in
  `Cargo.toml`).
- Capture real numbers on at least one Intel Arc / Battlemage card (or
  the best Vulkan GPU available) and land them in
  `docs/benchmarks/PERFORMANCE.md` alongside the existing CUDA IVF
  section.
- Update `README.md` backend matrix: mark Intel as shipping, remove the
  "Intel designed but not implemented" caveat.
- Update `CHANGELOG.md`:
  - `0.2.3` if only brute-force passes
  - `0.3.0` (or the next minor) if IVF also passes
- Tag the release once merged.

## Prerequisite reading for the Intel / Vulkan maintainer

- [`src/intel/context.rs`](../../../src/intel/context.rs) — Vulkan
  instance creation, device selection (Intel vendor filter +
  universal-fallback env var), queue + command pool + descriptor pool
  setup, SPIR-V pipeline pre-build. Start here.
- [`src/intel/vector_storage.rs`](../../../src/intel/vector_storage.rs)
  — brute-force search path. Look at
  `dispatch_three_buffer_compute_ranged` for the dispatch scaffold.
- [`src/intel/ivf.rs`](../../../src/intel/ivf.rs) — IVF index. Uses
  the same two pre-built compute pipelines (`sgemv_dot`, `sgemm_dot`).
- [`src/intel/shaders/sgemv_dot.wgsl`](../../../src/intel/shaders/sgemv_dot.wgsl)
  and [`sgemm_dot.wgsl`](../../../src/intel/shaders/sgemm_dot.wgsl) —
  WGSL compute kernels compiled to SPIR-V by `build.rs` via `naga`.
- [`build.rs`](../../../build.rs) — `compile_intel_shaders`, the
  WGSL→SPIR-V compile step that runs when `CARGO_FEATURE_INTEL` is
  set.
- [`src/cuda/ivf.rs`](../../../src/cuda/ivf.rs) and
  [`benches/cuda_ivf.rs`](../../../benches/cuda_ivf.rs) — the
  already-validated CUDA reference the Intel code was ported from.

## Expected failure modes

Ranked by probability. Treat them as hypotheses — if the validation
layer reports something else, trust the validation layer.

1. **Validation layer errors on descriptor-set binding order.** The
   Vulkan spec is strict about binding the descriptor set *after* the
   pipeline is bound, and push constants must be set before dispatch.
   If validation complains, it is almost always a reordering fix
   inside `dispatch_three_buffer_compute_ranged` in
   `src/intel/vector_storage.rs`.
2. **Memory type selection picks the wrong heap.** Intel integrated
   GPUs and Arc discrete cards expose different memory type bit
   layouts. The helper `find_memory_type` in `vector_storage.rs`
   picks the first `HOST_VISIBLE | HOST_COHERENT` type; if mapping
   fails, add `DEVICE_LOCAL` to the candidate list and retry.
3. **SPIR-V binding indices mismatch.** The WGSL shaders declare
   `@group(0) @binding(0..2)` and push constants in `pc`. If naga
   emits different SPIR-V decorations than expected, the descriptor
   set layout built in `context.rs::build_compute_pipeline` will
   disagree with the shader. Diagnose with `spirv-cross --reflect`
   on the compiled `.spv` in `target/.../build/.../out/*.spv`.
4. **SGEMM operand orientation.** The IVF assignment SGEMM is
   transcribed from cuBLAS convention. Unlike rocBLAS / cuBLAS the
   custom WGSL kernel is row-major throughout, so the operand layout
   differs. If recall tests fail with very low numbers (< 0.3) the
   culprit is almost certainly a transpose mismatch in
   `src/intel/ivf.rs::assign_to_centroids` — compare index formulas
   with the CUDA reference and swap if needed.
5. **Push-constant size mismatch.** WGSL struct layout rules round
   up to 16-byte alignment for the push-constant block. If the host
   sends 8 bytes (`u32 + u32`) but the shader expects 16, validation
   will complain. Pad the Rust side to `[u32; 4]` if needed.
6. **Universal-fallback mode refuses to select an integrated GPU.**
   If the host only has an Intel iGPU / AMD iGPU / NVIDIA dGPU, set
   `HIVE_GPU_VULKAN_UNIVERSAL=1` before running the tests. If the
   fallback still refuses, check `IntelContext::new_with_preference`
   for the vendor-filter logic and confirm the env-var parse matches.
7. **Recall threshold too tight.** `tests/intel_ivf.rs` asserts
   `>= 0.65` random recall at `nprobe = n_list / 4` — conservatively
   below the CUDA suite's 0.76 result. If the measured number comes
   in at 0.60 after inspection confirms everything is correct,
   document the Intel-specific number and tune the threshold.

## Impact

- Affected specs: none new — consumes the specs delivered by phase3c.
- Affected code:
  - likely small edits inside `src/intel/context.rs` (vendor filter,
    memory type selection, descriptor layout) and
    `src/intel/vector_storage.rs` (dispatch order, buffer mapping)
  - potentially `src/intel/ivf.rs` if SGEMM operand orientation needs
    flipping
  - possibly `src/intel/shaders/*.wgsl` if SPIR-V decorations need
    tightening to match the descriptor layout
  - new `benches/intel_ops.rs` and `benches/intel_ivf.rs` ported from
    the CUDA equivalents
- Breaking change: NO — all work is behind the `intel` feature and
  cfg-gated to Linux / Windows.
- User benefit: Intel backend stops being "written but unverified".
  Intel Arc / Battlemage users get brute-force + IVF parity with the
  CUDA 0.2.0 release. Combined with the CUDA + ROCm + Metal backends,
  total market coverage reaches ~95 % (~5 % Metal + ~70 % CUDA +
  ~15 % ROCm + ~5 % Intel dGPU + universal Vulkan fallback).

Budget estimate: **half a day on a Linux or Windows + Vulkan host**,
plus benchmark capture and docs.
