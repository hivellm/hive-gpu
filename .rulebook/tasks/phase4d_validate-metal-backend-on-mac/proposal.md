# Proposal: phase4d_validate-metal-backend-on-mac

## Why

Phases 4a (`finish-metal-bruteforce-search`) and 4c (`add-metal-ivf-index`)
were authored from a Windows / RTX 4090 host. The Metal code path is gated
to `cfg(target_os = "macos")` and was archived via `skipValidation: true`
because the tests that ship with them — `tests/metal_bruteforce.rs` and
`tests/metal_ivf.rs` — have never been run on real Apple Silicon hardware.

Those two test suites are already written; they just need to execute on a
Mac. When they pass, the Metal backend reaches functional parity with the
CUDA backend that shipped in 0.2.0. When they fail, the failures will point
to concrete Metal / `objc2-metal` / compute-kernel fixes that unblock the
`0.3.0-metal` release.

This is a one-time validation task, not a new feature. It is scoped
narrowly so a maintainer with a Mac can pick it up in a single session.

## What Changes

- Run the two existing test suites on Apple Silicon and record results:
  - `cargo test --features metal-native --test metal_bruteforce`
  - `cargo test --features metal-native --test metal_ivf`
- Fix any compilation / API / runtime errors that surface. Expected
  failure modes — listed as starting hypotheses, not certainties — are in
  the companion spec and the tasks.md checklist.
- Run the full Metal quality gate: `cargo clippy --features metal-native
  --lib --tests --benches -- -D warnings`, `cargo fmt --all --check`.
- Capture real performance numbers:
  - `cargo bench --features metal-native --bench gpu_operations` (extend
    with a real `search_bruteforce` group mirroring `benches/cuda_ops.rs`
    if the file does not already have one).
  - Adapt `benches/cuda_ivf.rs` to `benches/metal_ivf.rs` (build time vs
    `n_list`, search latency vs `nprobe`, IVF vs brute-force at 1M).
- Land the numbers in `docs/benchmarks/PERFORMANCE.md`:
  - Replace the historical Apple M1 Pro Metal section (which carries
    fabricated search-latency figures) with measurements from the Mac
    host running this task.
  - Add a Metal IVF section alongside the existing CUDA IVF section.
- Update `CHANGELOG.md` with a `0.2.1` patch entry if only brute-force
  validation passes, or a `0.3.0` minor entry if Metal IVF also lands.

## Impact

- Affected specs: none new — this task consumes the specs delivered by
  phase4a and phase4c.
- Affected code: likely small edits inside
  - `src/metal/vector_storage.rs::run_sgemv_dot` (compute-encoder API
    names are the most likely mismatch)
  - `src/metal/ivf.rs::dispatch_sgemm_dot` (same API surface)
  - `src/metal/context.rs::compute_pipeline` (pipeline construction)
  - `src/shaders/metal_hnsw.metal` (if the new kernels fail to compile
    under the Metal shader compiler's strict typing)
- Breaking change: NO — all work is behind the `metal-native` feature
  which does not compile on non-macOS hosts.
- User benefit: Metal backend stops lying. Post-validation a macOS
  developer calling `search()` gets real GPU-computed distances instead of
  the mock scores that 0.1.x shipped with; `MetalIvfIndex` becomes
  available for production use.

## Prerequisite for the Mac maintainer

Before touching code, read (or re-read):

- [`docs/analysis/cuda/05-implementation-plan.md`](../../../docs/analysis/cuda/05-implementation-plan.md)
  for the reference shape that CUDA IVF followed and Metal IVF mirrors.
- The CUDA IVF implementation that's already green on RTX 4090:
  - [`src/cuda/ivf.rs`](../../../src/cuda/ivf.rs)
  - [`tests/cuda_ivf.rs`](../../../tests/cuda_ivf.rs)
  - [`benches/cuda_ivf.rs`](../../../benches/cuda_ivf.rs)
- The Metal code being validated:
  - [`src/metal/vector_storage.rs`](../../../src/metal/vector_storage.rs)
    (search path rewritten in phase4a)
  - [`src/metal/ivf.rs`](../../../src/metal/ivf.rs) (phase4c)
  - [`src/shaders/metal_hnsw.metal`](../../../src/shaders/metal_hnsw.metal)
    (sgemv_dot and sgemm_dot kernels added at the end of the file)

## Expected Mac-side failures to watch for

These are the plausible breakages, ranked by probability. Treat them as
starting hypotheses — if the compiler / runtime complains about something
else, trust the tool over this list.

1. **`objc2-metal` method name mismatch.** The helper uses
   `setBuffer_offset_atIndex`, `setBytes_length_atIndex`,
   `dispatchThreads_threadsPerThreadgroup`,
   `newFunctionWithName`, `newComputePipelineStateWithFunction_error`.
   If any of these were renamed in the `objc2-metal 0.3.x` minor range,
   the fix is a straight substitution — the semantics are stable.
2. **`computeCommandEncoder` availability.** On older Macs the method is
   only exposed through an options variant. All supported targets
   (Apple Silicon on macOS 13+) have the no-argument form; failing
   macOS 12 and earlier is acceptable.
3. **`dispatchThreads_threadsPerThreadgroup` vs `dispatchThreadgroups`**.
   The former is the non-uniform grid variant (macOS 10.15+); Apple
   Silicon supports it but Intel-integrated Macs on older macOS may
   not. If that surfaces, switch to the threadgroup-count form with a
   ceiling-div on the grid.
4. **`MTLBuffer::contents()` pointer lifetime.** Current code copies
   straight out of a shared-mode buffer after
   `command_buffer.waitUntilCompleted()`. Should be fine; if the buffer
   is unexpectedly private-mode, route readback through a blit encoder
   into a staging buffer instead.
5. **Kernel compile error in `sgemm_dot`.** The 2D grid dispatch uses
   `uint2 gid [[thread_position_in_grid]]`. If the Metal compiler
   rejects that tag in a library compiled with `newLibraryWithSource`,
   rewrite with separate `uint x [[thread_position_in_grid.x]]` /
   `uint y [[thread_position_in_grid.y]]` attributes — semantically
   identical.
6. **Recall threshold too tight.** `tests/metal_bruteforce.rs` asserts
   `1e-3` tolerance and `tests/metal_ivf.rs` asserts `>= 0.65` random
   recall. Both are conservative versions of what the CUDA suite uses
   and landed green there; any Metal-side number that lands inside a
   tight window (say 0.60 vs 0.65 recall) is a knob-tuning question,
   not a correctness bug — document the empirical number and bump or
   tighten the threshold to match reality.

None of these require architectural changes. Budget: **half a day on an
Apple Silicon host**, plus benchmark capture and `docs/benchmarks/`
updates.
