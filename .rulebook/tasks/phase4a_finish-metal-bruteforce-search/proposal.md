# Proposal: phase4a_finish-metal-bruteforce-search

## Why

The Metal backend is shipping with a **non-functional `search()`** path:
`src/metal/vector_storage.rs::search()` still contains a mock loop that
returns synthetic scores of the form `1.0 - (i as f32 * 0.1)`, tagged with a
`// TODO: Implement GPU-accelerated search using Metal shaders` comment. The
function compiles, 72 Metal tests pass, but none of them exercise search
accuracy — so downstream consumers (notably the Vectorizer) receive wrong
rankings silently.

Shipping the CUDA backend in 0.2.0 made this gap glaring: CUDA computes real
distances via cuBLAS SGEMV, Metal returns fake ones. Before we layer IVF on
top of Metal (phase4c), brute-force must be correct first — IVF reuses the
same distance path per probed cluster.

This task is scoped narrowly to **fix brute-force on Metal** using Metal
Performance Shaders (MPS). It is a prerequisite for any ANN work on Apple
hardware and a blocker for the Vectorizer integration path.

## What Changes

- Replace the mock loop in
  [`src/metal/vector_storage.rs::search()`](../../../src/metal/vector_storage.rs)
  with a real GPU-accelerated implementation, mirroring the shape of the
  CUDA backend in [`src/cuda/vector_storage.rs`](../../../src/cuda/vector_storage.rs).
- Use `MPSMatrixVectorMultiplication` from Metal Performance Shaders as the
  SGEMV equivalent. MPS is Apple-maintained and ships as part of macOS /
  iOS, so no new Rust crate is needed beyond the existing `objc2-metal`
  stack plus an `objc2-metal-performance-shaders` dependency.
- Cache squared vector norms on the CPU at `add_vectors` time, same pattern
  the CUDA backend uses. For Cosine metric divide dot products by
  `||v|| * ||q||`; for Euclidean derive `||v - q||^2` from dots + norms.
- Top-K stays on CPU after a single dtoh copy of the score vector — matches
  the CUDA scope and is sufficient until GPU radix-select lands in a later
  phase.
- Port the CUDA integration tests to Metal:
  - `tests/metal_bruteforce_smoke.rs` — context, Cosine, Euclidean, buffer
    growth parity with the CUDA suite.
  - `tests/metal_search_accuracy.rs` — numerical agreement with a CPU
    reference within 1e-3.
  - Extend `benches/gpu_operations.rs` with real search numbers on Apple
    Silicon.
- Remove the `// TODO` and mock-score comment from source.
- Update `docs/benchmarks/PERFORMANCE.md` with an Apple Silicon (M-series)
  column showing actual search latency instead of the previously fabricated
  numbers.

## Impact

- Affected specs: new spec `metal-bruteforce-search` under this task.
- Affected code:
  - `src/metal/vector_storage.rs` (full rewrite of the `search` impl plus
    CPU-side norm cache)
  - `Cargo.toml` (add `objc2-metal-performance-shaders` target-gated to
    macOS)
  - new `tests/metal_bruteforce_smoke.rs`, `tests/metal_search_accuracy.rs`
  - `benches/gpu_operations.rs` extended with real search benches
  - `docs/benchmarks/PERFORMANCE.md` Apple Silicon numbers refreshed
- Breaking change: **Behavioural for users relying on the mock scores.**
  Any consumer that happened to depend on the monotonic fake ordering will
  see real results after this lands. Since the mock ordering was
  insertion-order plus a synthetic score, any semantic use was a bug.
- User benefit:
  - Correct search results on Apple Silicon, matching the CUDA backend.
  - Cross-backend consistency test becomes meaningful — we can compare
    Metal and CUDA numerically.
  - Unblocks phase4c (Metal IVF), which needs a working brute-force
    primitive per probed cluster.
  - Unblocks the Vectorizer integration on macOS in production, not just
    as a toy.

This is a **correctness fix**, not a feature. It should land in a patch
release (0.2.1) ahead of any further work on Metal.
