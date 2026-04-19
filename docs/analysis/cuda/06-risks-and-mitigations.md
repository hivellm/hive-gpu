# 06 — Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| `cudarc` does not cover cuBLAS SGEMV | Medium | Medium | Complement with `cublas-sys` or `cublas-rs` |
| CI flakiness without a real GPU | High | Low | Skip when `is_available()==false`; self-hosted runner for daily smoke |
| Embedded PTX binaries grow too large | Medium | Low | Compress or lazy-download; restrict embedded SMs |
| Numerical divergence Metal ↔ CUDA | Medium | Medium | `1e-4` tolerance in consistency tests; consistent FMA flags |
| `#![allow(warnings)]` in [lib.rs:6](../../../src/lib.rs#L6) hides bugs | High | Medium | Remove before release; enforce clippy with `-D warnings` |
| Driver version skew (older CUDA on server, newer on dev) | Medium | Medium | Pin minimum driver (12.0+); document in README |
| Memory fragmentation after many expand/shrink cycles | Low | Medium | Add "compact" path that rebuilds buffer on idle |
| `cudaMemcpyAsync` without matching `cudaStreamSynchronize` races | Medium | High | Code review checklist; `Drop` impl forces sync |
| Running example without `cuda` feature silently misleads users | High | Low | Compile-time `#[cfg(feature = "cuda")]` gate with a clear `compile_error!` message in `examples/cuda_basic.rs` |
| Licensing of embedded PTX derived from open kernels | Low | Low | All custom kernels shipped under Apache 2.0 matching the crate |

## Watch list beyond v1

- **Multi-GPU:** not scoped here; when it lands, the `CudaContext` must stop assuming `device_id = 0`. Plan for `Arc<CudaContext>` → `Arc<CudaDevice>` split.
- **Tensor Cores / `cutlass`:** worth evaluating for large-dim similarity but adds a heavy dependency; out of scope until performance numbers from the baseline are measured.
- **CUDA Graphs:** stream-of-work optimization that can shave 10–15% off repeated searches; low priority until v0.2 is stable.

## Exit criteria for "CUDA is production-ready"

1. `tests/cross_backend_consistency.rs` passes on every CI run.
2. Numerical divergence ≤ `1e-4` over ≥10k random queries.
3. No `unsafe` block without a `// SAFETY:` comment.
4. `cargo clippy --features cuda -- -D warnings` is green.
5. `cargo test --features cuda` runs the full suite on at least one self-hosted runner.
6. Memory leak suite (valgrind or `compute-sanitizer --tool memcheck`) clean on a representative workload.
