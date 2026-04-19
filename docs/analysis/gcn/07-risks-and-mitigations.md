# 07 — Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Outdated / incomplete Rust HIP bindings | High | High | Generate with `bindgen` from the installed header; do not depend on third-party crates |
| Wavefront 32 vs 64 breaking kernels | High | Medium | Use runtime `warpSize`; test on gfx1030 (32) **and** gfx90a (64) before merge |
| `hipcc` unavailable in CI | High | Medium | Official `rocm/dev-ubuntu:*` container; skip tests when `HIP_PATH` unset |
| Numerical divergence rocBLAS ↔ cuBLAS | Medium | Low | `1e-4` tolerance in cross-backend tests |
| Windows ROCm support is immature | Medium | Low | Mark as `experimental` on Windows until gfx1100 SDK stabilizes |
| Lack of AMD hardware for dev / CI | High | High | Cloud: GCP / Azure with MI300 or Hetzner with RX 7900; or contributors with AMD hardware |
| License constraints on `.hip` kernels | Low | Low | Apache 2.0 already covers; validate restrictions in ROCm headers (Apache / MIT) |
| Driver version skew (ROCm 5.x vs 6.x) | Medium | Medium | Pin minimum ROCm 5.6; test against both 5.x and 6.x in nightly |
| `Drop` order races on context shutdown | Medium | High | Strict sequence rocBLAS handle → stream → device; enforce with `impl Drop` and a guard struct |
| Payload map divergence between backends | Medium | Low | Extract shared helper after ROCm lands (not before, to avoid premature abstraction) |

## Watch list beyond v1

- **Matrix cores (MFMA / WMMA):** worth evaluating for large-dim similarity. Out of scope for v1.
- **HSA async copy engines on CDNA:** can overlap uploads with compute. Low priority until benchmark data requires it.
- **Multi-GPU / xGMI fabrics:** significant for MI250X / MI300X rack deployments. Plan for v0.3.

## Exit criteria for "ROCm is production-ready"

1. `tests/cross_backend_consistency.rs` green for Metal × CUDA × ROCm on every CI run.
2. Numerical divergence ≤ `1e-4` over ≥10k random queries, on **both** gfx90a and gfx1030.
3. No `unsafe` block without a `// SAFETY:` comment.
4. `cargo clippy --features rocm -- -D warnings` green.
5. `cargo test --features rocm` runs the full suite on at least one self-hosted AMD runner.
6. Memory leak audit (`compute-sanitizer` or `rocprof --sys-trace` with a simulated leak check) clean on a representative workload.
7. Documented performance numbers on at least MI210 and RX 7900 XTX in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md).
