# 07 — Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Vulkan Compute is 20–40% slower than native on NVIDIA/AMD | High | Low (not the primary target) | Document honestly; do not claim parity with CUDA on NVIDIA |
| No BLAS equivalent reachable from Rust on Intel | High | Medium | Hand-write SGEMV kernels; budget extra days in Phase 4 |
| `shaderc-sys` build complexity (requires Vulkan SDK on build host) | Medium | Medium | Ship prebuilt SPIR-V in the crate as a fallback; allow the `shaderc` recompile as an opt-in |
| `rust-gpu` still 0.x — risk if chosen for kernels | Medium | Medium | Default to GLSL + `shaderc`; revisit `rust-gpu` only after kernels are stable |
| Intel discrete GPU market share remains <1% | High | Strategic | Accept that this is a differentiator investment, not a coverage investment |
| Ponte Vecchio EOL invalidates early testing on that hardware | Certain | Low | Avoid Ponte Vecchio as a benchmark target; prioritize Battlemage + Arc Pro |
| Crescent Island delayed beyond Q3 2026 | Medium | Low | Backend works on Arc/Battlemage already; datacenter upside is a bonus |
| Driver skew across `i915` vs `xe` Linux kernel modules | Medium | Medium | CI uses Ubuntu 24.04 with `xe` driver; document both paths |
| Vulkan validation layer warnings drown the signal | Low | Low | Enable layers only in `debug_assertions`; scrub noise before shipping |
| Descriptor-set management memory leak | Medium | High | Use `vk-mem` (Vulkan Memory Allocator) + audit with `VK_EXT_validation_features` |
| Users confuse "Intel backend" with "runs on Intel CPUs" | Medium | Low | README clarifies: GPU backend only; CPU users get the CPU fallback |
| Licensing of any embedded SPIR-V derived from Khronos samples | Low | Low | Write kernels from scratch; avoid Khronos sample code unless Apache-2.0-compatible |
| `ash` API churn on minor version bumps | Low | Low | Pin to a specific minor version; upgrade behind a test suite |

## Watch list beyond v1

- **XMX matrix engines** (Battlemage, Arc Pro, Xe3P) — accessible via `VK_KHR_cooperative_matrix`. Significant perf win for similarity on large dimensions. Plan for v0.4.
- **oneMKL via a C++ shim crate** — if someone contributes a maintained wrapper, BLAS-accelerated Cosine/Dot becomes possible. Low priority unless performance becomes a blocker.
- **Level Zero direct** — if the Vulkan path shows a consistent performance floor we cannot break, consider writing a parallel L0 path for Intel hardware specifically. High cost, dubious payoff.
- **Intel NPU (Neural Processing Unit)** in Meteor Lake / Lunar Lake — a separate accelerator, not the GPU. Out of scope.

## Exit criteria for "Intel is production-ready"

1. `tests/cross_backend_consistency.rs` passes with Metal + CUDA + ROCm + Intel on every CI run.
2. Numerical divergence ≤ `1e-4` over ≥10k random queries on Arc B580 **and** Arc Pro B70.
3. No `unsafe` block without a `// SAFETY:` comment.
4. `cargo clippy --features intel -- -D warnings` green.
5. Vulkan validation layers produce zero error/warning messages during the test suite on a debug build.
6. `cargo test --features intel` runs the full suite on at least one self-hosted runner with an Arc card.
7. Documented performance numbers for at least Arc B580 + Arc Pro B70 in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md), with an honest comparison against CUDA/ROCm on comparable silicon (expect 40–60% of native).
8. `docs/guides/INTEL_SETUP.md` reviewed by at least one external user.

## Abort / pivot conditions

Stop the work and re-evaluate if any of the following occur during Phase 2–4:

- `ash` cannot reliably enumerate Intel physical devices on Windows 11 (driver-loading anomaly).
- GLSL → SPIR-V compilation via `shaderc` requires a toolchain heavier than the project is willing to inherit in `build.rs`.
- Benchmark numbers on Arc B580 are worse than the CPU fallback for any metric (would imply a fundamental kernel bug, not an optimization target).
- No team member can commit to maintaining the backend after v1 — shipping an unmaintained backend is worse than not shipping.
