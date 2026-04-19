# 10 — Next Immediate Steps

1. **Confirm AMD hardware availability** for development / testing (gfx1030 minimum; gfx90a ideal). Without it, the implementation is speculative.
2. **Wait for CUDA Phase 3 to land** (real vector storage) to establish the pattern ROCm will mirror.
3. Open branch `feat/rocm-backend` only when the two conditions above are satisfied.
4. Decide whether the `hip-runtime-sys` crate covers today's needs or whether `bindgen` in `build.rs` is required — a decision that shapes the first ~8h of work.

## Handoff checklist for the next engineer

- [ ] Read [02-current-state.md](02-current-state.md) to understand what exists (it is close to nothing).
- [ ] Read [03-hardware-targets.md](03-hardware-targets.md) to understand wavefront variability.
- [ ] Read [05-architecture-decisions.md](05-architecture-decisions.md) to understand the chosen path.
- [ ] Read [06-implementation-plan.md](06-implementation-plan.md) for the phased breakdown.
- [ ] Read [07-risks-and-mitigations.md](07-risks-and-mitigations.md) before wiring any `unsafe` code.
- [ ] Read [../cuda/](../cuda/) in full — the ROCm backend is a structural mirror of CUDA.
- [ ] Confirm a ROCm-capable host is reachable (`rocm-smi` lists at least one GPU) before Phase 2.

## Open decisions to resolve before coding

- **Binding crate vs. in-tree `bindgen`** — owner input needed.
- **Windows support commitment** — experimental or first-class?
- **Minimum ROCm version** — 5.6 (first with full HIP SDK on Windows) or 6.0 (official RDNA3 GA)?
- **Which AMD GPU is the reference benchmark card** — this affects the numbers quoted in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md).

## Escalation triggers

Pause and re-evaluate this plan if any of the following is true at the end of Phase 2:

- `bindgen` fails to produce usable bindings for the installed ROCm version.
- Wavefront-dependent reductions cannot be expressed generically and require per-gfx kernel variants.
- Numerical divergence from Metal exceeds `1e-3` even on trivial kernels (hints at a deeper algorithmic mismatch).
- `hipcc` build times exceed 2 minutes per kernel on the CI runner (makes iteration unproductive).

In any of those cases, revisit [05-architecture-decisions.md](05-architecture-decisions.md) before continuing.

## Once done

When ROCm reaches "production-ready" per [07-risks-and-mitigations.md](07-risks-and-mitigations.md):

- Announce in [CHANGELOG.md](../../../CHANGELOG.md) with supported gfx list.
- Update [README.md](../../../README.md) backend matrix and coverage claims (Metal + CUDA + ROCm ≈ 90% market coverage).
- Merge the OpenSpec change at [openspec/changes/add-rocm-backend/](../../../openspec/changes/add-rocm-backend/) into the main specs tree.
- Record benchmark numbers in [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) for at least MI210 + RX 7900 XTX.
