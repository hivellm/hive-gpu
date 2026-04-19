# 08 — Next Immediate Steps

1. Uncomment `cudarc` in [Cargo.toml:39](../../../Cargo.toml#L39) and run `cargo check --features cuda`.
2. Remove `#![allow(warnings)]` from [src/lib.rs:6](../../../src/lib.rs#L6).
3. Create branch `feat/cuda-backend` and start [Phase 1](05-implementation-plan.md#phase-1--infrastructure-23-days).
4. Secure access to a real NVIDIA GPU (or cloud instance) for smoke tests before Phase 2.

## Open decisions to resolve before coding

- **Binding choice:** confirm `cudarc` (default recommendation) or opt for `cuda-runtime-sys`. Needs sign-off from the maintainers.
- **Kernel strategy:** embedded PTX vs. `build.rs` compile-on-host. Affects CI workflow design.
- **Minimum compute capability:** the OpenSpec proposal mandates sm_70+; confirm whether sm_60 (Pascal) support is required for any deployed server.
- **Windows support commitment:** CI coverage on Windows is optional but affects how `build.rs` detects `CUDA_PATH`.

## Handoff checklist for the next engineer

- [ ] Read [02-current-state.md](02-current-state.md) to understand what exists.
- [ ] Read [04-architecture-decisions.md](04-architecture-decisions.md) to understand the chosen path.
- [ ] Read [05-implementation-plan.md](05-implementation-plan.md) for the phase breakdown.
- [ ] Read [06-risks-and-mitigations.md](06-risks-and-mitigations.md) before wiring `unsafe` code.
- [ ] Confirm a GPU is reachable (`cudarc::driver::CudaDevice::new(0)` returns `Ok`) before Phase 2.
- [ ] Open a tracking issue that mirrors the task list in [openspec/changes/add-cuda-backend/tasks.md](../../../openspec/changes/add-cuda-backend/tasks.md).

## Escalation triggers

Pause and re-evaluate this plan if any of the following is true at the end of Phase 2:

- `cudarc` does not expose a stable API for stream + memcpy at the version pinned in `Cargo.toml`.
- Numerical divergence from Metal exceeds `1e-3` even on trivial kernels (hints at a deeper algorithmic mismatch).
- The embedded PTX exceeds 2 MB after compressing (makes the crate too heavy for `crates.io`).

In any of those cases, revisit [04-architecture-decisions.md](04-architecture-decisions.md) before continuing.
