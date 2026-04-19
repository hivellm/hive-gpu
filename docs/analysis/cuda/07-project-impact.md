# 07 — Impact on the Rest of the Project

## 7.1 Files that change

- [src/lib.rs](../../../src/lib.rs): keep the feature gate as is (`#[cfg(feature = "cuda")]`). Remove `#![allow(warnings)]` at [lib.rs:6](../../../src/lib.rs#L6).
- [src/traits.rs](../../../src/traits.rs): **no changes required** — the `GpuBackend` / `GpuContext` / `GpuVectorStorage` traits are already sufficient.
- [src/types.rs](../../../src/types.rs): `GpuDeviceInfo` already has `compute_capability: Option<String>` — compatible with CUDA strings like `"8.9"`.
- [src/backends/detector.rs](../../../src/backends/detector.rs): `GpuBackendType::Cuda` already exists; update `is_cuda_available()` and prioritize CUDA before CPU on Linux/Windows.
- [Cargo.toml](../../../Cargo.toml): enable the `cudarc` dependency behind the `cuda` feature; add `build-dependencies = ["cc"]` if the `build.rs` path is chosen.

## 7.2 Warning suppression

[src/lib.rs:6](../../../src/lib.rs#L6) currently reads `#![allow(warnings)]`. This is a project-wide silencing that actively hides issues in the CUDA stubs (unused fields, dead code, missing docs). **Removing it is a prerequisite for the CUDA work** — otherwise the team cannot rely on `cargo check` or clippy to spot regressions during the migration.

Recommended sequence:

1. Remove the attribute.
2. Run `cargo check --all-features` and record the baseline warning count.
3. Add `#[allow(...)]` narrowly on the known-stub modules until their implementation phase lands.
4. Re-enable `clippy = "deny"` in CI at the end of Phase 2.

## 7.3 Tests

- [tests/gpu_detection_tests.rs](../../../tests/gpu_detection_tests.rs) and [tests/device_info_tests.rs](../../../tests/device_info_tests.rs) already mention CUDA in documentation comments and assertions against the string `"CUDA"`. Adding the CUDA path must not break those tests — preserve the existing expectations when `cfg(feature = "cuda")` is off.
- New suites to create:
  - `tests/cuda_device_info.rs`
  - `tests/cuda_vector_ops.rs`
  - `tests/cuda_stress.rs`
  - `tests/cross_backend_consistency.rs` (Metal × CUDA)

## 7.4 Documentation

- [README.md](../../../README.md) has a section labeled "CUDA Backend (Linux/Windows)" that promises usage which does not work. Update once Phase 3 lands; add a "Supported compute capabilities" matrix.
- [docs/API_REFERENCE.md](../../../docs/API_REFERENCE.md) should gain a CUDA subsection mirroring the Metal one.
- [docs/DEVELOPMENT.md](../../../docs/DEVELOPMENT.md) needs a "Building with CUDA" section: required driver version, env vars, how to run the suite with a local GPU.
- [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) should add a CUDA column to its benchmark tables.

## 7.5 Dependency graph

Adding `cudarc` pulls:

- `cudarc` itself
- `libloading` (for runtime driver discovery)
- `half` (already a direct dependency — [Cargo.toml:22](../../../Cargo.toml#L22) — so no duplication)

No other transitive additions of concern. Audit with `cargo tree --features cuda` before merging Phase 1.

## 7.6 Release sequencing

Suggested versioning:

- `0.1.11` → Phase 1 + 2 (infrastructure + context, feature-gated, experimental).
- `0.2.0` → Phases 3 + 4 complete. Announces CUDA support in CHANGELOG; still "beta".
- `0.2.1` → Phases 5 + 6 (HNSW + CI). CUDA declared production-ready.

All CUDA work is feature-gated and additive, so there are no breaking changes for existing Metal users across 0.1.10 → 0.2.1.
