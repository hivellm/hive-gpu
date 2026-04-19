# 08 — Impact on the Rest of the Project

## 8.1 Files that change

- [src/lib.rs](../../../src/lib.rs): add `#[cfg(feature = "rocm")] pub mod rocm;`. **Remove** `#![allow(warnings)]` at [lib.rs:6](../../../src/lib.rs#L6) before accepting the new backend.
- [src/traits.rs](../../../src/traits.rs): **no changes** — the traits are backend-agnostic.
- [src/types.rs](../../../src/types.rs): already foresees ROCm; confirm `compute_capability` accepts `"gfx*"` strings.
- [src/backends/detector.rs](../../../src/backends/detector.rs): add `Rocm` to the enum + function `is_rocm_available()` + prioritization.
- [Cargo.toml](../../../Cargo.toml): new `rocm` feature + `[target.'cfg(any(target_os = "linux", target_os = "windows"))'.dependencies]`.
- [tests/gpu_detection_tests.rs](../../../tests/gpu_detection_tests.rs): include a ROCm path with a graceful skip when unavailable.

## 8.2 Collision with CUDA

On systems with NVIDIA + AMD simultaneously (rare but real — HPC workstations), the selection logic must be explicit, not "first one found". Proposal:

- Env var `HIVE_GPU_BACKEND=cuda|rocm|metal|cpu` for override.
- API `select_best_backend_with(preferred: BackendPreference)` with a documented fallback chain.

This avoids the "Linux host auto-picks the AMD GPU when the user expected the NVIDIA one" class of bug.

## 8.3 Warning suppression

Same argument as CUDA: remove `#![allow(warnings)]` at [src/lib.rs:6](../../../src/lib.rs#L6) before merging a new backend. Otherwise the team cannot rely on `cargo check` / clippy to flag regressions in the new `unsafe` code.

## 8.4 Tests

- [tests/device_info_tests.rs](../../../tests/device_info_tests.rs) and [tests/gpu_detection_tests.rs](../../../tests/gpu_detection_tests.rs) already reference ROCm in strings — do not regress those expectations.
- New suites:
  - `tests/rocm_device_info.rs`
  - `tests/rocm_vector_ops.rs`
  - `tests/rocm_stress.rs`
  - `tests/cross_backend_consistency.rs` (Metal × CUDA × ROCm)

## 8.5 Documentation

- [README.md](../../../README.md) — update the backend matrix; add "Supported AMD architectures" subsection.
- [docs/API_REFERENCE.md](../../../docs/API_REFERENCE.md) — mirror the Metal/CUDA subsections.
- [docs/DEVELOPMENT.md](../../../docs/DEVELOPMENT.md) — add "Building with ROCm": required driver, env vars, how to run the suite on a local AMD GPU.
- [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md) — new ROCm column in benchmark tables.
- New guide `docs/guides/ROCM_SETUP.md` — Ubuntu / RHEL install, `rocm-smi`, `rocprof` basics.

## 8.6 Dependency graph

Adding `hip-sys` + `rocblas-sys` + `bindgen` pulls a moderate set of transitive crates. Audit with `cargo tree --features rocm` before merging Phase 1. Mind:

- `bindgen` adds `clang-sys` → requires `libclang` on the build host.
- `rocblas-sys` requires `librocblas.so` at link time → `build.rs` must detect ROCm install.
- None of the additions conflict with existing dependencies like `half`, `tracing`, or `serde`.

## 8.7 Release sequencing

Suggested versioning, assuming CUDA is already merged:

- `0.2.2` → Phases 1 + 2 (infrastructure + context, feature-gated, experimental).
- `0.2.3` → Phases 3 + 4. Announces ROCm support in CHANGELOG; still "beta".
- `0.3.0` → Phases 5 + 6 (consistency + CI). ROCm declared production-ready.

All ROCm work is feature-gated and additive; no breaking changes for existing Metal or CUDA users.

## 8.8 Cross-platform matrix after ROCm ships

| OS | Metal | CUDA | ROCm | CPU |
|---|---|---|---|---|
| macOS (Apple Silicon) | ✅ Primary | ❌ | ❌ | ✅ |
| macOS (Intel + AMD eGPU) | 🟡 Legacy | ❌ | ❌ (no macOS ROCm) | ✅ |
| Linux x86_64 | ❌ | ✅ | ✅ | ✅ |
| Windows x86_64 | ❌ | ✅ | 🟡 Experimental | ✅ |
| Linux ARM64 | ❌ | 🟡 (Jetson) | ❌ | ✅ |

This matrix belongs in [README.md](../../../README.md) after ROCm ships.
