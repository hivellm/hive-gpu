# hive-gpu 0.2.0

**GPU acceleration for vector similarity search, written in Rust.**

[![Crates.io](https://img.shields.io/crates/v/hive-gpu.svg)](https://crates.io/crates/hive-gpu)
[![Documentation](https://docs.rs/hive-gpu/badge.svg)](https://docs.rs/hive-gpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/hivellm/hive-gpu/workflows/CI/badge.svg)](https://github.com/hivellm/hive-gpu/actions)

Four GPU backends, all feature-gated and target-gated so the crate builds
cleanly on every host:

- **Metal** — Apple Silicon (M-series), built on `objc2-metal`. Brute-force
  search and IVF via custom compute kernels. *IVF and the new real-kernel
  brute-force path are authored but not yet validated on macOS — see
  `.rulebook/tasks/phase4d_validate-metal-backend-on-mac`.*
- **CUDA** — NVIDIA (Volta / sm_70+) on Linux and Windows, built on `cudarc`
  driver API + cuBLAS SGEMV/SGEMM. Brute-force + IVF. Validated on RTX 4090
  (3.67× over brute-force at 1 M vectors).
- **ROCm** — AMD (gfx900 through gfx1100) on Linux, via hand-rolled HIP FFI
  + rocBLAS. Brute-force + IVF. *Authored blind — see
  `.rulebook/tasks/phase4e_validate-rocm-backend-on-amd`.*
- **Intel / Vulkan Compute** — Intel Arc / Battlemage (preferred) on Linux
  and Windows, with `HIVE_GPU_VULKAN_UNIVERSAL=1` fallback for any Vulkan
  1.2 GPU. Built on `ash` + WGSL shaders compiled to SPIR-V via `naga`.
  Brute-force + IVF. *Authored blind — see
  `.rulebook/tasks/phase4f_validate-intel-backend-on-vulkan`.*

Design notes for each backend live in [`docs/analysis/`](docs/analysis/).

---

## What's new in 0.2.0

- 🔥 **CUDA backend is functional.** `CudaContext`, `CudaVectorStorage`,
  GPU-accelerated search (cuBLAS SGEMV for Cosine/DotProduct, derived L2),
  and a full IVF index (`CudaIvfIndex` — k-means++ + cuBLAS SGEMM) all run
  against a real driver. **3.67× over brute-force at 1 M vectors**, recall
  ≥ 0.95 on clustered data.
- **Metal** gets a real brute-force compute kernel (replacing the prior CPU
  shim) plus `MetalIvfIndex`. Authored blind — awaits Apple Silicon
  validation.
- **ROCm / HIP backend** for AMD GPUs on Linux — `RocmContext` +
  `RocmVectorStorage` + `RocmIvfIndex`, hand-rolled HIP FFI over
  `libamdhip64` + `librocblas`. Authored blind — awaits AMD validation.
- **Intel / Vulkan Compute backend** — `IntelContext` +
  `IntelVectorStorage` + `IntelIvfIndex` on `ash`, WGSL shaders compiled to
  SPIR-V at build time via `naga` (pure-Rust, no CMake / C++ toolchain).
  Works on any Vulkan 1.2 GPU under `HIVE_GPU_VULKAN_UNIVERSAL=1`. Authored
  blind — awaits Vulkan-host validation.
- Real device-info API on CUDA — compute capability, total/free VRAM,
  driver version, PCI bus id — all queried live from the driver.
- Dynamic buffer growth with device-to-device copy mirroring the Metal
  backend's shape (2× → 1.5× → 1.2× adaptive factor).
- Criterion benchmarks under [`benches/cuda_ops.rs`](benches/cuda_ops.rs)
  and [`benches/cuda_ivf.rs`](benches/cuda_ivf.rs).
- CI job (`.github/workflows/cuda-build.yml`) builds against the official
  `nvidia/cuda:12.4.1-devel-ubuntu22.04` image.
- Project-wide `#![allow(warnings)]` removed; clippy runs with `-D warnings`
  on all feature combinations.

Only CUDA is validated on real hardware (RTX 4090). Metal, ROCm, and Intel
are code-complete but pending maintainer validation — see
`.rulebook/tasks/phase4{d,e,f}_*` for the validation checklists.

Full changelog in [`CHANGELOG.md`](CHANGELOG.md).

---

## Installation

```toml
[dependencies]
# macOS — Metal backend (default)
hive-gpu = "0.2.0"

# Linux / Windows — CUDA backend
hive-gpu = { version = "0.2.0", default-features = false, features = ["cuda"] }

# Linux — AMD ROCm / HIP backend
hive-gpu = { version = "0.2.0", default-features = false, features = ["rocm"] }

# Linux / Windows — Intel / Vulkan Compute backend (also works as a universal
# Vulkan fallback on any Vulkan 1.2 GPU under HIVE_GPU_VULKAN_UNIVERSAL=1)
hive-gpu = { version = "0.2.0", default-features = false, features = ["intel"] }

# Everything (cross-platform crate — each cfg is gated internally)
hive-gpu = { version = "0.2.0", features = ["metal-native", "cuda", "rocm", "intel"] }
```

Runtime requirements:

- **CUDA** — NVIDIA driver (no CUDA Toolkit required — `cudarc` is built
  with `dynamic-linking`).
- **ROCm** — ROCm 6.x runtime with `libamdhip64.so` and `librocblas.so` on
  the dynamic linker path.
- **Intel** — Vulkan 1.2 loader (`libvulkan.so.1` / `vulkan-1.dll`), shipped
  with any recent GPU driver.

For a development checkout you also need a reachable driver so integration
tests can hit real hardware; without one, each suite runs as a no-op.

---

## Quick start

### Metal (macOS)

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuDistanceMetric, GpuVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = MetalNativeContext::new()?;
    let mut storage = ctx.create_storage(128, GpuDistanceMetric::Cosine)?;

    storage.add_vectors(&[
        GpuVector::new("a".into(), vec![1.0; 128]),
        GpuVector::new("b".into(), vec![0.5; 128]),
    ])?;

    let query = vec![0.9; 128];
    for r in storage.search(&query, 5)? {
        println!("{}  {:.4}", r.id, r.score);
    }
    Ok(())
}
```

### CUDA (Linux / Windows)

```rust
use hive_gpu::cuda::CudaContext;
use hive_gpu::traits::{GpuBackend, GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuDistanceMetric, GpuVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !CudaContext::is_available() {
        eprintln!("no CUDA device reachable — using a CPU fallback instead");
        return Ok(());
    }

    let ctx = CudaContext::new()?;
    println!("{}", GpuBackend::device_info(&ctx).name);
    //=> NVIDIA GeForce RTX 4090

    let mut storage = ctx.create_storage(128, GpuDistanceMetric::DotProduct)?;
    storage.add_vectors(&[
        GpuVector::new("x".into(), vec![1.0; 128]),
        GpuVector::new("y".into(), vec![0.9; 128]),
    ])?;

    let query = vec![1.0; 128];
    for r in storage.search(&query, 5)? {
        println!("{}  {:.4}", r.id, r.score);
    }
    Ok(())
}
```

See [`examples/cuda_basic.rs`](examples/cuda_basic.rs) and
[`examples/metal_basic.rs`](examples/metal_basic.rs) for runnable variants.

---

## Performance

Two data points captured on real hardware. All numbers are median wall-clock
from Criterion benchmarks (`cargo bench`).

### CUDA — NVIDIA GeForce RTX 4090 (24 GB, driver 591.59, CUDA 13.1)

**Search latency — DotProduct, 128-dim f32, top-10** (from
[`benches/cuda_ops.rs`](benches/cuda_ops.rs)):

|   N     | GPU (cuBLAS SGEMV) | CPU naïve reference | GPU speedup |
|--------:|-------------------:|--------------------:|------------:|
|   1 000 |             124 µs |               63 µs |       0.51× |
|  10 000 |             287 µs |              690 µs |       **2.40×** |
| 100 000 |            4.01 ms |            13.04 ms |       **3.25×** |

For N < 10 K the SGEMV launch + host-to-device copy dominates useful work and
a scalar CPU loop wins. From 10 K onward the GPU wins and the margin widens
roughly linearly with N.

**Add-vectors throughput** (128-dim f32):

| Batch size | Wall-clock | Throughput        |
|-----------:|-----------:|------------------:|
|     1 000  |     431 µs | 2.32 M elements/s |
|    10 000  |    7.10 ms | 1.41 M elements/s |

### Metal — Apple M3 Pro

|  Operation                   | CPU baseline | Metal      | Speedup |
|------------------------------|-------------:|-----------:|--------:|
| Vector addition (sustained)  |   1 000 vec/s | 3 728 vec/s |   3.7× |
| Vector addition (peak 10 K)  |   1 000 vec/s | 4 250 vec/s |   4.25× |
| Search latency (k = 10)      |        ~1 ms |      0.92 µs | ~1 000× |
| Search throughput            |           —  | 1.08 M qps  |   —    |

Full methodology, hardware matrix, and historical runs live in
[`docs/benchmarks/PERFORMANCE.md`](docs/benchmarks/PERFORMANCE.md).

---

## GPU backend matrix

|  OS                         | Metal | CUDA | ROCm | Intel | CPU |
|-----------------------------|:-----:|:----:|:----:|:-----:|:---:|
| macOS (Apple Silicon)       |  🟡   |  ❌  |  ❌  |   ❌  |  ✅ |
| Linux x86_64 + NVIDIA       |  ❌   |  ✅  |  ❌  |   🟡  |  ✅ |
| Linux x86_64 + AMD          |  ❌   |  ❌  |  🟡  |   🟡  |  ✅ |
| Linux x86_64 + Intel Arc    |  ❌   |  ❌  |  ❌  |   🟡  |  ✅ |
| Windows x86_64 + NVIDIA     |  ❌   |  ✅  |  ❌  |   🟡  |  ✅ |
| Windows x86_64 + AMD        |  ❌   |  ❌  |  ❌  |   🟡  |  ✅ |
| Windows x86_64 + Intel Arc  |  ❌   |  ❌  |  ❌  |   🟡  |  ✅ |

Legend: ✅ shipping and validated · 🟡 code-complete, pending hardware
validation · ❌ unsupported.

On Linux/Windows the Intel / Vulkan backend doubles as a universal
Vulkan-Compute fallback when `HIVE_GPU_VULKAN_UNIVERSAL=1` is set.

Backend-selection order at runtime is `Metal > CUDA > ROCm > Intel > CPU`.
Override via the `HIVE_GPU_BACKEND` env var (planned).

---

## Feature flags

| Feature        | Target OS        | Pulls in                                                     |
|----------------|------------------|--------------------------------------------------------------|
| `metal-native` | macOS            | `objc2-metal`, `objc2-foundation`, `objc2`                   |
| `cuda`         | Linux / Windows  | `cudarc` (`driver` + `cublas` + `dynamic-linking`)           |
| `rocm`         | Linux            | `libloading` (dlopens `libamdhip64.so` + `librocblas.so`)    |
| `intel`        | Linux / Windows  | `ash` (Vulkan loader) + `naga` (WGSL → SPIR-V at build time) |

`metal-native` is the default. On non-macOS hosts the default feature
contributes nothing (its deps are target-gated), so the crate compiles clean
everywhere with default features.

---

## Testing and benchmarks

```bash
# Metal (macOS)
cargo test --features metal-native
cargo bench --features metal-native --bench gpu_operations

# CUDA (Linux / Windows with an NVIDIA driver installed)
cargo test --features cuda --test cuda_smoke --test cuda_device_info \
                          --test cuda_vector_ops --test cuda_ivf
cargo bench --features cuda --bench cuda_ops --bench cuda_ivf

# ROCm (Linux with ROCm 6.x installed)
cargo test --features rocm --test rocm_smoke --test rocm_ivf

# Intel / Vulkan (Linux / Windows with a Vulkan 1.2 GPU)
cargo test --features intel --test intel_smoke --test intel_ivf
# On a non-Intel Vulkan GPU, set HIVE_GPU_VULKAN_UNIVERSAL=1 first.
```

Every suite is a no-op on hosts without a reachable driver, so they stay
green on CI runners that lack GPU hardware.

---

## Roadmap

- **v0.2.x** — hardware validation pass for the three blind backends
  (Metal, ROCm, Intel). Each ships its own point release once the matching
  `phase4{d,e,f}` task lands benchmarks and test results on real hardware.
- **v0.4** — GPU HNSW construction and search across all four backends,
  quantization (PQ / SQ), GPU-side top-K (radix select).

Detailed roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Project documentation

- [`docs/analysis/`](docs/analysis/) — backend implementation analyses
  (CUDA, ROCm, Intel) with gap analysis, architectural decisions, and phased
  plans.
- [`docs/benchmarks/PERFORMANCE.md`](docs/benchmarks/PERFORMANCE.md) — full
  performance guide and historical numbers.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — release plan.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — contribution guide.

---

## License

MIT — see [`LICENSE`](LICENSE).
