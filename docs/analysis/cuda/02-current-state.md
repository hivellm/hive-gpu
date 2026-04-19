# 02 — Current State of the Code

## 2.1 Feature flag (Cargo.toml)

[Cargo.toml:39](../../../Cargo.toml#L39):

```toml
cuda = [] # cudarc = { version = "0.13", optional = true }
```

- Feature is declared but has **no real dependencies**. The `cudarc` line is commented out.
- There is no `build.rs` at the repository root — therefore no compilation of `.cu` kernels with `nvcc`.
- No `[target.'cfg(...)'.dependencies]` section for CUDA (Linux/Windows).

## 2.2 Modules under `src/cuda/`

| File | Lines | Real content | Verdict |
|---|---|---|---|
| [mod.rs](../../../src/cuda/mod.rs) | 17 | Re-exports only | OK as facade |
| [context.rs](../../../src/cuda/context.rs) | 149 | `CudaContext` with fixed fields (`device_id=0`, `compute_capability=(7,5)`, `total_memory=1GB`) | Stub — never calls `cudaGetDeviceProperties` |
| [vector_storage.rs](../../../src/cuda/vector_storage.rs) | 108 | `add_vector` only bumps a counter; `search`/`get_vector` return `Err("not implemented yet")` | Total stub |
| [buffer_pool.rs](../../../src/cuda/buffer_pool.rs) | 38 | `get_buffer` returns immediate `Err` | Total stub |
| [hnsw_graph.rs](../../../src/cuda/hnsw_graph.rs) | 55 | `build_graph`/`search` return `Err` | Total stub |
| [vram_monitor.rs](../../../src/cuda/vram_monitor.rs) | 36 | Returns fixed values (`0` used, `1GB` available) | Total stub |
| [helpers.rs](../../../src/cuda/helpers.rs) | 51 | Block-size arithmetic purely on CPU | Useful but isolated |

## 2.3 Hardcoded values of concern

- [context.rs:29](../../../src/cuda/context.rs#L29): compute capability pinned to `(7, 5)`.
- [context.rs:30](../../../src/cuda/context.rs#L30): total VRAM pinned to `1 GB`.
- [context.rs:72](../../../src/cuda/context.rs#L72): `is_available()` unconditionally returns `false`.
- [backends/detector.rs:84](../../../src/backends/detector.rs#L84): detection based solely on env vars (`CUDA_VISIBLE_DEVICES`, `CUDA_HOME`) — does not query the driver.

## 2.4 Example and tests

- [examples/cuda_basic.rs](../../../examples/cuda_basic.rs) exists but **fails at runtime** on `storage.search(...)`, hitting `Err("CUDA search not implemented yet")`.
- No `tests/cuda_*.rs` file in the [tests/](../../../tests/) directory — the 72-test suite is Metal-only.

## 2.5 Kernels

- No `.cu` / `.cuh` / `.ptx` file in the repository.
- Existing shaders ([src/shaders/](../../../src/shaders/)) are Metal (`.metal`) and WGSL (`.wgsl`). There is **no direct reuse plan** — the HNSW algorithmic path in Metal would need to be ported.

## 2.6 Public API surface (for contract continuity)

The Metal backend already implements the traits in [src/traits.rs](../../../src/traits.rs). Any CUDA implementation must satisfy the same contract without requiring changes to the trait definitions, so the integration surface is stable:

- `GpuBackend::device_info() -> GpuDeviceInfo`
- `GpuBackend::supports_operations() -> GpuCapabilities`
- `GpuBackend::memory_stats() -> GpuMemoryStats`
- `GpuContext::create_storage(dimension, metric) -> Result<Box<dyn GpuVectorStorage>>`
- `GpuContext::create_storage_with_config(..., HnswConfig) -> ...`
- `GpuVectorStorage::add_vectors`, `search`, `remove_vectors`, `vector_count`, `dimension`, `get_vector`, `clear`
