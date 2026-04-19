# 05 — Architectural Decisions

## 5.1 Binding choice

There is no mature `cudarc` equivalent for HIP in Rust. Options:

1. **`hip-sys` + `rocblas-sys`** (raw bindings, the path the OpenSpec proposal cites) — larger `unsafe` surface, but coverage is guaranteed. Recommended baseline.
2. **Hand-written bindings via `bindgen`** — avoids third-party crates that may be stale; take only what is needed.
3. **`hip-runtime-sys`** — exists on crates.io but with irregular maintenance. Evaluate before adopting.

**Recommendation:** combine (2) + (1). Generate bindings with `bindgen` pointing at `$ROCM_PATH/include/hip/` inside `build.rs` for a focused subset (~30 functions) and use `rocblas-sys` for BLAS. Isolate both in an internal `ffi` module so the rest of the crate never sees raw pointers.

This hybrid keeps the surface small, future-proofs against upstream crate churn, and still benefits from `rocblas-sys`'s coverage of BLAS routines that would be tedious to hand-bind.

## 5.2 Kernel compilation

Only realistic option: **`hipcc`** with multi-target:

```bash
hipcc --offload-arch=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100 \
      -c src/rocm/kernels.hip -o kernels.o
```

The `build.rs` must:

1. Detect `ROCM_PATH` / `ROCM_HOME` with `/opt/rocm` fallback.
2. Invoke `hipcc` with the list above.
3. Link `libamdhip64` + `librocblas`.
4. `cargo:rerun-if-changed=src/rocm/*.hip`.

### Bootstrap alternative

Begin with **HIP source JIT** (compile on first run via `hipModuleLoadData`). Slower cold start but removes the `hipcc` build-time dependency. Useful while the `build.rs` is under development, or as a fallback for users without ROCm dev tools installed.

## 5.3 Memory model

Same contract as Metal ([vector_storage.rs:50-102](../../../src/metal/vector_storage.rs#L50)) and the CUDA plan:

| Metal operation | HIP equivalent |
|---|---|
| `MTLResourceOptions::StorageModePrivate` | `hipMalloc(&ptr, size)` |
| `StorageModeShared` staging | Host pinned via `hipHostMalloc(..., hipHostMallocDefault)` |
| `blit_encoder.copyFromBuffer` | `hipMemcpyAsync(..., hipMemcpyHostToDevice, stream)` |
| `command_buffer.waitUntilCompleted()` | `hipStreamSynchronize(stream)` |
| `expand_buffer` (reallocation) | New `hipMalloc` → `hipMemcpyAsync` D2D → `hipFree` of old |

## 5.4 Distance computation

| Metric | Recommended HIP path |
|---|---|
| `Euclidean` (L2) | Custom kernel `hip_l2_distance_kernel`, block = 256 (consumer) / multiple of 64 (datacenter), LDS for reduction |
| `Cosine` | `rocBLAS rocblas_sgemv` + separate normalization kernel |
| `DotProduct` | `rocblas_sgemv` directly |

### AMD-specific optimizations

- **Wavefront-aware reductions:** use `__shfl_xor_sync` with a mask aware of the wavefront size.
- **LDS bank conflicts:** AMD has 32 banks of 4B on RDNA; ensure a stride that does not collide.
- **Memory coalescing:** tile 64 threads × 1 wave on datacenter; 32 × 2 on RDNA.
- **Dual-issue VOPD on RDNA3:** fused instructions — the ROCm compiler usually captures them, but verify with `rocprof`.

## 5.5 Runtime detection

```rust
// src/backends/detector.rs
fn is_rocm_available() -> bool {
    // 1. Try hipGetDeviceCount via lazy dlopen
    // 2. Fallback: check /opt/rocm/bin/rocm-smi or ROCM_PATH
}
```

Target platforms: **Linux** (official), **Windows** (HIP SDK since 2023, less tested). **Never** on macOS (ROCm dropped macOS support).

## 5.6 Error-type design

Extend [error.rs](../../../src/error.rs) with:

```rust
#[error("HIP error: {0}")]
HipError(String),

#[error("rocBLAS error: {0}")]
RocblasError(String),

#[error("ROCm error: {0}")]
RocmError(String),
```

Favor `From<hip_sys::hipError_t>` and `From<rocblas_sys::rocblas_status>` impls over stringly-typed construction at every call site.

## 5.7 Backend selection

Multi-vendor systems (NVIDIA + AMD, rare but real in HPC workstations) need explicit selection, not "first one found". Proposal:

- Default priority: `Metal > CUDA > ROCm > CPU`.
- Override via env var `HIVE_GPU_BACKEND=cuda|rocm|metal|cpu`.
- Or an API `select_best_backend_with(BackendPreference::Rocm)` that fails if the preferred backend is absent.

Document this in the [API reference](../../API_REFERENCE.md) once implemented.
