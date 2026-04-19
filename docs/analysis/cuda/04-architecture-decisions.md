# 04 — Architectural Decisions

## 4.1 Binding choice

Three viable paths; recommendation in order:

1. **`cudarc` (recommended)** — pure-Rust crate, relatively safe, no custom `build.rs` required for runtime PTX. The already-commented line at [Cargo.toml:39](../../../Cargo.toml#L39) points to this path. Works against the **driver API** and supports loading PTX built offline with `nvcc` or embedded at compile time.
   - Pros: idiomatic Rust, no raw `unsafe` C bindings on the surface, streams/graphs support.
   - Cons: depends on the driver (not the runtime), a few advanced features missing.
2. **`cust` (Rust-CUDA)** — also allows kernels written in Rust via `rustc_codegen_nvvm`. More ambitious and more fragile on macOS/Windows.
3. **`cuda-runtime-sys` + `cublas-sys` + `cc`** — raw bindings, what the OpenSpec proposal cites ([tasks.md](../../../openspec/changes/add-cuda-backend/tasks.md)). Larger `unsafe` surface, but full control.

**Recommendation:** start with **`cudarc`** to cover 80% of the use case (context, stream, allocation, memcpy, PTX launch). Evaluate `cublas-sys` later only if similarity performance demands it.

## 4.2 Kernel compilation

- **For bootstrap:** embed **PTX via `include_str!`**. Compile offline once with `nvcc -ptx`; allows the crate to ship without requiring `nvcc` on end-user machines.
- **In a mature release:** `build.rs` detects `CUDA_PATH` / `CUDA_HOME`, compiles `.cu` → `.ptx` multi-SM (7.0, 7.5, 8.0, 8.6, 8.9, 9.0) and embeds via `OUT_DIR`.

The choice affects distribution: the embedded-PTX path yields a self-contained crate; the `build.rs` path yields smaller binaries but a heavier build-time footprint.

## 4.3 Memory model

The Metal backend ([vector_storage.rs:57-102](../../../src/metal/vector_storage.rs#L57-L102)) uses:

- `StorageModePrivate` (VRAM-only, no CPU mirror).
- `StorageModeShared` staging buffer + blit encoder for uploads.
- Adaptive expansion factor (2.0 → 1.5 → 1.2) with a 1 GB cap.

The CUDA equivalents:

- `cudaMalloc` for dedicated VRAM.
- `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)` as the "staging" copy.
- Reallocation = new `cudaMalloc` + `cudaMemcpyAsync(...Device2Device)` + `cudaFree` of the old pointer.

Using **the same contract** as Metal (`buffer_capacity`, `expand_buffer`, `removed_indices` tracking) keeps cross-backend consistency testing trivial.

## 4.4 Distance computation

| Metric | Recommended path |
|---|---|
| `Euclidean` (L2) | Custom kernel `l2_distance_kernel`, 256 threads/block, shared memory for reduction |
| `Cosine` | `cuBLAS SGEMV` (matrix × vector) + normalization; or fused kernel |
| `DotProduct` | `cuBLAS SGEMV` directly |

Top-K can start on CPU (copy scores back and sort) and later migrate to a `radix_topk` kernel or `cub::DeviceRadixSort` via `cuBLAS`/CUB.

## 4.5 Backend detection

Today [backends/detector.rs:84](../../../src/backends/detector.rs#L84) only inspects env vars. It needs:

- `cudarc::driver::CudaDevice::count()` (or equivalent) to detect the runtime.
- Silent fallback when the driver DLL/so is absent.
- Result caching (detection costs milliseconds).

## 4.6 Error-type design

Extend [error.rs](../../../src/error.rs) with:

```rust
#[error("CUDA driver error: {0}")]
CudaError(String),

#[error("cuBLAS error: {0}")]
CublasError(String),

#[error("Invalid CUDA device id: {0}")]
InvalidDeviceId(i32),
```

Keep the existing `NoDeviceAvailable` and `VramLimitExceeded` variants — they are already backend-agnostic. Prefer `From<cudarc::driver::DriverError>` impls over stringly-typed construction at every call site.

## 4.7 Concurrency model

- **One stream per context.** All operations on a `CudaContext` serialize on that stream, matching Metal's single command queue ([metal/context.rs:35](../../../src/metal/context.rs#L35)).
- `Arc<CudaContext>` shared across storages is safe because the stream is owned by the context; per-operation synchronization is the caller's responsibility.
- Multi-GPU support is out of scope for v1; document the constraint and gate behind `device_id = 0`.
