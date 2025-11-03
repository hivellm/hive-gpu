# Add Memory Pooling

## Why

GPU memory allocation and deallocation are expensive operations. Current implementation allocates/frees buffers for every operation, causing:
- **High latency overhead** (~0.1-1ms per allocation)
- **Fragmentation** over time reducing available memory
- **Reduced throughput** for batch operations
- **Driver overhead** from frequent cudaMalloc/hipMalloc calls

Memory pooling is a proven optimization technique that **reuses pre-allocated buffers**, reducing overhead by 50-80% for workloads with many small allocations.

**Impact**: For batch operations processing 1000s of vectors, pooling can improve throughput from 10K to 50K+ vectors/second.

## What Changes

- Add `GpuMemoryPool` for buffer reuse across all backends
- Support configurable pool size and buffer sizes
- Implement automatic pool warming (pre-allocation)
- Add pool statistics for monitoring
- Thread-safe pool with concurrent access
- Per-backend integration (Metal, CUDA, ROCm)
- Automatic fallback to direct allocation if pool exhausted
- Pool cleanup on context drop

**Breaking Changes**: None (optional opt-in feature)

## Impact

**Affected specs:**
- New: `memory-pool` - Memory pool specification
- `gpu-context` - Add optional pool integration
- `metal` - Integrate pool into MetalNativeContext
- `cuda` - Integrate pool into CudaContext
- `rocm` - Integrate pool into RocmContext

**Affected code:**
- NEW: `src/pool/` - Memory pool implementation
  - `memory_pool.rs` - Generic pool
  - `mod.rs` - Module exports
- UPDATE: `src/metal/context.rs` - Optional pool usage
- UPDATE: `src/cuda/context.rs` - Optional pool usage
- UPDATE: `src/rocm/context.rs` - Optional pool usage
- NEW: `tests/pool_tests.rs` - Pool tests
- `Cargo.toml` - Add `memory-pooling` feature flag (optional)

**Benefits:**
- **50-80% reduction** in allocation overhead
- **Higher throughput** for batch operations
- **Reduced fragmentation**
- **Better memory utilization**
- **Optional** (no impact if not used)

**Timeline**: 3-5 days implementation + testing
**Priority**: ⚡ MEDIUM (optimization after core backends)
**Dependencies**: All backends stable (Phase 3.2 complete)

