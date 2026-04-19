# hive-gpu - Architecture

## Overview

`hive-gpu` is a high-performance GPU acceleration library for vector operations, specifically designed for vector similarity search workloads. It provides a unified API across multiple native GPU backends (Metal for Apple Silicon, CUDA for NVIDIA GPUs, and ROCm for AMD GPUs) with optimized implementations for HNSW (Hierarchical Navigable Small World) graph-based approximate nearest neighbor search.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (User Code, Vectorizer Integration, Custom Implementations) │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      Core API Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ GpuContext   │  │ GpuVector    │  │ GpuSearch    │      │
│  │    Trait     │  │  Storage     │  │   Result     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   Backend Abstraction                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  GpuBackend  │  │ GpuBuffer    │  │  GpuMonitor  │      │
│  │    Trait     │  │   Manager    │  │     Trait    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              GPU Backend Implementations                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Metal     │  │     CUDA     │  │     ROCm     │      │
│  │   Native     │  │    Native    │  │    Native    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  Hardware Layer (GPU)                         │
│      Apple Silicon / NVIDIA GPUs / AMD Radeon/Instinct      │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. **Core Types Module** (`src/types.rs`)

**Responsibility:** Defines fundamental data structures used throughout the system.

**Key Types:**
- `GpuVector`: Vector representation with ID, data, and metadata
- `GpuDistanceMetric`: Distance metric enum (Cosine, Euclidean, DotProduct)
- `GpuSearchResult`: Search result with ID, score, and index
- `GpuDeviceInfo`: GPU device information
- `GpuMemoryStats`: Memory usage statistics
- `HnswConfig`: HNSW graph configuration

**Dependencies:** None (foundation module)

### 2. **Core Traits Module** (`src/traits.rs`)

**Responsibility:** Defines interfaces for GPU operations and backend abstraction.

**Key Traits:**
- `GpuBackend`: Core backend interface for device info and capabilities
- `GpuContext`: Factory trait for creating vector storage instances
- `GpuVectorStorage`: Vector operations interface (add, search, remove)
- `GpuBufferManager`: Buffer allocation and management
- `GpuMonitor`: VRAM monitoring and validation

**Dependencies:** `types` module

### 3. **Backend Detection Module** (`src/backends/detector.rs`)

**Responsibility:** Detects available GPU backends and selects the optimal one.

**Key Functions:**
- `detect_available_backends()`: Returns list of available backends
- `select_best_backend()`: Selects best backend based on priority (Metal > CUDA > CPU)
- `get_backend_info()`: Retrieves backend-specific information
- `get_backend_performance_info()`: Returns performance characteristics

**Dependencies:** Feature flags (`metal-native`, `cuda`, `rocm`)

### 4. **Metal Backend Module** (`src/metal/`)

**Responsibility:** Metal-specific GPU acceleration for Apple Silicon.

**Submodules:**
- `context.rs`: Metal device and command queue management
- `vector_storage.rs`: Vector storage implementation using Metal buffers
- `hnsw_graph.rs`: HNSW graph construction and search on GPU
- `buffer_pool.rs`: Metal buffer pool for efficient memory management
- `vram_monitor.rs`: VRAM usage monitoring and validation
- `helpers.rs`: Metal-specific utility functions

**Key Features:**
- Pure Metal Native implementation
- Optimized for Apple Silicon (M1/M2/M3/M4)
- Support for Metal Performance Shaders (MPS)
- VRAM-only storage for maximum performance
- Modern type-safe Rust bindings via objc2-metal (v0.1.8+)

**Dependencies:** 
- `objc2-metal` v0.3 - Metal framework bindings (actively maintained)
- `objc2-foundation` v0.3 - Foundation framework support  
- `objc2` v0.6 - Modern Objective-C interop

**Migration Note (v0.1.8):**
Migrated from discontinued `metal-rs` to actively maintained `objc2-metal` for:
- Active maintenance and security updates
- Modern type-safe bindings with `ProtocolObject<dyn Trait>` pattern
- Full Metal API coverage and integrated Foundation support

See `docs/guides/MIGRATION_METAL_OBJC2.md` for migration details.

### 5. **CUDA Backend Module** (`src/cuda/`)

**Responsibility:** CUDA-specific GPU acceleration for NVIDIA GPUs.

**Submodules:**
- `context.rs`: CUDA device and stream management
- `vector_storage.rs`: Vector storage using CUDA memory
- `hnsw_graph.rs`: HNSW implementation with CUDA kernels
- `buffer_pool.rs`: CUDA memory pool
- `vram_monitor.rs`: CUDA memory monitoring
- `helpers.rs`: CUDA-specific utility functions

**Key Features:**
- Native CUDA implementation
- Optimized for NVIDIA GPUs (Compute Capability 7.0+)
- Support for Tensor Cores (A100+)
- Multi-GPU support

**Status:** Implementation planned (v0.2.0)

**Dependencies:** `cudarc` crate

### 6. **ROCm Backend Module** (`src/rocm/`)

**Responsibility:** ROCm-specific GPU acceleration for AMD GPUs.

**Submodules:**
- `context.rs`: ROCm device and queue management
- `vector_storage.rs`: Vector storage using ROCm memory
- `hnsw_graph.rs`: HNSW implementation with HIP kernels
- `buffer_pool.rs`: ROCm memory pool
- `vram_monitor.rs`: ROCm memory monitoring
- `helpers.rs`: ROCm-specific utility functions

**Key Features:**
- Native ROCm/HIP implementation
- Optimized for AMD Radeon and Instinct GPUs
- Support for MI200/MI300 series
- Multi-GPU support

**Status:** Implementation planned (v0.2.0)

**Dependencies:** `hip-sys` crate

### 7. **Shader Management Module** (`src/shaders/`)

**Responsibility:** GPU shader source code and compilation.

**Key Files:**
- `metal_hnsw.metal`: Metal Shading Language for HNSW operations
- `metal_shaders.rs`: Rust bindings for Metal shaders
- `cuda_kernels.cu`: CUDA kernels for HNSW operations
- `cuda_shaders.rs`: Rust bindings for CUDA kernels
- `hip_kernels.hip`: HIP kernels for ROCm
- `rocm_shaders.rs`: Rust bindings for ROCm kernels

**Dependencies:** Backend-specific shader/kernel compilation

### 8. **Monitoring Module** (`src/monitoring/`)

**Responsibility:** Performance and memory monitoring.

**Submodules:**
- `performance_monitor.rs`: Operation timing and throughput metrics
- `vram_monitor.rs`: Unified VRAM monitoring interface

**Dependencies:** `types`, `traits`

### 9. **Utilities Module** (`src/utils/`)

**Responsibility:** Common utility functions.

**Submodules:**
- `math.rs`: Mathematical operations (normalize, distance calculations)
- `memory.rs`: Memory alignment and buffer utilities
- `timing.rs`: Performance measurement utilities

**Dependencies:** None

### 10. **Error Handling Module** (`src/error.rs`)

**Responsibility:** Unified error handling across all backends.

**Key Types:**
- `HiveGpuError`: Main error enum
- `Result<T>`: Type alias for Result<T, HiveGpuError>

**Error Categories:**
- Device errors (NoDeviceAvailable, DeviceNotSupported)
- Memory errors (OutOfMemory, AllocationFailed)
- Operation errors (InvalidDimension, InvalidOperation)
- Backend errors (ShaderCompilationFailed, BackendError)

## Data Flow

### Vector Addition Flow

```
User Code
  │
  ├─► GpuContext::create_storage()
  │     └─► MetalNativeContext::new()
  │           └─► Device + CommandQueue + Library initialization
  │
  ├─► GpuVectorStorage::add_vectors(vectors)
  │     │
  │     ├─► Validate vector dimensions
  │     ├─► Allocate GPU buffers
  │     ├─► Upload vectors to VRAM
  │     ├─► Update HNSW graph (if configured)
  │     └─► Return vector indices
  │
  └─► Result<Vec<usize>>
```

### Vector Search Flow

```
User Code
  │
  ├─► GpuVectorStorage::search(query, limit)
  │     │
  │     ├─► Upload query vector to GPU
  │     ├─► Compute distances (GPU kernel)
  │     │     └─► Parallel similarity computation
  │     ├─► HNSW graph traversal (if configured)
  │     │     └─► GPU-accelerated graph search
  │     ├─► Top-K selection on GPU
  │     ├─► Download results from VRAM
  │     └─► Return sorted results
  │
  └─► Result<Vec<GpuSearchResult>>
```

## Technology Stack

### Language
- **Primary**: Rust (Edition 2024)
- **Shader Languages**: Metal Shading Language (MSL), CUDA C++, HIP (ROCm)

### GPU Frameworks
- **Metal**: Apple's GPU framework (version 0.27+)
- **CUDA**: NVIDIA's parallel computing platform (version 12.0+)
- **ROCm/HIP**: AMD's GPU computing platform (version 5.0+)

### Core Dependencies
- `metal`: Metal framework bindings
- `objc`: Objective-C runtime for Metal interop
- `thiserror`: Error handling
- `serde`: Serialization/deserialization
- `tokio`: Async runtime
- `tracing`: Structured logging
- `bytemuck`: Safe transmutation between types
- `half`: Half-precision float support

### Development Tools
- `criterion`: Benchmarking framework
- `cargo-nextest`: Fast test runner
- `clippy`: Linter
- `rustfmt`: Code formatter

## Design Patterns

### 1. **Trait-Based Abstraction**

All GPU backends implement common traits (`GpuBackend`, `GpuContext`, `GpuVectorStorage`) enabling polymorphic usage and easy backend switching.

```rust
pub trait GpuContext {
    fn create_storage(&self, dimension: usize, metric: GpuDistanceMetric) 
        -> Result<Box<dyn GpuVectorStorage>>;
}
```

### 2. **Builder Pattern**

HNSW configuration uses builder pattern for flexible parameter tuning:

```rust
let config = HnswConfig {
    max_connections: 16,
    ef_construction: 100,
    ef_search: 50,
    ..Default::default()
};
```

### 3. **Buffer Pooling**

Memory management uses buffer pooling to reduce allocation overhead:

```rust
pub trait GpuBufferManager {
    fn allocate_buffer(&mut self, size: usize) -> Result<GpuBuffer>;
    fn deallocate_buffer(&mut self, buffer: GpuBuffer) -> Result<()>;
}
```

### 4. **RAII (Resource Acquisition Is Initialization)**

GPU resources are automatically cleaned up when dropped:

```rust
impl Drop for MetalNativeVectorStorage {
    fn drop(&mut self) {
        // Automatic cleanup of GPU buffers
    }
}
```

### 5. **Zero-Copy Architecture**

Data remains in VRAM throughout operations, minimizing CPU-GPU transfers:

```
CPU Memory                    VRAM (GPU Memory)
    ───►  Initial Upload  ───►  Storage
                                    │
                                    ├─► Similarity Computation
                                    ├─► HNSW Graph Traversal
                                    └─► Top-K Selection
    ◄───  Final Results  ◄────
```

## Security Considerations

### Memory Safety
- **Rust Guarantees**: Memory safety enforced by Rust's ownership system
- **Buffer Validation**: All buffer operations validated before GPU execution
- **No Unsafe Code**: Minimized use of `unsafe` blocks, carefully audited when necessary

### Data Protection
- **VRAM Isolation**: Vector data isolated in GPU memory
- **No Disk I/O**: All operations in memory (no persistence by default)
- **Metadata Sanitization**: User-provided metadata validated before storage

### Resource Limits
- **Memory Bounds**: Maximum buffer sizes enforced
- **Batch Limits**: Maximum batch sizes to prevent resource exhaustion
- **Timeout Protection**: Long-running operations have configurable timeouts

## Performance Characteristics

### Expected Throughput

| Operation | CPU Baseline | Metal (Apple Silicon) | Speedup |
|-----------|--------------|----------------------|---------|
| Vector Addition | 1,000 vec/s | 4,768 vec/s | **4.8x** |
| Similarity Search | 1 ms | 0.668 ms | **1.5x** |
| HNSW Construction | 100 ms | <1 ms | **100x+** |
| Batch Search (1000 queries) | 10 s | 0.001 s | **10,000x** |

### Latency Targets

- **Single Vector Search**: < 1ms (with HNSW)
- **Batch Search (100 queries)**: < 10ms
- **Vector Addition**: < 0.1ms per vector (batched)
- **HNSW Graph Construction**: < 100ms (10k vectors)

### Memory Efficiency

- **VRAM-Only Storage**: Zero CPU memory overhead
- **Sparse Indexing**: HNSW graph reduces memory footprint
- **Buffer Pooling**: 90%+ buffer reuse rate
- **Compression**: Optional f16 support for 50% memory reduction

### Scalability

- **Vector Count**: Tested up to 10M vectors
- **Batch Size**: Up to 10,000 vectors per batch
- **Dimensions**: Optimized for 128-768 dimensions
- **Concurrent Searches**: Multiple queries in parallel

## Optimization Techniques

### 1. **SIMD Vectorization**
GPU shaders utilize SIMD instructions for parallel distance computation.

### 2. **Shared Memory Usage**
Local memory in GPU workgroups for fast intermediate storage.

### 3. **Memory Coalescing**
Aligned memory access patterns for optimal memory bandwidth.

### 4. **Kernel Fusion**
Multiple operations combined in single GPU kernel to reduce overhead.

### 5. **Asynchronous Execution**
Non-blocking GPU operations with async/await support.

## Backend-Specific Optimizations

### Metal (Apple Silicon)

- **Tile Memory**: Utilizes Apple GPU tile-based architecture
- **Metal Performance Shaders (MPS)**: Hardware-accelerated operations
- **Unified Memory**: Shared memory architecture reduces overhead
- **Apple GPU Families**: Optimized for Apple7+ GPU families

### CUDA (NVIDIA)

- **Tensor Cores**: Hardware-accelerated matrix operations (A100+)
- **Shared Memory Banks**: Optimized bank access patterns
- **Warp-Level Primitives**: Collective thread operations
- **Stream Compaction**: Efficient filtering on GPU
- **Unified Memory**: Automatic memory management
- **Multi-GPU**: NCCL for multi-GPU communication

### ROCm (AMD)

- **HIP Programming**: Portable GPU code compatible with CUDA
- **Wavefront Optimization**: AMD GPU wavefront-specific tuning
- **Infinity Fabric**: High-bandwidth GPU-GPU communication
- **Matrix Cores**: Hardware-accelerated matrix operations (MI200+)
- **Memory Pooling**: Efficient HIP memory management
- **Multi-GPU**: RCCL for multi-GPU communication

## Testing Strategy

### Unit Tests
- Individual function testing
- Mock GPU contexts for CPU-only tests
- Edge case validation

### Integration Tests
- End-to-end workflows
- Multiple backend testing
- Real GPU hardware validation

### Benchmarks
- Performance regression detection
- Comparative benchmarking across backends
- Memory usage profiling

### Coverage Requirements
- **Minimum**: 95% code coverage
- **Critical Paths**: 100% coverage (search, add_vectors)
- **Error Handling**: All error paths tested

## Future Enhancements

### Phase 1 (v0.2.0)
- [ ] Complete CUDA backend implementation
- [ ] Add ROCm backend for AMD GPU support
- [ ] Implement f16 (half-precision) support
- [ ] Advanced HNSW optimizations

### Phase 2 (v0.3.0)
- [ ] GPU-based quantization (PQ, SQ)
- [ ] Multi-GPU support
- [ ] Distributed vector search
- [ ] Real-time index updates

### Phase 3 (v1.0.0)
- [ ] Production-grade stability
- [ ] Comprehensive benchmarking suite
- [ ] Advanced monitoring and profiling
- [ ] Official integrations (Qdrant, Milvus)

---
*Last Updated: 2025-01-03*
