# hive-gpu - Component Dependencies (DAG)

## Overview

This document describes the dependency graph (Directed Acyclic Graph) between components in hive-gpu. Understanding these dependencies is crucial for maintaining code quality, preventing circular dependencies, and ensuring proper layering.

## Dependency Graph Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                         Layer 4: Examples                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ metal_basic  │  │ cuda_basic   │  │ rocm_basic   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼─────────────────┐
│         │          Layer 3: Backend Implementations              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐          │
│  │    Metal     │  │     CUDA     │  │     ROCm     │          │
│  │   Backend    │  │   Backend    │  │   Backend    │          │
│  │              │  │              │  │              │          │
│  │ • context    │  │ • context    │  │ • context    │          │
│  │ • storage    │  │ • storage    │  │ • storage    │          │
│  │ • hnsw_graph │  │ • hnsw_graph │  │ • hnsw_graph │          │
│  │ • buffer_pool│  │ • buffer_pool│  │ • buffer_pool│          │
│  │ • vram_mon   │  │ • vram_mon   │  │ • vram_mon   │          │
│  │ • helpers    │  │ • helpers    │  │ • helpers    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    Layer 2: Core Abstractions                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Traits     │  │  Backends    │  │  Monitoring  │           │
│  │              │  │              │  │              │           │
│  │ • GpuBackend │  │ • detector   │  │ • perf_mon   │           │
│  │ • GpuContext │  │ • selector   │  │ • vram_mon   │           │
│  │ • GpuStorage │  │              │  │              │           │
│  │ • GpuBuffer  │  │              │  │              │           │
│  │ • GpuMonitor │  │              │  │              │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    Layer 1: Foundation                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Types     │  │    Error     │  │    Utils     │           │
│  │              │  │              │  │              │           │
│  │ • GpuVector  │  │ • HiveGpuErr │  │ • math       │           │
│  │ • Metrics    │  │ • Result     │  │ • memory     │           │
│  │ • Config     │  │              │  │ • timing     │           │
│  │ • Stats      │  │              │  │              │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

## Detailed Component Dependencies

### Layer 1: Foundation

These components have no internal dependencies and serve as building blocks.

#### `src/types.rs` (Types)
**Dependencies:** None

**Provides:**
- `GpuVector` - Vector data structure
- `GpuDistanceMetric` - Distance metric enum
- `GpuSearchResult` - Search result structure
- `GpuDeviceInfo` - Device information
- `GpuCapabilities` - Device capabilities
- `GpuMemoryStats` - Memory statistics
- `HnswConfig` - HNSW configuration
- `VectorMetadata` - Metadata structure

**Dependents:** All modules

#### `src/error.rs` (Error)
**Dependencies:** None

**Provides:**
- `HiveGpuError` - Error enum
- `Result<T>` - Type alias

**Dependents:** All modules

#### `src/utils/` (Utilities)
**Dependencies:** None

**Provides:**
- `math.rs` - Vector normalization, distance calculations
- `memory.rs` - Memory alignment utilities
- `timing.rs` - Performance measurement

**Dependents:** Backend implementations, monitoring

---

### Layer 2: Core Abstractions

These components depend only on Layer 1 (Foundation).

#### `src/traits.rs` (Traits)
**Dependencies:**
- `types` (GpuVector, GpuDistanceMetric, etc.)
- `error` (Result)

**Provides:**
- `GpuBackend` - Backend interface
- `GpuContext` - Context factory trait
- `GpuVectorStorage` - Vector operations trait
- `GpuBufferManager` - Buffer management trait
- `GpuMonitor` - Monitoring trait

**Dependents:** Backend implementations, examples

#### `src/backends/` (Backend Detection)
**Dependencies:**
- `error` (HiveGpuError, Result)

**Provides:**
- `detector::detect_available_backends()` - Backend detection
- `detector::select_best_backend()` - Backend selection
- `detector::get_backend_info()` - Backend information
- `GpuBackendType` - Backend enum

**Dependents:** User code, examples

#### `src/monitoring/` (Monitoring)
**Dependencies:**
- `types` (GpuMemoryStats, etc.)
- `traits` (GpuMonitor)
- `utils/timing` (Performance measurement)

**Provides:**
- `performance_monitor::PerformanceMonitor` - Operation timing
- `vram_monitor::VramMonitor` - Memory monitoring

**Dependents:** Backend implementations

---

### Layer 3: Backend Implementations

These components depend on Layers 1-2 and provide concrete GPU backend implementations.

#### `src/metal/` (Metal Backend)
**Dependencies:**
- `types` (All core types)
- `traits` (All core traits)
- `error` (HiveGpuError, Result)
- `utils` (math, memory, timing)
- `monitoring` (PerformanceMonitor, VramMonitor)
- External: `metal` crate, `objc` crate

**Submodules:**

##### `metal/context.rs`
**Dependencies:** Layer 1, Layer 2, `metal` crate

**Provides:**
- `MetalNativeContext` - Metal device and command queue

**Dependents:** `metal/vector_storage`, `metal/hnsw_graph`

##### `metal/vector_storage.rs`
**Dependencies:** `metal/context`, `metal/buffer_pool`, `metal/vram_monitor`

**Provides:**
- `MetalNativeVectorStorage` - Vector storage implementation

**Dependents:** User code via trait

##### `metal/hnsw_graph.rs`
**Dependencies:** `metal/context`, `metal/buffer_pool`

**Provides:**
- `MetalHnswGraph` - HNSW graph on GPU

**Dependents:** `metal/vector_storage`

##### `metal/buffer_pool.rs`
**Dependencies:** `metal/context`

**Provides:**
- `MetalBufferPool` - Buffer pool management

**Dependents:** `metal/vector_storage`, `metal/hnsw_graph`

##### `metal/vram_monitor.rs`
**Dependencies:** `metal/context`, `monitoring/vram_monitor`

**Provides:**
- `MetalVramMonitor` - Metal-specific VRAM monitoring

**Dependents:** `metal/vector_storage`

##### `metal/helpers.rs`
**Dependencies:** `metal/context`, `utils`

**Provides:**
- Utility functions for Metal operations

**Dependents:** Other `metal/` modules

#### `src/cuda/` (CUDA Backend)
**Status:** Planned (placeholder)

**Dependencies:** Same as Metal backend + `cudarc` crate

**Submodules:**
- `cuda/context.rs` - CUDA device and stream management
- `cuda/vector_storage.rs` - CUDA vector storage
- `cuda/hnsw_graph.rs` - CUDA HNSW implementation
- `cuda/buffer_pool.rs` - CUDA memory pool
- `cuda/vram_monitor.rs` - CUDA memory monitoring
- `cuda/helpers.rs` - CUDA utilities

#### `src/rocm/` (ROCm Backend)
**Status:** Planned (placeholder)

**Dependencies:** Same as Metal backend + `hip-sys` crate, `rocm-sys` crate

**Submodules:**
- `rocm/context.rs` - ROCm/HIP device and context
- `rocm/vector_storage.rs` - ROCm vector storage
- `rocm/hnsw_graph.rs` - ROCm HNSW implementation with HIP kernels
- `rocm/buffer_pool.rs` - ROCm buffer pool
- `rocm/vram_monitor.rs` - ROCm memory monitoring (rocm-smi)
- `rocm/helpers.rs` - ROCm utilities

---

### Layer 4: Examples and Integration

These are end-user code that depends on all lower layers.

#### `examples/metal_basic.rs`
**Dependencies:**
- `metal::context::MetalNativeContext`
- `traits::{GpuContext, GpuVectorStorage}`
- `types::*`

#### `examples/cuda_basic.rs`
**Dependencies:**
- `cuda::context::CudaContext` (when implemented)
- `traits::{GpuContext, GpuVectorStorage}`
- `types::*`

#### `examples/rocm_basic.rs`
**Dependencies:**
- `rocm::context::RocmContext` (when implemented)
- `traits::{GpuContext, GpuVectorStorage}`
- `types::*`

---

## Dependency Rules

### 1. **No Circular Dependencies**

Components MUST form a Directed Acyclic Graph (DAG). Circular dependencies are strictly prohibited.

**✅ GOOD:**
```
types → traits → metal/context → metal/vector_storage
```

**❌ BAD:**
```
metal/vector_storage → metal/context → metal/vector_storage (circular!)
```

### 2. **Layer Separation**

Higher layers can depend on lower layers, but NOT vice versa.

**✅ GOOD:**
```
Backend (Layer 3) → Traits (Layer 2) → Types (Layer 1)
```

**❌ BAD:**
```
Types (Layer 1) → Backend (Layer 3)
```

### 3. **Interface Boundaries**

Cross-layer communication MUST go through trait interfaces, not concrete types.

**✅ GOOD:**
```rust
fn create_storage(ctx: &dyn GpuContext) -> Box<dyn GpuVectorStorage> {
    ctx.create_storage(128, GpuDistanceMetric::Cosine)
}
```

**❌ BAD:**
```rust
fn create_storage() -> MetalNativeVectorStorage {  // Concrete type!
    // ...
}
```

### 4. **Backend Isolation**

Backend implementations (Metal, CUDA, ROCm) MUST NOT depend on each other.

**✅ GOOD:**
```
metal/context → traits
cuda/context → traits
```

**❌ BAD:**
```
metal/context → cuda/context (cross-backend dependency!)
```

### 5. **No Upward Dependencies**

Foundation modules MUST NOT depend on higher layers.

**✅ GOOD:**
```rust
// types.rs
pub struct GpuVector {
    pub id: String,
    pub data: Vec<f32>,
}
```

**❌ BAD:**
```rust
// types.rs
use crate::metal::context::MetalNativeContext; // Depends on Layer 3!
```

---

## Verification

### Manual Verification

Check dependencies manually:

```bash
# Check dependencies of a module
cargo tree --package hive-gpu --edges normal,build --invert types

# Check for circular dependencies
cargo check --all-features
```

### Automated Verification (Future)

```bash
# Run dependency checker (to be implemented)
cargo run --bin check-deps

# Expected output:
# ✅ No circular dependencies found
# ✅ Layer separation respected
# ✅ Interface boundaries validated
```

---

## Dependency Matrix

| From ↓ To → | types | error | utils | traits | backends | monitoring | metal | cuda | rocm | examples |
|-------------|-------|-------|-------|--------|----------|------------|-------|------|------|----------|
| **types**   | -     | ❌    | ❌    | ❌     | ❌       | ❌         | ❌    | ❌   | ❌   | ❌       |
| **error**   | ❌    | -     | ❌    | ❌     | ❌       | ❌         | ❌    | ❌   | ❌   | ❌       |
| **utils**   | ✅    | ✅    | -     | ❌     | ❌       | ❌         | ❌    | ❌   | ❌   | ❌       |
| **traits**  | ✅    | ✅    | ❌    | -      | ❌       | ❌         | ❌    | ❌   | ❌   | ❌       |
| **backends**| ✅    | ✅    | ❌    | ❌     | -        | ❌         | ❌    | ❌   | ❌   | ❌       |
| **monitoring**| ✅  | ✅    | ✅    | ✅     | ❌       | -          | ❌    | ❌   | ❌   | ❌       |
| **metal**   | ✅    | ✅    | ✅    | ✅     | ✅       | ✅         | ✅*   | ❌   | ❌   | ❌       |
| **cuda**    | ✅    | ✅    | ✅    | ✅     | ✅       | ✅         | ❌    | ✅*  | ❌   | ❌       |
| **rocm**    | ✅    | ✅    | ✅    | ✅     | ✅       | ✅         | ❌    | ❌   | ✅*  | ❌       |
| **examples**| ✅    | ✅    | ❌    | ✅     | ✅       | ❌         | ✅    | ✅   | ✅   | -        |

**Legend:**
- ✅ = Allowed dependency
- ❌ = Prohibited dependency
- ✅* = Internal dependencies within same module (e.g., metal/context → metal/helpers)

---

## Module Load Order

When the library initializes, modules are loaded in this order:

1. **Foundation Layer** (parallel)
   - `types`
   - `error`
   - `utils`

2. **Core Abstractions** (parallel, after Foundation)
   - `traits`
   - `backends`
   - `monitoring`

3. **Shaders** (parallel, after Core)
   - `shaders/metal_shaders` (Phase 1 ✅)
   - `shaders/cuda_shaders` (Phase 3.1 🔥 CUDA)
   - `shaders/rocm_shaders` (Phase 3.2 ⚡ ROCm)

4. **Backend Implementations** (parallel, after Core + Shaders)
   - `metal` (Phase 1 ✅ - Apple Silicon, 5% market)
   - `cuda` (Phase 3.1 🔥 - NVIDIA, 70% market)
   - `rocm` (Phase 3.2 ⚡ - AMD, 15% market)

5. **Library Root** (`lib.rs`)
   - Re-exports public APIs

---

## Import Guidelines

### DO's ✅

```rust
// ✅ Import from lower layers
use crate::types::{GpuVector, GpuDistanceMetric};
use crate::error::{Result, HiveGpuError};
use crate::traits::GpuContext;

// ✅ Import within same layer
use super::context::MetalNativeContext;
use super::helpers::create_buffer;

// ✅ Use trait imports for abstraction
use crate::traits::GpuVectorStorage;
```

### DON'Ts ❌

```rust
// ❌ Import from higher layers
use crate::metal::context::MetalNativeContext;  // In types.rs - BAD!

// ❌ Import concrete backend in core traits
use crate::metal::vector_storage::MetalNativeVectorStorage;  // In traits.rs - BAD!

// ❌ Cross-backend imports
use crate::cuda::context::CudaContext;  // In metal/context.rs - BAD!
```

---

## Adding New Components

When adding a new component, follow these steps:

1. **Identify Layer**: Determine which layer your component belongs to
2. **Check Dependencies**: Ensure all dependencies are from lower layers
3. **Update DAG**: Add your component to this document
4. **Update Matrix**: Update the dependency matrix
5. **Verify**: Run `cargo check --all-features`

---

## Future Improvements

- [ ] Automated dependency checker tool
- [ ] CI/CD integration for dependency validation
- [ ] Dependency visualization tool
- [ ] Real-time dependency graph generation

---

*Last Updated: 2025-01-03*
*Next Review: 2025-02-01*
