# hive-gpu - API Reference

## Overview

This document provides a comprehensive reference for the hive-gpu API. For tutorials and examples, see the [README](../README.md) and [examples](../examples/).

---

## Core Types

### `GpuVector`

Represents a vector with its associated data and metadata.

```rust
pub struct GpuVector {
    pub id: String,
    pub data: Vec<f32>,
    pub metadata: HashMap<String, String>,
}
```

#### Methods

##### `new`
```rust
pub fn new(id: String, data: Vec<f32>) -> Self
```

Creates a new GPU vector with the given ID and data.

**Parameters:**
- `id`: Unique identifier for the vector
- `data`: Vector data (f32 values)

**Example:**
```rust
let vector = GpuVector::new("vec_1".to_string(), vec![1.0, 2.0, 3.0]);
```

##### `with_metadata`
```rust
pub fn with_metadata(
    id: String,
    data: Vec<f32>,
    metadata: HashMap<String, String>
) -> Self
```

Creates a new GPU vector with metadata.

**Example:**
```rust
let mut metadata = HashMap::new();
metadata.insert("source".to_string(), "document_1".to_string());
let vector = GpuVector::with_metadata("vec_1".into(), vec![1.0, 2.0], metadata);
```

##### `dimension`
```rust
pub fn dimension(&self) -> usize
```

Returns the dimension (length) of the vector.

##### `memory_size`
```rust
pub fn memory_size(&self) -> usize
```

Returns the approximate memory usage in bytes.

---

### `GpuDistanceMetric`

Distance metric for vector similarity computation.

```rust
pub enum GpuDistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}
```

#### Variants

- **`Cosine`**: Cosine similarity (1 - cosine distance)
  - Range: [0, 2] (higher = more similar)
  - Best for: Semantic similarity, normalized vectors
  
- **`Euclidean`**: Euclidean (L2) distance
  - Range: [0, ∞] (lower = more similar)
  - Best for: Spatial data, absolute distances
  
- **`DotProduct`**: Dot product similarity
  - Range: (-∞, ∞) (higher = more similar)
  - Best for: Magnitude-aware similarity

---

### `GpuSearchResult`

Result of a vector similarity search.

```rust
pub struct GpuSearchResult {
    pub id: String,
    pub score: f32,
    pub index: usize,
}
```

#### Fields

- **`id`**: Vector identifier
- **`score`**: Similarity score (interpretation depends on metric)
- **`index`**: Internal storage index

---

### `HnswConfig`

Configuration for HNSW (Hierarchical Navigable Small World) graph.

```rust
pub struct HnswConfig {
    pub max_connections: usize,      // Default: 16
    pub ef_construction: usize,      // Default: 100
    pub ef_search: usize,            // Default: 50
    pub max_level: usize,            // Default: 8
    pub level_multiplier: f32,       // Default: 0.5
    pub seed: Option<u64>,           // Default: None
}
```

#### Fields

- **`max_connections`**: Maximum bidirectional links per node (M parameter)
  - Higher = better recall, more memory
  - Typical range: 8-64
  
- **`ef_construction`**: Size of dynamic candidate list during construction
  - Higher = better quality, slower construction
  - Typical range: 100-500
  
- **`ef_search`**: Size of dynamic candidate list during search
  - Higher = better recall, slower search
  - Typical range: 50-500
  
- **`max_level`**: Maximum number of layers in the hierarchy
  - Typical range: 6-10
  
- **`level_multiplier`**: Level assignment probability
  - Default: 0.5
  
- **`seed`**: Random seed for reproducible level assignment
  - None = random seed

**Example:**
```rust
let config = HnswConfig {
    max_connections: 32,
    ef_construction: 200,
    ef_search: 100,
    max_level: 8,
    level_multiplier: 0.5,
    seed: Some(42),
};
```

---

### `GpuDeviceInfo`

Information about the GPU device.

```rust
pub struct GpuDeviceInfo {
    pub name: String,
    pub device_type: String,
    pub memory_bytes: u64,
    pub max_buffer_size: u64,
    pub compute_capability: Option<String>,
}
```

---

### `GpuMemoryStats`

GPU memory usage statistics.

```rust
pub struct GpuMemoryStats {
    pub total_allocated: usize,
    pub available: usize,
    pub utilization: f32,
    pub buffer_count: usize,
}
```

---

## Core Traits

### `GpuContext`

Factory trait for creating GPU vector storage instances.

```rust
pub trait GpuContext {
    fn create_storage(
        &self,
        dimension: usize,
        metric: GpuDistanceMetric
    ) -> Result<Box<dyn GpuVectorStorage>>;

    fn create_storage_with_config(
        &self,
        dimension: usize,
        metric: GpuDistanceMetric,
        config: HnswConfig
    ) -> Result<Box<dyn GpuVectorStorage>>;

    fn memory_stats(&self) -> GpuMemoryStats;
    fn device_info(&self) -> GpuDeviceInfo;
}
```

#### Methods

##### `create_storage`

Creates a new vector storage with default configuration.

**Parameters:**
- `dimension`: Vector dimension (must be consistent for all vectors)
- `metric`: Distance metric to use

**Returns:** `Result<Box<dyn GpuVectorStorage>>`

**Example:**
```rust
let context = MetalNativeContext::new()?;
let storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
```

##### `create_storage_with_config`

Creates a new vector storage with HNSW configuration.

**Parameters:**
- `dimension`: Vector dimension
- `metric`: Distance metric
- `config`: HNSW configuration

**Example:**
```rust
let config = HnswConfig {
    max_connections: 32,
    ef_construction: 200,
    ..Default::default()
};
let storage = context.create_storage_with_config(128, GpuDistanceMetric::Cosine, config)?;
```

---

### `GpuVectorStorage`

Main interface for vector operations.

```rust
pub trait GpuVectorStorage {
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>>;
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>>;
    fn remove_vectors(&mut self, ids: &[String]) -> Result<()>;
    fn vector_count(&self) -> usize;
    fn dimension(&self) -> usize;
    fn get_vector(&self, id: &str) -> Result<Option<GpuVector>>;
    fn clear(&mut self) -> Result<()>;
}
```

#### Methods

##### `add_vectors`

Adds multiple vectors to storage.

**Parameters:**
- `vectors`: Slice of `GpuVector` to add

**Returns:** `Result<Vec<usize>>` - Internal indices of added vectors

**Errors:**
- `InvalidDimension`: Vector dimension doesn't match storage
- `OutOfMemory`: Insufficient GPU memory
- `GpuOperationFailed`: GPU computation failed

**Example:**
```rust
let vectors = vec![
    GpuVector::new("v1".into(), vec![1.0, 2.0, 3.0]),
    GpuVector::new("v2".into(), vec![4.0, 5.0, 6.0]),
];
let indices = storage.add_vectors(&vectors)?;
```

**Performance:**
- Time Complexity: O(n × d) where n = vector count, d = dimension
- With HNSW: O(n × log(n) × d)
- Batching is more efficient than individual additions

##### `search`

Searches for k nearest neighbors.

**Parameters:**
- `query`: Query vector (must match storage dimension)
- `limit`: Maximum number of results (k)

**Returns:** `Result<Vec<GpuSearchResult>>` - Sorted by similarity (descending)

**Errors:**
- `InvalidDimension`: Query dimension doesn't match storage
- `InvalidOperation`: Empty storage
- `GpuOperationFailed`: GPU computation failed

**Example:**
```rust
let query = vec![1.0, 2.0, 3.0];
let results = storage.search(&query, 10)?;
for result in results {
    println!("{}: {}", result.id, result.score);
}
```

**Performance:**
- Time Complexity: O(n × d) brute-force
- With HNSW: O(log(n) × d)
- GPU-accelerated: Up to 100x faster than CPU

##### `remove_vectors`

Removes vectors by their IDs.

**Parameters:**
- `ids`: Slice of vector IDs to remove

**Returns:** `Result<()>`

**Example:**
```rust
storage.remove_vectors(&["v1", "v2"])?;
```

##### `vector_count`

Returns the total number of vectors in storage.

```rust
let count = storage.vector_count();
```

##### `dimension`

Returns the vector dimension of the storage.

```rust
let dim = storage.dimension();
```

##### `get_vector`

Retrieves a vector by ID.

**Parameters:**
- `id`: Vector ID

**Returns:** `Result<Option<GpuVector>>` - `None` if not found

**Example:**
```rust
if let Some(vector) = storage.get_vector("v1")? {
    println!("Found vector with {} dimensions", vector.dimension());
}
```

##### `clear`

Removes all vectors from storage.

```rust
storage.clear()?;
assert_eq!(storage.vector_count(), 0);
```

---

### `GpuBackend`

Backend information and capabilities.

```rust
pub trait GpuBackend {
    fn device_info(&self) -> GpuDeviceInfo;
    fn supports_operations(&self) -> GpuCapabilities;
    fn memory_stats(&self) -> GpuMemoryStats;
}
```

---

## Backend Implementations

### Metal Native (macOS)

#### `MetalNativeContext`

Metal backend context for Apple Silicon.

```rust
impl MetalNativeContext {
    pub fn new() -> Result<Self>;
    pub fn device(&self) -> &MetalDevice;
    pub fn command_queue(&self) -> &CommandQueue;
    pub fn device_name(&self) -> String;
    pub fn supports_mps(&self) -> bool;
    pub fn max_threadgroup_size(&self) -> MTLSize;
    pub fn max_buffer_size(&self) -> u64;
}
```

**Example:**
```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};

let context = MetalNativeContext::new()?;
println!("Device: {}", context.device_name());

let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
```

---

## Error Handling

### `HiveGpuError`

Main error type for the library.

```rust
pub enum HiveGpuError {
    NoDeviceAvailable,
    DeviceNotSupported(String),
    OutOfMemory(String),
    AllocationFailed(String),
    InvalidDimension { expected: usize, got: usize },
    InvalidOperation(String),
    InvalidConfiguration(String),
    ShaderCompilationFailed(String),
    GpuOperationFailed(String),
    BackendError(String),
    Io(std::io::Error),
    Other(String),
}
```

#### Variants

- **`NoDeviceAvailable`**: No GPU device found
- **`DeviceNotSupported`**: Device doesn't support required features
- **`OutOfMemory`**: Insufficient GPU memory
- **`AllocationFailed`**: Buffer allocation failed
- **`InvalidDimension`**: Vector dimension mismatch
- **`InvalidOperation`**: Operation not allowed in current state
- **`InvalidConfiguration`**: Invalid configuration parameters
- **`ShaderCompilationFailed`**: GPU shader compilation error
- **`GpuOperationFailed`**: GPU computation error
- **`BackendError`**: Backend-specific error
- **`Io`**: I/O error
- **`Other`**: Other errors

---

## Backend Detection

### Functions

#### `detect_available_backends`

```rust
pub fn detect_available_backends() -> Vec<GpuBackendType>
```

Returns a list of all available GPU backends on the system.

**Example:**
```rust
use hive_gpu::backends::detector::detect_available_backends;

let backends = detect_available_backends();
for backend in backends {
    println!("Available: {}", backend);
}
```

#### `select_best_backend`

```rust
pub fn select_best_backend() -> Result<GpuBackendType>
```

Selects the best available backend based on performance priority.

**Priority:** Metal > CUDA > CPU

**Example:**
```rust
use hive_gpu::backends::detector::select_best_backend;

let best = select_best_backend()?;
println!("Using backend: {}", best);
```

---

## Feature Flags

Enable specific backends via Cargo features:

```toml
[dependencies]
hive-gpu = { version = "0.1", features = ["metal-native"] }  # macOS
hive-gpu = { version = "0.1", features = ["cuda"] }          # NVIDIA
hive-gpu = { version = "0.1", features = ["rocm"] }          # AMD GPU
hive-gpu = { version = "0.1", features = ["metal-native", "cuda", "rocm"] }  # All native backends
```

### Available Features

- **`metal-native`** (default): Pure Metal backend for Apple Silicon
- **`cuda`**: CUDA backend for NVIDIA GPUs (planned)
- **`rocm`**: ROCm backend for AMD GPUs (planned)

---

## Performance Characteristics

### Time Complexity

| Operation | Brute Force | With HNSW |
|-----------|-------------|-----------|
| Add Vector | O(1) | O(log n) |
| Search | O(n × d) | O(log n × d) |
| Remove Vector | O(n) | O(log n) |

Where:
- `n` = number of vectors
- `d` = vector dimension

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Vector Data | n × d × 4 bytes | f32 values |
| HNSW Graph | n × M × 8 bytes | M = max_connections |
| Metadata | n × ~64 bytes | ID + metadata |

### GPU Memory (VRAM)

All vector data and HNSW graphs are stored entirely in GPU memory (VRAM) for maximum performance. CPU memory is only used for:
- Initial vector upload
- Final search results download
- Metadata storage

---

## Best Practices

### 1. Batch Operations

```rust
// ✅ GOOD: Batch addition
let vectors = (0..1000).map(|i| GpuVector::new(/*...*/)).collect::<Vec<_>>();
storage.add_vectors(&vectors)?;

// ❌ BAD: Individual additions
for i in 0..1000 {
    storage.add_vectors(&[GpuVector::new(/*...*/)])?;  // Slow!
}
```

### 2. Dimension Consistency

```rust
// ✅ GOOD: All vectors have same dimension
let storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
let v1 = GpuVector::new("v1".into(), vec![0.0; 128]);
let v2 = GpuVector::new("v2".into(), vec![0.0; 128]);

// ❌ BAD: Dimension mismatch
let v3 = GpuVector::new("v3".into(), vec![0.0; 64]);  // Error!
```

### 3. HNSW Configuration

```rust
// For high recall (accuracy):
let config = HnswConfig {
    max_connections: 32,
    ef_construction: 200,
    ef_search: 100,
    ..Default::default()
};

// For high speed:
let config = HnswConfig {
    max_connections: 16,
    ef_construction: 100,
    ef_search: 50,
    ..Default::default()
};
```

### 4. Error Handling

```rust
// ✅ GOOD: Handle errors properly
match storage.search(&query, 10) {
    Ok(results) => process_results(results),
    Err(HiveGpuError::InvalidDimension { expected, got }) => {
        eprintln!("Dimension mismatch: expected {}, got {}", expected, got);
    }
    Err(e) => eprintln!("Search failed: {}", e),
}

// ❌ BAD: Unwrap
let results = storage.search(&query, 10).unwrap();  // Panic on error!
```

---

## Version Compatibility

| hive-gpu | Rust | Metal | CUDA | ROCm |
|----------|------|-------|------|------|
| 0.1.x | 1.85+ | 0.27+ | N/A | N/A |
| 0.2.x (planned) | 1.85+ | 0.27+ | TBD | TBD |

---

*Last Updated: 2025-01-03*
*API Version: 0.1.6*

