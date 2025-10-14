# Migration Guide

This guide helps you migrate from the old GPU module in vectorizer to the new hive-gpu crate.

## Overview

The GPU module has been extracted from vectorizer into a separate crate called `hive-gpu`. This provides better modularity, reusability, and maintainability.

## For Users of vectorizer

**Good news!** If you're using vectorizer normally, nothing changes for you. The GPU acceleration is now powered by hive-gpu internally, but the API remains the same.

### Before (vectorizer 0.5.x)
```rust
use vectorizer::VectorStore;
use vectorizer::models::{CollectionConfig, DistanceMetric};

let store = VectorStore::new_auto();
let config = CollectionConfig {
    dimension: 512,
    metric: DistanceMetric::Cosine,
    // ... other config
};
store.create_collection("my_collection", config)?;
```

### After (vectorizer 0.6.x)
```rust
use vectorizer::VectorStore;
use vectorizer::models::{CollectionConfig, DistanceMetric};

// Same API, now powered by hive-gpu internally
let store = VectorStore::new_auto();
let config = CollectionConfig {
    dimension: 512,
    metric: DistanceMetric::Cosine,
    // ... other config
};
store.create_collection("my_collection", config)?;
```

## For Direct GPU Users

If you were using vectorizer's GPU module directly, you'll need to update your code.

### Before (Direct GPU Usage)
```rust
use vectorizer::gpu::MetalNativeCollection;
use vectorizer::models::{Vector, DistanceMetric};

let collection = MetalNativeCollection::new_with_name_and_config(
    "my_collection",
    config
)?;
```

### After (Using hive-gpu directly)
```rust
use hive_gpu::metal::{MetalNativeContext, MetalNativeVectorStorage};
use hive_gpu::{GpuVector, GpuDistanceMetric};

let context = MetalNativeContext::new()?;
let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
```

### After (Using vectorizer adapter)
```rust
use vectorizer::gpu_adapter::MetalNativeCollection;
use vectorizer::models::{Vector, DistanceMetric};

// Same API as before, but now uses hive-gpu internally
let collection = MetalNativeCollection::new_with_name_and_config(
    "my_collection",
    config
)?;
```

## API Changes

### Error Types

| Before | After | Notes |
|--------|-------|-------|
| `VectorizerError` | `HiveGpuError` | New error type for GPU operations |
| `VectorizerError::DimensionMismatch` | `HiveGpuError::DimensionMismatch` | Same structure |
| `VectorizerError::Other` | `HiveGpuError::Other` | Same structure |

### Vector Types

| Before | After | Notes |
|--------|-------|-------|
| `Vector` | `GpuVector` | Simplified structure |
| `Vector.payload` | `GpuVector.metadata` | Changed from `Option<Payload>` to `HashMap<String, String>` |
| `Vector.data` | `GpuVector.data` | Same structure |

### Distance Metrics

| Before | After | Notes |
|--------|-------|-------|
| `DistanceMetric::Cosine` | `GpuDistanceMetric::Cosine` | Same functionality |
| `DistanceMetric::Euclidean` | `GpuDistanceMetric::Euclidean` | Same functionality |
| `DistanceMetric::DotProduct` | `GpuDistanceMetric::DotProduct` | Same functionality |

## Feature Flags

### vectorizer Cargo.toml

```toml
[dependencies]
hive-gpu = { version = "0.1.0", optional = true }

[features]
# Metal Native via hive-gpu
hive-gpu-metal = ["hive-gpu", "hive-gpu/metal-native"]
# CUDA via hive-gpu
hive-gpu-cuda = ["hive-gpu", "hive-gpu/cuda"]
# wgpu via hive-gpu
hive-gpu-wgpu = ["hive-gpu", "hive-gpu/wgpu"]
```

### hive-gpu Cargo.toml

```toml
[dependencies]
hive-gpu = { version = "0.1.0", features = ["metal-native"] }
```

## Migration Steps

### 1. Update Dependencies

Add hive-gpu to your Cargo.toml:

```toml
[dependencies]
hive-gpu = { version = "0.1.0", features = ["metal-native"] }
```

### 2. Update Imports

```rust
// Before
use vectorizer::gpu::MetalNativeCollection;

// After
use hive_gpu::metal::{MetalNativeContext, MetalNativeVectorStorage};
use hive_gpu::{GpuVector, GpuDistanceMetric};
```

### 3. Update Vector Creation

```rust
// Before
let vector = Vector {
    id: "my_vector".to_string(),
    data: vec![1.0; 512],
    payload: Some(vec![
        ("category".to_string(), "test".to_string()),
    ]),
};

// After
let vector = GpuVector {
    id: "my_vector".to_string(),
    data: vec![1.0; 512],
    metadata: {
        let mut map = std::collections::HashMap::new();
        map.insert("category".to_string(), "test".to_string());
        map
    },
};
```

### 4. Update Context Creation

```rust
// Before
let collection = MetalNativeCollection::new_with_name_and_config(
    "my_collection",
    config
)?;

// After
let context = MetalNativeContext::new()?;
let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
```

### 5. Update Operations

```rust
// Before
collection.add_vector(vector)?;
let results = collection.search(&query, 10)?;

// After
storage.add_vectors(&[vector])?;
let results = storage.search(&query, 10)?;
```

## Performance Considerations

### Memory Usage

- **Before**: Vectors stored in CPU memory with GPU acceleration for operations
- **After**: Vectors stored in VRAM for maximum performance

### Batch Operations

- **Before**: Single vector operations
- **After**: Batch operations for better performance

```rust
// Before
for vector in vectors {
    collection.add_vector(vector)?;
}

// After
storage.add_vectors(&vectors)?;
```

## Troubleshooting

### Common Issues

1. **"No GPU device available"**
   - Ensure you have the correct GPU drivers installed
   - Check that the appropriate feature flags are enabled

2. **"Dimension mismatch"**
   - Ensure all vectors have the same dimension
   - Check that the storage was created with the correct dimension

3. **"VRAM limit exceeded"**
   - Reduce the number of vectors or their dimension
   - Consider using quantization or compression

### Debug Mode

Enable debug logging to troubleshoot issues:

```rust
use log::LevelFilter;
use tracing_subscriber;

tracing_subscriber::fmt()
    .with_max_level(LevelFilter::Debug)
    .init();
```

## Examples

### Complete Migration Example

```rust
// Before
use vectorizer::gpu::MetalNativeCollection;
use vectorizer::models::{Vector, DistanceMetric, CollectionConfig};

let config = CollectionConfig {
    dimension: 512,
    metric: DistanceMetric::Cosine,
    // ... other config
};

let collection = MetalNativeCollection::new_with_name_and_config(
    "my_collection",
    config
)?;

let vector = Vector {
    id: "test".to_string(),
    data: vec![1.0; 512],
    payload: None,
};

collection.add_vector(vector)?;
let results = collection.search(&vec![1.0; 512], 10)?;

// After
use hive_gpu::metal::{MetalNativeContext, MetalNativeVectorStorage};
use hive_gpu::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

let context = MetalNativeContext::new()?;
let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;

let vector = GpuVector {
    id: "test".to_string(),
    data: vec![1.0; 512],
    metadata: HashMap::new(),
};

storage.add_vectors(&[vector])?;
let results = storage.search(&vec![1.0; 512], 10)?;
```

## Support

If you encounter issues during migration:

1. Check the [documentation](https://docs.rs/hive-gpu)
2. Look at the [examples](examples/)
3. Report issues on [GitHub](https://github.com/hivellm/hive-gpu/issues)
4. Join discussions on [GitHub Discussions](https://github.com/hivellm/hive-gpu/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed changes between versions.
