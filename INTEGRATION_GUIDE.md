# 🔗 Hive-GPU Integration Guide

**Complete guide for integrating hive-gpu with vectorizer and standalone projects**

## 📦 Version Information

- **hive-gpu**: v0.1.0
- **vectorizer**: v0.6.0
- **Rust Edition**: 2024

## 🚀 Quick Integration

### 1. Add Dependencies

```toml
# Cargo.toml
[dependencies]
# For standalone use
hive-gpu = "0.1.0"

# For vectorizer integration
vectorizer = { git = "https://github.com/hivellm/vectorizer.git" }
hive-gpu = "0.1.0"
```

### 2. Enable GPU Features

```toml
# Choose your GPU backend
[dependencies]
hive-gpu = { version = "0.1.0", features = ["metal-native"] }  # macOS
hive-gpu = { version = "0.1.0", features = ["cuda"] }          # Linux/Windows
hive-gpu = { version = "0.1.0", features = ["wgpu"] }          # Cross-platform
```

## 🏗️ Integration Patterns

### Pattern 1: Direct GPU Usage

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let context = MetalNativeContext::new()?;
    let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
    
    // Your vector operations here
    let vectors = create_vectors();
    storage.add_vectors(&vectors)?;
    
    let results = storage.search(&query, 10)?;
    println!("Found {} results", results.len());
    
    Ok(())
}
```

### Pattern 2: Vectorizer Integration

```rust
use vectorizer::VectorStore;
use vectorizer::models::{CollectionConfig, DistanceMetric, HnswConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create vectorizer store
    let mut store = VectorStore::new();
    
    // Configure collection with GPU acceleration
    let config = CollectionConfig {
        dimension: 512,
        metric: DistanceMetric::Cosine,
        hnsw_config: HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            seed: 42,
        },
    };
    
    // Create collection
    store.create_collection("my_collection", config)?;
    
    // Add vectors
    let vectors = create_document_vectors();
    store.add_vectors("my_collection", vectors)?;
    
    // Search
    let query = create_query_vector();
    let results = store.search("my_collection", &query, 10)?;
    
    println!("Search completed: {} results", results.len());
    Ok(())
}
```

### Pattern 3: Custom Wrapper

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};

pub struct MyVectorDatabase {
    storage: Box<dyn GpuVectorStorage>,
    dimension: usize,
}

impl MyVectorDatabase {
    pub async fn new(dimension: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(dimension, GpuDistanceMetric::Cosine)?;
        
        Ok(Self {
            storage,
            dimension,
        })
    }
    
    pub async fn add_document(&mut self, id: &str, embedding: Vec<f32>, metadata: HashMap<String, String>) -> Result<(), Box<dyn std::error::Error>> {
        let vector = GpuVector {
            id: id.to_string(),
            data: embedding,
            metadata,
        };
        
        self.storage.add_vectors(&[vector])?;
        Ok(())
    }
    
    pub async fn search_similar(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
        let results = self.storage.search(query, limit)?;
        Ok(results.into_iter().map(|r| (r.id, r.score)).collect())
    }
    
    pub fn vector_count(&self) -> usize {
        self.storage.vector_count()
    }
}
```

## 🔧 Configuration Examples

### Basic Configuration

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuDistanceMetric, HnswConfig};

// Basic setup
let context = MetalNativeContext::new()?;
let storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
```

### Advanced Configuration with HNSW

```rust
use hive_gpu::types::HnswConfig;

// Configure HNSW for better search performance
let hnsw_config = HnswConfig {
    m: 16,              // Number of bi-directional links
    ef_construction: 200, // Size of dynamic candidate list
    ef_search: 50,      // Size of dynamic candidate list for search
    seed: 42,           // Random seed
};

let storage = context.create_storage_with_config(
    512, 
    GpuDistanceMetric::Cosine, 
    hnsw_config
)?;
```

### Multi-Backend Support

```rust
use hive_gpu::traits::{GpuContext, GpuVectorStorage};

// Auto-detect best available backend
let context = if cfg!(target_os = "macos") {
    Box::new(hive_gpu::metal::context::MetalNativeContext::new()?)
} else if cfg!(feature = "cuda") {
    Box::new(hive_gpu::cuda::context::CudaContext::new()?)
} else {
    Box::new(hive_gpu::wgpu::context::WgpuContext::new()?)
};

let storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
```

## 📊 Performance Optimization

### Batch Operations

```rust
// Efficient batch processing
let batch_size = 1000;
let mut vectors = Vec::with_capacity(batch_size);

for i in 0..batch_size {
    vectors.push(GpuVector {
        id: format!("batch_{}", i),
        data: generate_embedding(&format!("Document {}", i)),
        metadata: HashMap::new(),
    });
}

// Add in batches for better performance
storage.add_vectors(&vectors)?;
```

### Memory Management

```rust
use hive_gpu::monitoring::VramMonitor;

// Monitor GPU memory usage
let monitor = VramMonitor::new()?;
let stats = monitor.get_stats()?;

println!("GPU Memory Usage: {:.2} MB", stats.used_memory_mb);
println!("Available Memory: {:.2} MB", stats.available_memory_mb);
```

### Async Operations

```rust
use tokio::task;

// Parallel vector processing
let handles: Vec<_> = vector_batches
    .into_iter()
    .map(|batch| {
        let storage = storage.clone();
        task::spawn(async move {
            storage.add_vectors(&batch).await
        })
    })
    .collect();

// Wait for all batches to complete
for handle in handles {
    handle.await??;
}
```

## 🧪 Testing Integration

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use hive_gpu::metal::context::MetalNativeContext;
    use hive_gpu::traits::{GpuContext, GpuVectorStorage};

    #[tokio::test]
    async fn test_gpu_integration() {
        let context = MetalNativeContext::new().unwrap();
        let mut storage = context.create_storage(4, GpuDistanceMetric::Cosine).unwrap();
        
        let vectors = vec![
            GpuVector {
                id: "test_1".to_string(),
                data: vec![1.0, 0.0, 0.0, 0.0],
                metadata: HashMap::new(),
            },
        ];
        
        storage.add_vectors(&vectors).unwrap();
        
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = storage.search(&query, 1).unwrap();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test_1");
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_vectorizer_integration() {
    let mut store = VectorStore::new();
    
    let config = CollectionConfig {
        dimension: 128,
        metric: DistanceMetric::Cosine,
        hnsw_config: HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            seed: 42,
        },
    };
    
    store.create_collection("test_collection", config).unwrap();
    
    let vectors = vec![
        Vector {
            id: "doc_1".to_string(),
            data: vec![1.0; 128],
            payload: None,
        },
    ];
    
    store.add_vectors("test_collection", vectors).unwrap();
    
    let query = vec![1.0; 128];
    let results = store.search("test_collection", &query, 1).unwrap();
    
    assert_eq!(results.len(), 1);
}
```

## 🚀 Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM rust:1.75 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libclang-dev \
    pkg-config

# Copy source
COPY . .
RUN cargo build --release

FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libclang-14 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/my-app /usr/local/bin/

CMD ["my-app"]
```

### Environment Variables

```bash
# Development
export RUST_LOG=debug
export HIVE_GPU_MEMORY_LIMIT=4GB

# Production
export RUST_LOG=info
export HIVE_GPU_PROFILE=false
```

## 🔍 Troubleshooting

### Common Issues

#### 1. GPU Not Available
```rust
// Check GPU availability
if let Ok(context) = MetalNativeContext::new() {
    println!("GPU available");
} else {
    println!("GPU not available, falling back to CPU");
    // Implement CPU fallback
}
```

#### 2. Memory Issues
```rust
// Monitor memory usage
let monitor = VramMonitor::new()?;
let stats = monitor.get_stats()?;

if stats.used_memory_mb > 1000.0 {
    println!("Warning: High GPU memory usage");
}
```

#### 3. Performance Issues
```rust
// Use batch operations
let batch_size = 1000;
for chunk in vectors.chunks(batch_size) {
    storage.add_vectors(chunk)?;
}
```

### Debug Information

```rust
use tracing::{info, debug, error};

// Enable debug logging
tracing_subscriber::fmt::init();

// Log GPU operations
info!("Initializing GPU context");
let context = MetalNativeContext::new()?;
debug!("GPU context created successfully");

let storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
info!("Vector storage created with dimension: 128");
```

## 📚 Additional Resources

- **API Documentation**: [docs.rs/hive-gpu](https://docs.rs/hive-gpu)
- **Examples**: [GitHub Examples](https://github.com/hivellm/hive-gpu/tree/main/examples)
- **Vectorizer Integration**: [Vectorizer Docs](https://github.com/hivellm/vectorizer)
- **Performance Guide**: [Benchmarking Guide](BENCHMARKING.md)

## 🤝 Support

- **GitHub Issues**: [Report issues](https://github.com/hivellm/hive-gpu/issues)
- **Discussions**: [Community discussions](https://github.com/hivellm/hive-gpu/discussions)
- **Documentation**: [Complete API reference](https://docs.rs/hive-gpu)

---

**Ready to accelerate your vector operations with GPU power! 🚀**
