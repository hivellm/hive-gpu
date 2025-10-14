# Hive GPU

[![Crates.io](https://img.shields.io/crates/v/hive-gpu.svg)](https://crates.io/crates/hive-gpu)
[![Documentation](https://docs.rs/hive-gpu/badge.svg)](https://docs.rs/hive-gpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance GPU acceleration for vector operations in Rust. Supports Metal (Apple Silicon), CUDA (NVIDIA), and wgpu (cross-platform) backends.

## 🚀 Features

- **🚀 Metal Native**: Apple Silicon (M1/M2/M3/M4) acceleration with zero-copy operations
- **🔥 CUDA**: NVIDIA GPU acceleration for Linux/Windows
- **🌐 wgpu**: Cross-platform GPU acceleration via Vulkan/DirectX12/Metal
- **⚡ Zero-copy operations**: All vector data stored in VRAM for maximum performance
- **🔗 HNSW indexing**: GPU-accelerated graph construction and search
- **📊 VRAM monitoring**: Real-time memory usage tracking
- **🔄 Buffer pooling**: Efficient memory management
- **🎯 Multiple distance metrics**: Cosine, Euclidean, Dot Product

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hive-gpu = { version = "0.1.0", features = ["metal-native"] }
```

### Feature Flags

- `metal-native`: Metal Native support (macOS only)
- `cuda`: CUDA support (Linux/Windows with NVIDIA GPUs)
- `wgpu`: Cross-platform GPU support via wgpu
- `full`: All backends enabled

## 🚀 Quick Start

### Metal Native (Apple Silicon)

```rust
use hive_gpu::metal::{MetalNativeContext, MetalNativeVectorStorage};
use hive_gpu::{GpuVector, GpuDistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create Metal context
    let context = MetalNativeContext::new()?;
    
    // Create vector storage
    let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
    
    // Add vectors
    let vectors = vec![
        GpuVector {
            id: "vec1".to_string(),
            data: vec![1.0; 512],
            metadata: std::collections::HashMap::new(),
        },
        GpuVector {
            id: "vec2".to_string(),
            data: vec![2.0; 512],
            metadata: std::collections::HashMap::new(),
        },
    ];
    
    storage.add_vectors(&vectors)?;
    
    // Search for similar vectors
    let results = storage.search(&vec![1.5; 512], 10)?;
    
    for result in results {
        println!("Found vector {} with score {}", result.id, result.score);
    }
    
    Ok(())
}
```

### CUDA (NVIDIA GPUs)

```rust
use hive_gpu::cuda::{CudaContext, CudaVectorStorage};
use hive_gpu::{GpuVector, GpuDistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CUDA context
    let context = CudaContext::new()?;
    
    // Create vector storage
    let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
    
    // Use the same API as Metal
    // ...
    
    Ok(())
}
```

### wgpu (Cross-platform)

```rust
use hive_gpu::wgpu::{WgpuContext, WgpuVectorStorage};
use hive_gpu::{GpuVector, GpuDistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create wgpu context
    let context = WgpuContext::new()?;
    
    // Create vector storage
    let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
    
    // Use the same API as Metal/CUDA
    // ...
    
    Ok(())
}
```

## 🏗️ Architecture

### Core Components

- **Context**: GPU device management and resource allocation
- **Vector Storage**: High-performance vector storage in VRAM
- **HNSW Graph**: GPU-accelerated graph construction and search
- **Buffer Pool**: Efficient memory management
- **VRAM Monitor**: Real-time memory usage tracking

### Backend Support

| Backend | Platform | GPU | Status |
|---------|----------|-----|--------|
| Metal Native | macOS | Apple Silicon | ✅ Production Ready |
| CUDA | Linux/Windows | NVIDIA | 🚧 In Development |
| wgpu | Cross-platform | AMD/NVIDIA/Intel | 🚧 In Development |

## 📊 Performance

### Benchmarks

| Operation | CPU (1 core) | Metal Native | Speedup |
|-----------|--------------|--------------|---------|
| Vector Addition (10K vectors) | 15ms | 2ms | 7.5x |
| Similarity Search (1K vectors) | 25ms | 3ms | 8.3x |
| HNSW Construction (10K vectors) | 120ms | 15ms | 8.0x |

*Benchmarks on Apple M2 Pro with 512-dimensional vectors*

## 🔧 Advanced Usage

### HNSW Graph Construction

```rust
use hive_gpu::metal::{MetalNativeContext, MetalNativeHnswGraph};
use hive_gpu::{GpuVector, GpuDistanceMetric, HnswConfig};

let context = MetalNativeContext::new()?;
let mut hnsw = MetalNativeHnswGraph::new(
    context,
    512,
    GpuDistanceMetric::Cosine,
    HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
        max_connections: 32,
    }
)?;

// Build graph from vectors
hnsw.build_graph(&vectors)?;

// Search using HNSW
let results = hnsw.search(&query, 10)?;
```

### VRAM Monitoring

```rust
use hive_gpu::monitoring::VramMonitor;

let monitor = VramMonitor::new(total_vram_bytes);
monitor.start(1000).await; // Check every 1000ms

// Get current usage
let used = monitor.used_vram();
let available = monitor.available_vram();
```

### Buffer Pooling

```rust
use hive_gpu::metal::{MetalNativeContext, MetalBufferPool};

let context = MetalNativeContext::new()?;
let mut pool = MetalBufferPool::new(context)?;

// Get buffer from pool
let buffer = pool.get_buffer(1024 * 1024)?; // 1MB buffer

// Return buffer to pool
pool.return_buffer(buffer)?;
```

## 🧪 Testing

Run tests with specific features:

```bash
# Test Metal Native
cargo test --features metal-native

# Test CUDA
cargo test --features cuda

# Test wgpu
cargo test --features wgpu

# Test all features
cargo test --features full
```

## 📈 Benchmarks

Run benchmarks:

```bash
# Metal Native benchmarks
cargo bench --features metal-native

# All backends
cargo bench --features full
```

## 🔍 Examples

See the `examples/` directory for more detailed examples:

- `metal_basic.rs`: Basic Metal Native usage
- `cuda_basic.rs`: Basic CUDA usage
- `wgpu_basic.rs`: Basic wgpu usage
- `hnsw_construction.rs`: HNSW graph construction
- `vram_monitoring.rs`: VRAM usage monitoring

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Clone the repository
2. Install Rust toolchain
3. Install platform-specific dependencies:
   - **macOS**: Xcode Command Line Tools
   - **Linux**: CUDA Toolkit (for CUDA support)
   - **Windows**: Visual Studio Build Tools

### Running Tests

```bash
# All tests
cargo test

# Metal Native only
cargo test --features metal-native

# CUDA only
cargo test --features cuda

# wgpu only
cargo test --features wgpu
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Metal](https://developer.apple.com/metal/) for Apple Silicon acceleration
- [CUDA](https://developer.nvidia.com/cuda-toolkit) for NVIDIA GPU acceleration
- [wgpu](https://wgpu.rs/) for cross-platform GPU support
- [HNSW](https://github.com/nmslib/hnswlib) for the graph algorithm inspiration

## 📚 Documentation

- [API Documentation](https://docs.rs/hive-gpu)
- [Examples](examples/)
- [Benchmarks](benches/)
- [Changelog](CHANGELOG.md)

## 🐛 Issues

Found a bug? Please report it on our [issue tracker](https://github.com/hivellm/hive-gpu/issues).

## 💬 Discussion

Join our community discussions on [GitHub Discussions](https://github.com/hivellm/hive-gpu/discussions).