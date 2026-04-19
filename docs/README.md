# hive-gpu Documentation

Welcome to the hive-gpu documentation! This directory contains comprehensive guides and references for using hive-gpu in your projects.

## 📚 Documentation Index

### Getting Started

- **[Main README](../README.md)** - Project overview, quick start, and installation
- **[Integration Guide](guides/INTEGRATION_GUIDE.md)** - Step-by-step integration examples
- **[Examples](../examples/)** - Working code examples

### Core Documentation

- **[Architecture](architecture/ARCHITECTURE.md)** - System design, components, and data flow
- **[Development Guide](guides/DEVELOPMENT.md)** - Setup, workflow, testing, and contributing
- **[API Reference](reference/API_REFERENCE.md)** - Complete API documentation
- **[Performance Guide](benchmarks/PERFORMANCE.md)** - Optimization tips and benchmarks

### Project Planning

- **[Roadmap](ROADMAP.md)** - Development roadmap and milestones
- **[DAG (Dependencies)](DAG.md)** - Component dependencies and layer architecture

## 🎯 Quick Navigation

### I want to...

#### **Get Started**
→ Read [Main README](../README.md) → Run [examples](../examples/metal_basic.rs)

#### **Integrate hive-gpu**
→ Read [Integration Guide](guides/INTEGRATION_GUIDE.md) → Choose your use case

#### **Optimize Performance**
→ Read [Performance Guide](benchmarks/PERFORMANCE.md) → Apply optimization techniques

#### **Understand the Codebase**
→ Read [Architecture](architecture/ARCHITECTURE.md) → Check [DAG](DAG.md)

#### **Contribute Code**
→ Read [Development Guide](guides/DEVELOPMENT.md) → Follow quality gates

#### **Check API Details**
→ Read [API Reference](reference/API_REFERENCE.md) → Find the method you need

#### **Plan Features**
→ Check [Roadmap](ROADMAP.md) → See what's coming

## 📖 Documentation Structure

```
docs/
├── README.md                        # This file (documentation index)
├── DAG.md                           # Component dependencies (DAG)
├── ROADMAP.md                       # Project roadmap and milestones
├── architecture/
│   ├── ARCHITECTURE.md              # System architecture and design
│   └── HIVE_GPU_IMPLEMENTATION_RECOMMENDATIONS.md
├── reference/
│   └── API_REFERENCE.md             # Complete API documentation
├── guides/
│   ├── INTEGRATION_GUIDE.md         # Integration examples and patterns
│   ├── DEVELOPMENT.md               # Development workflow and setup
│   ├── EXAMPLES.md                  # Annotated examples
│   ├── CI_FIXES.md                  # CI fixes history
│   ├── DEVICE_INFO_IMPLEMENTATION.md
│   ├── MIGRATION_METAL_OBJC2.md
│   ├── MIGRATION_OBJC2_SUMMARY.md
│   └── ROADMAP_CONFIG.md
├── benchmarks/
│   ├── PERFORMANCE.md               # Performance guide and tuning
│   └── PERFORMANCE_RESULTS.md       # Benchmark results
└── analysis/                        # Research and analysis notes
```

## 🚀 Quick Reference

### Core Types

```rust
use hive_gpu::types::{GpuVector, GpuDistanceMetric, GpuSearchResult, HnswConfig};
```

### Core Traits

```rust
use hive_gpu::traits::{GpuContext, GpuVectorStorage, GpuBackend};
```

### Backends

```rust
// Metal (macOS)
use hive_gpu::metal::context::MetalNativeContext;

// CUDA (NVIDIA) - Planned
use hive_gpu::cuda::context::CudaContext;

// ROCm (AMD GPUs) - Planned
use hive_gpu::rocm::context::RocmContext;
```

## 💡 Common Patterns

### Initialize and Search

```rust
let context = MetalNativeContext::new()?;
let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;

storage.add_vectors(&vectors)?;
let results = storage.search(&query, 10)?;
```

### With HNSW Configuration

```rust
let config = HnswConfig {
    max_connections: 32,
    ef_construction: 200,
    ef_search: 100,
    ..Default::default()
};

let storage = context.create_storage_with_config(128, GpuDistanceMetric::Cosine, config)?;
```

### Batch Operations

```rust
// Add in batches
for chunk in vectors.chunks(5000) {
    storage.add_vectors(chunk)?;
}

// Search multiple queries
for query in queries {
    let results = storage.search(&query, 10)?;
    process_results(results);
}
```

## 🔍 Finding What You Need

### By Topic

| Topic | Document |
|-------|----------|
| Installation & Setup | [Main README](../README.md), [Development Guide](guides/DEVELOPMENT.md) |
| Basic Usage | [Main README](../README.md), [Examples](../examples/) |
| Integration Patterns | [Integration Guide](guides/INTEGRATION_GUIDE.md) |
| API Reference | [API Reference](reference/API_REFERENCE.md) |
| Performance Tuning | [Performance Guide](benchmarks/PERFORMANCE.md) |
| Architecture Understanding | [Architecture](architecture/ARCHITECTURE.md), [DAG](DAG.md) |
| Contributing | [Development Guide](guides/DEVELOPMENT.md) |
| Future Plans | [Roadmap](ROADMAP.md) |

### By User Type

#### **Application Developer**
1. [Main README](../README.md) - Quick start
2. [Integration Guide](guides/INTEGRATION_GUIDE.md) - Integration patterns
3. [API Reference](reference/API_REFERENCE.md) - API details
4. [Performance Guide](benchmarks/PERFORMANCE.md) - Optimization

#### **Library Contributor**
1. [Development Guide](guides/DEVELOPMENT.md) - Setup and workflow
2. [Architecture](architecture/ARCHITECTURE.md) - System design
3. [DAG](DAG.md) - Dependencies
4. [Roadmap](ROADMAP.md) - Planned features

#### **Performance Engineer**
1. [Performance Guide](benchmarks/PERFORMANCE.md) - Benchmarks and tuning
2. [Architecture](architecture/ARCHITECTURE.md) - Design decisions
3. [API Reference](reference/API_REFERENCE.md) - Configuration options

## 🎓 Learning Path

### Beginner

1. **[Main README](../README.md)** - Understand what hive-gpu is
2. **[Examples](../examples/metal_basic.rs)** - Run your first example
3. **[API Reference](reference/API_REFERENCE.md)** - Learn core types and traits

### Intermediate

1. **[Integration Guide](guides/INTEGRATION_GUIDE.md)** - Build real applications
2. **[Performance Guide](benchmarks/PERFORMANCE.md)** - Optimize your code
3. **[Architecture](architecture/ARCHITECTURE.md)** - Understand internal design

### Advanced

1. **[Development Guide](guides/DEVELOPMENT.md)** - Contribute to the project
2. **[DAG](DAG.md)** - Understand dependencies
3. **[Roadmap](ROADMAP.md)** - Shape the future

## 🤝 Contributing to Documentation

Found a typo? Want to improve clarity? We welcome documentation contributions!

### How to Contribute

1. **Fork the repository**
2. **Edit the relevant `.md` file**
3. **Submit a pull request**

### Documentation Guidelines

- **Clear and Concise**: Write for developers of all skill levels
- **Code Examples**: Include working, tested examples
- **Consistent Style**: Follow existing formatting
- **Up-to-Date**: Ensure code matches current API
- **English**: All documentation must be in English

### What to Document

- ✅ New features
- ✅ API changes
- ✅ Performance improvements
- ✅ Integration patterns
- ✅ Common pitfalls
- ✅ Best practices

## 📝 Documentation Standards

### Code Examples

All code examples must:
- Be **syntactically correct**
- Include **necessary imports**
- Be **runnable** (or clearly marked as pseudo-code)
- Include **error handling**
- Follow **Rust best practices**

### Markdown Formatting

- Use **clear headings** (H2 for sections, H3 for subsections)
- Include **code fences** with language specification
- Add **tables** for structured data
- Use **lists** for sequential information
- Include **links** to related documentation

## 🔗 External Resources

- **Main Repository**: [github.com/hivellm/hive-gpu](https://github.com/hivellm/hive-gpu)
- **API Docs**: [docs.rs/hive-gpu](https://docs.rs/hive-gpu)
- **Crate**: [crates.io/crates/hive-gpu](https://crates.io/crates/hive-gpu)
- **Discussions**: [GitHub Discussions](https://github.com/hivellm/hive-gpu/discussions)
- **Issues**: [GitHub Issues](https://github.com/hivellm/hive-gpu/issues)

## 📧 Support

- **Questions**: [GitHub Discussions](https://github.com/hivellm/hive-gpu/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/hivellm/hive-gpu/issues)
- **Feature Requests**: [GitHub Issues](https://github.com/hivellm/hive-gpu/issues)
- **Documentation Issues**: [GitHub Issues](https://github.com/hivellm/hive-gpu/issues)

---

**Happy Coding! 🚀**
