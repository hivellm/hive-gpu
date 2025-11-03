# Project Context

## Purpose

`hive-gpu` is a high-performance GPU acceleration library for vector operations, specifically designed for vector similarity search workloads. The project aims to provide a unified API across multiple native GPU backends with optimized implementations for HNSW (Hierarchical Navigable Small World) graph-based approximate nearest neighbor search.

**Key Goals:**
- Achieve 10-50x speedup over CPU implementations
- Support all major GPU vendors (Apple, NVIDIA, AMD)
- Provide consistent API across backends
- Minimize latency (target: 0.5-3ms for searches)
- Maximize throughput (target: 10K+ vectors/second)

## Tech Stack

### Language
- **Rust** (Edition 2024, nightly toolchain 1.85+)

### GPU Frameworks
- **Metal** - Apple Silicon (M1/M2/M3/M4) - ✅ Implemented
- **CUDA** - NVIDIA GPUs (Volta+, compute capability 7.0+) - 🚧 Phase 3.1
- **ROCm/HIP** - AMD GPUs (gfx900+, Vega and newer) - 📋 Phase 3.2

### Key Dependencies
- `metal` - Metal framework bindings
- `cuda-runtime-sys`, `cublas-sys` - CUDA support
- `hip-runtime-sys`, `rocblas-sys` - ROCm support
- `thiserror` - Error handling
- `serde` - Serialization support

### Build Tools
- `cargo` - Build system and package manager
- `cc` - CUDA/HIP kernel compilation
- `nvcc` - NVIDIA CUDA compiler
- `hipcc` - AMD HIP compiler

## Project Conventions

### Code Style
- **Formatting**: `rustfmt` with nightly toolchain
- **Linting**: `clippy` with `-D warnings` (warnings as errors)
- **Naming**: Snake_case for functions/variables, PascalCase for types
- **Documentation**: Comprehensive rustdoc comments for all public APIs
- **Error Handling**: Use `Result<T, HiveGpuError>` for all fallible operations

### Architecture Patterns
- **Trait-based abstraction**: `GpuContext` and `GpuVectorStorage` traits
- **Builder pattern**: Contexts created with builders
- **RAII**: Automatic resource cleanup via Drop
- **Zero-copy**: VRAM-only storage, minimal host-GPU transfers
- **Backend isolation**: Backends don't depend on each other

### Testing Strategy
- **Coverage target**: ≥95%
- **Unit tests**: In same file with `#[cfg(test)]`
- **Integration tests**: In `/tests` directory
- **Backend-specific tests**: Conditional on feature flags
- **Cross-backend consistency**: Verify identical results across backends
- **Performance benchmarks**: Track regression and improvements

### Git Workflow
- **Branching**: Feature branches for all changes
- **Commits**: Conventional commit format (`feat:`, `fix:`, `docs:`, etc.)
- **Quality gates**: Format, lint, test, build must all pass
- **No direct commits**: to main without quality checks
- **Signed commits**: Recommended for production

## Domain Context

### Vector Similarity Search
- High-dimensional vectors (typically 128-1536 dimensions)
- Distance metrics: Cosine, Euclidean, Dot Product
- HNSW algorithm for approximate nearest neighbor search
- Trade-off between recall and performance

### GPU Computing Fundamentals
- **VRAM management**: Careful memory allocation and tracking
- **Async operations**: Stream-based execution
- **Kernel optimization**: Architecture-specific tuning
- **Memory coalescing**: Critical for performance
- **Batch processing**: Amortize overhead across operations

### Target Use Cases
- Vector databases (Qdrant, Milvus, Weaviate)
- Semantic search applications
- Recommendation systems
- RAG (Retrieval-Augmented Generation) pipelines
- Real-time embedding search

## Important Constraints

### Technical Constraints
- **Rust Edition 2024**: Required for latest features
- **Nightly toolchain**: Needed for experimental features
- **GPU compute capability**:
  - NVIDIA: 7.0+ (Volta and newer)
  - AMD: gfx900+ (Vega and newer)
  - Apple: M1 and newer
- **VRAM-only**: No CPU-GPU transfer during search
- **Single-precision**: f32 only (f16 planned for Phase 4)

### Business Constraints
- **Open source**: MIT/Apache-2.0 dual license
- **Production-ready by v1.0.0**: Full reliability features
- **Backward compatibility**: Maintain after v1.0.0

### Regulatory Constraints
- None currently

## External Dependencies

### GPU Drivers and Toolkits
- **macOS**: Xcode Command Line Tools (for Metal)
- **NVIDIA**: CUDA Toolkit 12.0+
- **AMD**: ROCm 5.0+

### System Libraries
- **CUDA**: cudart, cublas
- **ROCm**: HIP runtime, rocBLAS

### Cloud Providers (for CI/CD)
- **GitHub Actions**: Primary CI/CD
- **Docker**: nvidia/cuda and rocm containers for testing

## Development Phases

### Phase 1: Foundation ✅ (v0.1.x - COMPLETE)
- Core types and traits
- Metal Native backend
- Basic documentation
- Initial tests

### Phase 2: Device Info API 🔥 (v0.1.7 - NEXT, 1-2 days)
- `GpuDeviceInfo` struct
- Device query methods
- Metal implementation
- **Dependency for Phase 3**

### Phase 3: Multi-Backend Support 🔥 (v0.2.x, 4-6 weeks)
#### 3.1: CUDA Backend (v0.2.0, 1-2 weeks)
- CUDA context and storage
- CUDA kernels and cuBLAS
- **70% market coverage**

#### 3.2: ROCm Backend (v0.2.1, 1-2 weeks)
- ROCm/HIP implementation
- HIP kernels and rocBLAS
- **90% total market coverage**

### Phase 4: Advanced Features ⚡ (v0.3.0, 3-4 months)
- Vector quantization (PQ, SQ, Binary)
- HNSW optimizations
- Memory pooling
- Performance monitoring

### Phase 5: Production Ready 📦 (v1.0.0, 2-3 months)
- Reliability features
- Integrations (vectorizer, Qdrant, etc.)
- Observability
- Full documentation

## OpenSpec Workflow

### Creating Changes
1. Check existing specs: `openspec list --specs`
2. Check active changes: `openspec list`
3. Create proposal for:
   - New features/capabilities
   - Breaking changes
   - Architecture changes
4. Skip proposal for:
   - Bug fixes (restore intended behavior)
   - Typos, formatting, comments

### Proposal Structure
- `proposal.md` - Why, what, impact
- `tasks.md` - Implementation checklist
- `design.md` - Technical decisions (optional)
- `specs/[capability]/spec.md` - Delta changes

### Implementation
1. Read proposal and tasks
2. Implement sequentially
3. Run AGENT_AUTOMATION workflow after each feature
4. Update tasks.md with completion status
5. Get approval before proceeding

### Archiving
After deployment:
```bash
openspec archive <change-id> --yes
```

## Quality Standards

### Pre-Commit Checklist
- [ ] `cargo fmt --all`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --all-features` (100% pass)
- [ ] Coverage ≥95%
- [ ] `cargo build --release`
- [ ] Documentation builds without warnings

### Pre-Release Checklist
- [ ] All quality checks passed
- [ ] Performance benchmarks run
- [ ] Security audit clean (`cargo audit`)
- [ ] CHANGELOG.md updated
- [ ] Documentation complete
- [ ] OpenSpec tasks archived

## Market Context

### Current Coverage
- **Phase 1 (Metal only)**: ~5% of production servers
- **Phase 3.1 (+ CUDA)**: ~75% of production servers
- **Phase 3.2 (+ ROCm)**: ~90% of production servers

### Target Users
- **Primary**: ML/AI engineers, data scientists
- **Secondary**: Vector database developers
- **Tertiary**: DevOps/infrastructure teams

### Competitive Landscape
- **FAISS**: CPU-focused, some GPU support (CUDA only)
- **hnswlib**: CPU-only, very fast
- **Milvus/Qdrant**: Full databases with their own GPU implementations
- **Our niche**: Standalone library, vendor-neutral, Rust-native

## Success Metrics

### Performance (vs CPU baseline)
- Vector operations: 10-50x speedup
- Search latency: <3ms (vs 10-30ms CPU)
- Throughput: >10K vectors/second

### Adoption
- GitHub stars: Target 1K+ by v1.0.0
- Crate downloads: Target 10K+ downloads/month
- Production deployments: Target 10+ enterprises

### Code Quality
- Test coverage: ≥95%
- Zero critical security issues
- Documentation completeness: 100%
