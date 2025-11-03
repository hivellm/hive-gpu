# hive-gpu - Development Guide

## Prerequisites

### Required Tools

- **Rust**: Version 1.85+ (nightly toolchain with Edition 2024 support)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  rustup default nightly
  ```

- **Platform-Specific Requirements**:
  - **macOS**: Xcode Command Line Tools (for Metal framework)
    ```bash
    xcode-select --install
    ```
  - **Linux/Windows (NVIDIA)**: CUDA Toolkit 12.0+ (for NVIDIA GPU support)
    ```bash
    # Ubuntu/Debian
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-0
    ```
  - **Linux (AMD)**: ROCm 5.0+ (for AMD GPU support)
    ```bash
    # Ubuntu 22.04
    wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_*.deb
    sudo apt install ./amdgpu-install_*.deb
    sudo amdgpu-install --usecase=rocm
    ```
  - **Windows**: Visual Studio Build Tools + CUDA Toolkit (for NVIDIA GPU support)

- **Development Tools**:
  ```bash
  # Install cargo-nextest for faster test execution
  cargo install cargo-nextest
  
  # Install cargo-llvm-cov for coverage reports
  cargo install cargo-llvm-cov
  
  # Install cargo-audit for security checks
  cargo install cargo-audit
  ```

### System Requirements

- **OS**: macOS 12+ (for Metal), Linux (Ubuntu 20.04+), Windows 10+
- **GPU**: 
  - Apple Silicon (M1/M2/M3/M4) for Metal backend
  - NVIDIA GPU with CUDA compute capability 7.0+ for CUDA backend
  - AMD GPU (Radeon RX 6000+, Instinct MI200+) for ROCm backend
- **Memory**: 8GB+ RAM, 4GB+ VRAM recommended
- **Disk**: 1GB free space for dependencies

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/hivellm/hive-gpu.git
cd hive-gpu
```

### 2. Install Dependencies

```bash
# Install all dependencies
cargo build --all-features

# Or install specific backend only
cargo build --features metal-native  # macOS with Apple Silicon
cargo build --features cuda          # Linux/Windows with NVIDIA GPU
cargo build --features rocm          # Linux with AMD GPU
```

### 3. Verify Installation

```bash
# Run basic tests
cargo test --all-features

# Run Metal-specific tests (macOS only)
cargo test --features metal-native

# Run examples
cargo run --example metal_basic --features metal-native
```

## Development Workflow

### 1. Create Feature Branch

```bash
# Create and checkout feature branch
git checkout -b feature/my-feature-name

# Or create hotfix branch
git checkout -b hotfix/critical-bug-fix
```

### 2. Make Changes

- Follow Rust Edition 2024 best practices
- Write tests first (TDD approach)
- Keep commits focused and atomic
- Update documentation as you go

### 3. Quality Gates (MANDATORY)

**CRITICAL**: Run these checks before EVERY commit:

```bash
# 1. Format check
cargo +nightly fmt --all -- --check

# 2. Linting (NO warnings allowed)
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 3. Run ALL tests (100% pass rate required)
cargo nextest run --all-features
# OR
cargo test --workspace --all-features

# 4. Build verification
cargo build --release --all-features

# 5. Coverage check (95%+ required)
cargo llvm-cov --all --all-features --html
# Check target/llvm-cov/html/index.html

# 6. Security audit
cargo audit

# 7. Check for outdated dependencies
cargo outdated
```

**If ANY check fails:**
- ❌ DO NOT commit
- ❌ Fix issues first
- ❌ Re-run ALL checks

### 4. Run Tests

```bash
# Run all tests with coverage
cargo llvm-cov --all --all-features

# Run specific backend tests
cargo test --features metal-native
cargo test --features cuda
cargo test --features rocm

# Run specific test
cargo test test_gpu_vector_operations

# Run benchmarks
cargo bench --features metal-native
```

### 5. Commit Changes

**Use conventional commit format:**

```bash
git add .
git commit -m "feat(metal): Add HNSW graph construction optimization

- Implement parallel graph layer construction
- Add threadgroup memory optimization
- Include comprehensive tests (98% coverage)
- Update Metal shader documentation

Closes #123"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

### 6. Push Changes (Manual)

```bash
# After ALL quality checks pass
git push origin feature/my-feature-name
```

### 7. Create Pull Request

- Open PR on GitHub
- Fill in PR template
- Link related issues
- Request review from maintainers
- Wait for CI/CD checks to pass

## Project Structure

```
hive-gpu/
├── Cargo.toml              # Package manifest
├── Cargo.lock              # Dependency lock file (COMMIT THIS)
├── README.md               # Project overview
├── CHANGELOG.md            # Version history
├── AGENTS.md              # AI assistant rules
├── LICENSE                # MIT license
├── CONTRIBUTING.md        # Contribution guidelines
├── CODE_OF_CONDUCT.md    # Code of conduct
├── SECURITY.md           # Security policy
│
├── src/                   # Source code
│   ├── lib.rs            # Library root
│   ├── error.rs          # Error handling
│   ├── types.rs          # Core types
│   ├── traits.rs         # Core traits
│   │
│   ├── backends/         # Backend detection
│   │   ├── mod.rs
│   │   └── detector.rs
│   │
│   ├── metal/            # Metal backend (Apple Silicon)
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   ├── vector_storage.rs
│   │   ├── hnsw_graph.rs
│   │   ├── buffer_pool.rs
│   │   ├── vram_monitor.rs
│   │   └── helpers.rs
│   │
│   ├── cuda/             # CUDA backend (NVIDIA)
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   └── ...
│   │
│   ├── rocm/             # ROCm backend (AMD GPUs)
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   └── ...
│   │
│   ├── shaders/          # GPU shaders/kernels
│   │   ├── mod.rs
│   │   ├── metal_hnsw.metal      # Metal Shading Language
│   │   ├── metal_shaders.rs
│   │   ├── cuda_kernels.cu       # CUDA C++
│   │   ├── cuda_shaders.rs
│   │   ├── hip_kernels.hip       # HIP (ROCm)
│   │   └── rocm_shaders.rs
│   │
│   ├── monitoring/       # Performance monitoring
│   │   ├── mod.rs
│   │   ├── performance_monitor.rs
│   │   └── vram_monitor.rs
│   │
│   └── utils/            # Utility functions
│       ├── mod.rs
│       ├── math.rs
│       ├── memory.rs
│       └── timing.rs
│
├── tests/                # Integration tests
│   └── integration_tests.rs
│
├── benches/              # Benchmarks
│   └── gpu_operations.rs
│
├── examples/             # Example code
│   ├── metal_basic.rs
│   ├── cuda_basic.rs
│   └── rocm_basic.rs
│
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── ROADMAP.md
│   ├── DAG.md
│   └── specs/
│
├── openspec/             # OpenSpec specifications
│   ├── project.md
│   ├── changes/
│   └── specs/
│
└── .github/              # GitHub workflows
    └── workflows/
        ├── rust-test.yml
        ├── rust-lint.yml
        └── codespell.yml
```

## Code Style Guidelines

### Rust Best Practices

```rust
// ✅ GOOD: Idiomatic Rust with Edition 2024 features
pub struct GpuVector {
    pub id: String,
    pub data: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

impl GpuVector {
    /// Creates a new GPU vector with the given ID and data.
    ///
    /// # Examples
    ///
    /// ```
    /// let vector = GpuVector::new("vec_1".to_string(), vec![1.0, 2.0, 3.0]);
    /// assert_eq!(vector.dimension(), 3);
    /// ```
    pub fn new(id: String, data: Vec<f32>) -> Self {
        Self {
            id,
            data,
            metadata: HashMap::new(),
        }
    }
}

// ❌ BAD: Unnecessary clones, missing documentation
pub fn process(v: &GpuVector) -> Vec<f32> {
    v.data.clone() // Unnecessary clone
}
```

### Error Handling

```rust
// ✅ GOOD: Use thiserror for custom errors
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HiveGpuError {
    #[error("No GPU device available")]
    NoDeviceAvailable,
    
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ✅ GOOD: Return Result, propagate errors
pub fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>> {
    if vectors.is_empty() {
        return Err(HiveGpuError::InvalidOperation("Empty vector list".into()));
    }
    
    let indices = self.allocate_storage(vectors.len())?;
    Ok(indices)
}

// ❌ BAD: Unwrap in production code
pub fn add_vectors(&mut self, vectors: &[GpuVector]) -> Vec<usize> {
    self.allocate_storage(vectors.len()).unwrap() // NEVER DO THIS
}
```

### Documentation

```rust
// ✅ GOOD: Comprehensive documentation with examples
/// Searches for the k nearest neighbors of a query vector.
///
/// This method performs GPU-accelerated similarity search using the configured
/// distance metric. For HNSW-configured storage, the search uses the HNSW graph
/// for logarithmic time complexity.
///
/// # Arguments
///
/// * `query` - Query vector (must match storage dimension)
/// * `limit` - Maximum number of results to return (k)
///
/// # Returns
///
/// Returns a vector of `GpuSearchResult` sorted by similarity score (descending).
///
/// # Errors
///
/// Returns `HiveGpuError::InvalidDimension` if query dimension doesn't match storage.
/// Returns `HiveGpuError::GpuOperationFailed` if GPU computation fails.
///
/// # Examples
///
/// ```
/// use hive_gpu::metal::context::MetalNativeContext;
/// use hive_gpu::traits::{GpuContext, GpuVectorStorage};
///
/// let context = MetalNativeContext::new()?;
/// let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
/// 
/// // Add vectors...
/// 
/// let query = vec![1.0; 128];
/// let results = storage.search(&query, 10)?;
/// assert!(results.len() <= 10);
/// ```
pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
    // Implementation
}
```

### Async/Await

```rust
// ✅ GOOD: Proper async/await usage
use tokio::time::{timeout, Duration};

pub async fn search_async(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
    timeout(Duration::from_secs(30), async {
        self.search(query, limit)
    }).await
    .map_err(|_| HiveGpuError::Timeout)?
}

// ❌ BAD: Blocking in async context
pub async fn search_async(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
    self.search(query, limit) // Blocks async runtime!
}
```

## Testing Guidelines

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let vector = GpuVector::new("test".to_string(), vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.id, "test");
        assert_eq!(vector.dimension(), 3);
    }

    #[test]
    fn test_vector_normalization() {
        let data = vec![3.0, 4.0];
        let normalized = normalize_vector(&data);
        let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Empty vector")]
    fn test_empty_vector_panic() {
        GpuVector::new("test".to_string(), vec![]);
    }
}
```

### Integration Tests

```rust
// tests/integration_tests.rs
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};

#[test]
fn test_end_to_end_workflow() {
    let context = MetalNativeContext::new().unwrap();
    let mut storage = context.create_storage(4, GpuDistanceMetric::Cosine).unwrap();
    
    // Add vectors
    let vectors = vec![
        GpuVector::new("v1".into(), vec![1.0, 0.0, 0.0, 0.0]),
        GpuVector::new("v2".into(), vec![0.0, 1.0, 0.0, 0.0]),
    ];
    storage.add_vectors(&vectors).unwrap();
    
    // Search
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = storage.search(&query, 2).unwrap();
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "v1");
    assert!(results[0].score > results[1].score);
}
```

### Benchmarks

```rust
// benches/gpu_operations.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};

fn benchmark_vector_search(c: &mut Criterion) {
    let context = MetalNativeContext::new().unwrap();
    let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine).unwrap();
    
    // Setup: Add 10,000 vectors
    let vectors = (0..10000)
        .map(|i| GpuVector::new(
            format!("vec_{}", i),
            (0..128).map(|_| rand::random::<f32>()).collect()
        ))
        .collect::<Vec<_>>();
    storage.add_vectors(&vectors).unwrap();
    
    let query = (0..128).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();
    
    c.bench_function("search_10k_vectors", |b| {
        b.iter(|| {
            storage.search(black_box(&query), 10).unwrap()
        });
    });
}

criterion_group!(benches, benchmark_vector_search);
criterion_main!(benches);
```

## Debugging Tips

### Enable Debug Logging

```bash
# Enable all debug logs
export RUST_LOG=debug

# Enable specific module logs
export RUST_LOG=hive_gpu::metal=debug

# Run with logging
cargo run --example metal_basic
```

### GPU Debugging

**macOS (Metal):**
```bash
# Enable Metal API validation
export MTL_SHADER_VALIDATION=1
export MTL_DEBUG_LAYER=1

# Run with Metal debugger
xcrun metal-debugger -- cargo run --example metal_basic
```

**Linux (CUDA):**
```bash
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# Run with compute-sanitizer
compute-sanitizer --tool memcheck cargo run --example cuda_basic

# Use Nsight Compute for profiling
ncu --set full cargo run --release --example cuda_basic
```

**Linux (ROCm):**
```bash
# Enable HIP debugging
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_DEBUG=1

# Run with rocgdb
rocgdb --args ./target/debug/examples/rocm_basic

# Profile with rocprof
rocprof --stats ./target/release/examples/rocm_basic
```

### Performance Profiling

**macOS:**
```bash
# Profile with Instruments
cargo build --release --example metal_basic
xcrun xctrace record --template 'Metal System Trace' --launch ./target/release/examples/metal_basic
```

**Linux:**
```bash
# Profile with perf
cargo build --release
perf record --call-graph dwarf ./target/release/hive-gpu-bench
perf report
```

## Documentation Updates

**ALWAYS update documentation when:**
- Adding new public APIs
- Changing behavior of existing APIs
- Fixing bugs that affect documented behavior
- Adding new features to roadmap
- Making breaking changes

### Required Updates

1. **Code Documentation**: Update doc comments (`///`)
2. **README.md**: Update for public-facing changes
3. **ARCHITECTURE.md**: Update for architectural changes
4. **ROADMAP.md**: Update milestone progress
5. **CHANGELOG.md**: Add entry for changes

## Contributing Guidelines

### Before Submitting PR

- [ ] All quality checks pass (format, lint, test, coverage)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No compiler warnings
- [ ] Conventional commit messages used
- [ ] Branch rebased on latest main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests pass locally
- [ ] Coverage meets threshold (95%+)

## Related Issues
Closes #123
```

## CI/CD Workflows

### GitHub Actions

All PRs must pass:

1. **Rust Test** (`.github/workflows/rust-test.yml`):
   - Test on ubuntu-latest, macos-latest
   - Test all feature combinations
   - Upload test results

2. **Rust Lint** (`.github/workflows/rust-lint.yml`):
   - Format check: `cargo +nightly fmt --all -- --check`
   - Clippy: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
   - No warnings allowed

3. **Codespell** (`.github/workflows/codespell.yml`):
   - Check for typos in code and documentation
   - Fail on errors

## Publishing to crates.io

### Pre-Release Checklist

- [ ] All tests passing
- [ ] Documentation complete
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Git tag created
- [ ] Security audit clean

### Publishing Process

```bash
# 1. Update version
# Edit Cargo.toml: version = "0.2.0"

# 2. Update CHANGELOG.md
# Add release notes

# 3. Run quality checks
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --all-features
cargo audit

# 4. Dry run
cargo publish --dry-run

# 5. Create git tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# 6. Publish (GitHub Actions will do this automatically on tag push)
# Or manual: cargo publish
```

---
*Last Updated: 2025-01-03*
