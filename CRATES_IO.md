# 📦 Crates.io Publication Guide

## 🚀 Publishing hive-gpu to crates.io

### Prerequisites

1. **Crates.io Account**: Create an account at [crates.io](https://crates.io)
2. **API Token**: Generate a token in your account settings
3. **GitHub Secrets**: Set up the `CARGO_REGISTRY_TOKEN` secret

### Automatic Publication

#### Method 1: GitHub Actions (Recommended)
1. **Set up secrets**:
   - Go to repository Settings → Secrets and variables → Actions
   - Add `CARGO_REGISTRY_TOKEN` with your crates.io token

2. **Create a release**:
   ```bash
   # Create and push a tag
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. **GitHub Actions will automatically**:
   - Run tests
   - Build the package
   - Publish to crates.io
   - Create a GitHub release

#### Method 2: Manual Workflow
1. **Go to Actions tab** in your repository
2. **Select "Release" workflow**
3. **Click "Run workflow"**
4. **Enter version** (e.g., `v0.1.0`)
5. **Click "Run workflow"**

### Manual Publication

#### Using the publish script:
```bash
# Make sure you have the token set
export CARGO_REGISTRY_TOKEN=your_token_here

# Run the publish script
./scripts/publish.sh 0.1.0
```

#### Using cargo directly:
```bash
# Login to crates.io
echo $CARGO_REGISTRY_TOKEN | cargo login

# Update version in Cargo.toml
# Then publish
cargo publish
```

### Version Management

#### Semantic Versioning
- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

#### Pre-release versions
- **0.1.0-alpha.1**: Alpha release
- **0.1.0-beta.1**: Beta release
- **0.1.0-rc.1**: Release candidate

### Package Metadata

The package is configured with:
- **Name**: `hive-gpu`
- **Description**: High-performance GPU acceleration library
- **License**: MIT
- **Repository**: https://github.com/hivellm/hive-gpu
- **Documentation**: https://docs.rs/hive-gpu
- **Keywords**: gpu, metal, cuda, wgpu, vector, similarity, hnsw
- **Categories**: hardware-support, algorithms

### Features

#### Default Features
- None (CPU-only by default)

#### Optional Features
- **metal-native**: Metal Native GPU acceleration (macOS)
- **cuda**: CUDA GPU acceleration (Linux/Windows)
- **wgpu**: wgpu cross-platform GPU acceleration

### Dependencies

#### Core Dependencies
- `tracing`: Logging and debugging
- `serde`: Serialization
- `tokio`: Async runtime

#### GPU Dependencies
- **Metal**: `metal`, `objc` (macOS)
- **CUDA**: `cudarc` (Linux/Windows)
- **wgpu**: `wgpu`, `wgpu-core` (Cross-platform)

### Testing

#### Local Testing
```bash
# Test all features
cargo test --all-features

# Test specific features
cargo test --features metal-native
cargo test --features cuda
cargo test --features wgpu
```

#### CI/CD Testing
- **Ubuntu**: Tests CUDA and wgpu
- **macOS**: Tests Metal Native and wgpu
- **Windows**: Tests wgpu

### Documentation

#### API Documentation
```bash
# Generate docs
cargo doc --all-features --open

# Publish docs to docs.rs (automatic)
```

#### Examples
- **Metal Basic**: `examples/metal_basic.rs`
- **CUDA Basic**: `examples/cuda_basic.rs`
- **wgpu Basic**: `examples/wgpu_basic.rs`

### Troubleshooting

#### Common Issues

1. **Version already exists**:
   ```
   Error: crate `hive-gpu` version 0.1.0 already exists
   ```
   **Solution**: Increment version in Cargo.toml

2. **Authentication failed**:
   ```
   Error: failed to authenticate
   ```
   **Solution**: Check CARGO_REGISTRY_TOKEN

3. **Build failed**:
   ```
   Error: failed to build
   ```
   **Solution**: Run tests locally first

#### Debug Commands
```bash
# Check package
cargo check --all-features

# Dry run publish
cargo publish --dry-run

# Check if version exists
cargo search hive-gpu --limit 1
```

### Post-Publication

#### Verification
1. **Check crates.io**: https://crates.io/crates/hive-gpu
2. **Check docs.rs**: https://docs.rs/hive-gpu
3. **Test installation**: `cargo add hive-gpu`

#### Announcement
- **GitHub Release**: Automatic
- **Twitter**: Manual announcement
- **Reddit**: r/rust community
- **Discord**: Rust community servers

### Maintenance

#### Regular Updates
- **Security updates**: As needed
- **Feature updates**: Monthly
- **Bug fixes**: Weekly
- **Documentation**: Continuous

#### Monitoring
- **Downloads**: crates.io stats
- **Issues**: GitHub issues
- **Discussions**: GitHub discussions
- **Performance**: Benchmarks

---

## 🎯 Quick Start

1. **Set up secrets** in GitHub repository
2. **Create a tag**: `git tag v0.1.0 && git push origin v0.1.0`
3. **Wait for CI/CD** to complete
4. **Verify publication** on crates.io
5. **Test installation**: `cargo add hive-gpu`

That's it! Your crate is now published and available for everyone to use! 🚀
