# Migration from metal-rs to objc2-metal

## Why

The current implementation uses `metal-rs` (v0.27) with `objc` (v0.2), which has been **officially discontinued** due to lack of maintenance in the Rust Objective-C ecosystem. The upstream maintainers recommend migrating to `objc2` and `objc2-metal` for new developments.

**Critical Issues with Current Setup:**
- ⚠️ **Security risk**: No active maintenance means no security patches
- ⚠️ **Obsolete bindings**: Based on deprecated `objc 0.2` runtime
- ⚠️ **Missing features**: No access to newer Metal APIs and optimizations
- ⚠️ **Community migration**: Ecosystem moving to `objc2` framework

## What Changes

**BREAKING CHANGES:**
- Replace `metal` crate with `objc2-metal` (v0.2+)
- Replace `objc` crate with `objc2` (v0.5+) 
- Add `objc2-foundation` for Foundation types
- Update all Metal API calls to use objc2 bindings
- Modernize unsafe code patterns to follow objc2 safety model

**Dependencies:**
```toml
# Remove
metal = { version = "0.27", optional = true }
objc = { version = "0.2", optional = true }

# Add
objc2-metal = { version = "0.2", optional = true }
objc2-foundation = { version = "0.2", optional = true }
objc2 = { version = "0.5", optional = true }
```

**Benefits:**
- ✅ **Active maintenance**: Regularly updated with latest Metal features
- ✅ **Better safety**: Improved type-safe bindings and lifetime management
- ✅ **Modern Rust**: Follows current Rust patterns and idioms
- ✅ **Future-proof**: Aligned with ecosystem direction
- ✅ **Performance**: Potential optimizations in newer bindings
- ✅ **Metal Performance Shaders**: Full access to MPS framework (148K+ code snippets)

## Impact

**Affected Modules:**
- `src/metal/context.rs` - Metal device and queue management
- `src/metal/vector_storage.rs` - Buffer allocation and management
- `src/metal/buffer_pool.rs` - Buffer pooling implementation
- `src/metal/hnsw_graph.rs` - HNSW graph GPU operations
- `src/metal/vram_monitor.rs` - VRAM monitoring
- `src/metal/helpers.rs` - Metal utilities
- `src/backends/detector.rs` - Metal backend detection
- `examples/metal_basic.rs` - Example code
- `benches/gpu_operations.rs` - Benchmarks

**Affected Specs:**
- `specs/metal-backend/spec.md` (NEW) - Metal backend requirements
- OpenSpec tasks updated for migration tracking

**Backward Compatibility:**
- ⚠️ **Breaking change** for consumers using internal Metal types
- ✅ **Public API unchanged** - GpuContext and GpuVectorStorage traits remain identical
- ✅ **Behavior unchanged** - All functionality preserved
- ✅ **Performance neutral** - No performance regression expected

**Testing Impact:**
- All existing tests must pass without modification
- Add new tests for objc2-specific patterns
- Validate Metal shader compilation with new bindings
- Benchmark to ensure no performance regression

**Documentation Impact:**
- Update README with new dependency information
- Create migration guide for internal API users
- Update architecture documentation
- Document objc2 patterns and best practices

**Timeline:**
- Estimated effort: 2-3 days
- Risk: Low (well-documented migration path)
- Urgency: High (security and maintenance concerns)

**Rollback Plan:**
- Git tag before migration: `pre-objc2-migration`
- Keep metal-rs code in git history
- Document rollback procedure if critical issues found

