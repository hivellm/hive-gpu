# Migration to objc2-metal - Summary

## Overview

This OpenSpec change documents the migration from the deprecated `metal-rs` crate to the modern `objc2-metal` framework for improved safety, maintenance, and future-proofing.

## Files Created

### 1. Proposal (`proposal.md`)
**Purpose**: Justification and impact analysis  
**Key Points**:
- metal-rs is officially discontinued
- Security and maintenance concerns
- 109 detailed migration tasks
- Estimated 2-3 days effort

### 2. Tasks (`tasks.md`)
**Purpose**: Step-by-step implementation checklist  
**Structure**:
- 18 major sections
- 109 individual tasks
- Organized by module and concern
- Includes testing and validation steps

### 3. Specification (`specs/metal-backend/spec.md`)
**Purpose**: Formal requirements for Metal backend  
**Requirements**:
1. Modern Metal Bindings - objc2-metal v0.2+
2. Type-Safe Metal Device Access
3. Safe Buffer Management
4. Command Queue and Execution
5. Shader Library Compilation
6. VRAM Monitoring
7. Backward Compatible Public API
8. Performance Parity
9. Safe Rust Patterns

### 4. Migration Guide (`docs/guides/MIGRATION_METAL_OBJC2.md`)
**Purpose**: Developer documentation  
**Sections**:
- Background and rationale
- Step-by-step migration instructions
- API mapping table (metal-rs → objc2-metal)
- Common patterns and examples
- Testing strategy
- Performance considerations
- Troubleshooting guide

## Key Benefits

### Security & Maintenance
- ✅ Active development and security patches
- ✅ Regular updates with latest Metal features
- ✅ Community support and bug fixes

### Safety & Quality
- ✅ Improved type-safe bindings
- ✅ Better lifetime management
- ✅ Follows modern Rust patterns
- ✅ Reduced unsafe code

### Performance & Features
- ✅ No performance regression
- ✅ Access to Metal Performance Shaders
- ✅ Latest Metal API features
- ✅ Potential optimizations

## Migration Scope

### Affected Modules (9 files)
```
src/metal/
├── context.rs          - Device and queue management
├── vector_storage.rs   - Buffer operations
├── buffer_pool.rs      - Memory pooling
├── hnsw_graph.rs       - Graph operations
├── vram_monitor.rs     - Memory monitoring
└── helpers.rs          - Utilities

src/backends/detector.rs - Backend detection
examples/metal_basic.rs  - Example code
benches/gpu_operations.rs - Benchmarks
```

### Dependencies Changed
```toml
# Removed
metal = "0.27"
objc = "0.2"

# Added
objc2-metal = "0.2"
objc2-foundation = "0.2"
objc2 = "0.5"
```

## API Changes

### Type Naming
- `Device` → `MTLDevice`
- `CommandQueue` → `MTLCommandQueue`
- `Buffer` → `MTLBuffer`
- `Library` → `MTLLibrary`

### Method Naming (Examples)
- `new_buffer()` → `newBufferWithLength_options()`
- `new_command_queue()` → `newCommandQueue()`
- `system_default()` → `MTLCreateSystemDefaultDevice()`

## Backward Compatibility

### ✅ Public API Unchanged
- `GpuContext` trait - identical
- `GpuVectorStorage` trait - identical
- `GpuBackend` trait - identical
- External consumers require no changes

### ⚠️ Internal API Changes
- Metal type names updated
- Method call patterns modernized
- Import statements changed

## Testing Strategy

### Test Coverage
- ✅ All existing tests must pass
- ✅ Add objc2-specific tests
- ✅ Performance benchmarks
- ✅ Integration tests
- ✅ Coverage maintained at ≥95%

### Validation Steps
1. Unit tests per module
2. Integration tests end-to-end
3. Performance benchmarks vs baseline
4. Memory leak detection
5. Cross-platform validation (M1/M2/M3)

## Timeline

### Phase 1: Preparation (Day 1)
- Create git tag for rollback
- Run baseline benchmarks
- Review Context7 documentation

### Phase 2: Core Migration (Days 1-2)
- Update dependencies
- Migrate context, storage, and buffers
- Test continuously

### Phase 3: Validation (Day 2-3)
- HNSW, monitoring, helpers
- Backend detection and examples
- Full test suite

### Phase 4: Documentation (Day 3)
- Update all documentation
- Create migration guide
- Finalize release notes

## Success Criteria

### ✅ Must Have
- [ ] Zero test failures
- [ ] No performance regression
- [ ] All quality checks pass
- [ ] Documentation complete
- [ ] Coverage ≥95%

### ✅ Nice to Have
- Performance improvements
- Cleaner code structure
- Better error messages
- Additional safety guarantees

## Risk Mitigation

### Low Risk Factors
- Well-documented migration path
- Community examples available
- Incremental module migration
- Continuous testing

### Safety Nets
- Git tag for easy rollback
- Incremental commits per module
- Extensive testing at each step
- Performance monitoring throughout

## Next Steps

1. **Get Approval**: Review and approve this proposal
2. **Create Git Tag**: `git tag pre-objc2-migration`
3. **Start Implementation**: Follow tasks.md sequentially
4. **Continuous Testing**: Run tests after each module
5. **Documentation**: Update as you go
6. **Final Validation**: Full quality gate before release

## References

### Documentation
- `proposal.md` - Full justification
- `tasks.md` - 109 implementation tasks
- `specs/metal-backend/spec.md` - Formal requirements
- `docs/guides/MIGRATION_METAL_OBJC2.md` - Developer guide

### External Resources
- [objc2 Documentation](https://docs.rs/objc2/)
- [objc2-metal Documentation](https://docs.rs/objc2-metal/)
- [Context7 objc2 Examples](https://context7.com/madsmtm/objc2)
- [Apple Metal Documentation](https://developer.apple.com/metal/)

---

**Created**: 2025-01-07  
**Status**: Ready for approval  
**Estimated Effort**: 2-3 days  
**Risk Level**: Low  
**Urgency**: High (security/maintenance)

