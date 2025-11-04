# Migration Tasks: metal-rs to objc2-metal

## 1. Preparation and Planning

- [ ] 1.1 Create git tag `pre-objc2-migration` for rollback safety
- [ ] 1.2 Run full test suite baseline with metal-rs
- [ ] 1.3 Capture performance benchmarks baseline
- [ ] 1.4 Review Context7 documentation for objc2-metal patterns
- [ ] 1.5 Document current Metal API usage patterns
- [ ] 1.6 Identify all metal-rs imports across codebase

## 2. Dependency Updates

- [ ] 2.1 Update `Cargo.toml` dependencies:
  - Remove `metal = "0.27"`
  - Remove `objc = "0.2"`
  - Add `objc2-metal = "0.2"`
  - Add `objc2-foundation = "0.2"`
  - Add `objc2 = "0.5"`
- [ ] 2.2 Update feature flags if needed
- [ ] 2.3 Run `cargo update` to fetch new dependencies
- [ ] 2.4 Verify dependency tree with `cargo tree`
- [ ] 2.5 Check for version conflicts

## 3. Core Context Migration (src/metal/context.rs)

- [ ] 3.1 Update imports to objc2-metal
- [ ] 3.2 Migrate `Device` usage to `MTLDevice`
- [ ] 3.3 Migrate `CommandQueue` to objc2 bindings
- [ ] 3.4 Update `Library` compilation to use objc2 patterns
- [ ] 3.5 Migrate GPU family checks to objc2
- [ ] 3.6 Update threadgroup size queries
- [ ] 3.7 Migrate VRAM queries using objc2 methods
- [ ] 3.8 Update device name retrieval
- [ ] 3.9 Test context creation and basic operations
- [ ] 3.10 Verify all device info queries work correctly

## 4. Vector Storage Migration (src/metal/vector_storage.rs)

- [ ] 4.1 Update Buffer creation to use objc2-metal
- [ ] 4.2 Migrate `MTLResourceOptions` usage
- [ ] 4.3 Update `MTLStorageMode` enum usage
- [ ] 4.4 Migrate buffer content writing with objc2 patterns
- [ ] 4.5 Update buffer synchronization if needed
- [ ] 4.6 Migrate command buffer creation
- [ ] 4.7 Update compute pipeline state creation
- [ ] 4.8 Migrate compute command encoding
- [ ] 4.9 Test vector insertion operations
- [ ] 4.10 Test vector search operations
- [ ] 4.11 Verify buffer memory management

## 5. Buffer Pool Migration (src/metal/buffer_pool.rs)

- [ ] 5.1 Update buffer allocation with objc2-metal
- [ ] 5.2 Migrate buffer reuse logic
- [ ] 5.3 Update buffer tracking structures
- [ ] 5.4 Test buffer pool operations
- [ ] 5.5 Verify no memory leaks

## 6. HNSW Graph Migration (src/metal/hnsw_graph.rs)

- [ ] 6.1 Update Metal buffer usage in HNSW
- [ ] 6.2 Migrate kernel dispatch patterns
- [ ] 6.3 Update threadgroup size calculations
- [ ] 6.4 Test HNSW construction on GPU
- [ ] 6.5 Test HNSW search operations
- [ ] 6.6 Verify graph correctness

## 7. VRAM Monitor Migration (src/metal/vram_monitor.rs)

- [ ] 7.1 Update VRAM query methods
- [ ] 7.2 Migrate memory statistics collection
- [ ] 7.3 Update `recommended_max_working_set_size` usage
- [ ] 7.4 Update `current_allocated_size` usage
- [ ] 7.5 Test memory monitoring accuracy

## 8. Helpers and Utilities (src/metal/helpers.rs)

- [ ] 8.1 Update helper functions to use objc2
- [ ] 8.2 Migrate Metal utility patterns
- [ ] 8.3 Update error handling for objc2
- [ ] 8.4 Test all helper functions

## 9. Backend Detection (src/backends/detector.rs)

- [ ] 9.1 Update Metal device detection
- [ ] 9.2 Update backend capability checks
- [ ] 9.3 Test auto-detection on macOS
- [ ] 9.4 Verify fallback behavior

## 10. Shader Compilation

- [ ] 10.1 Verify Metal shader (.metal files) compile with objc2
- [ ] 10.2 Test shader function loading
- [ ] 10.3 Update compute pipeline creation if needed
- [ ] 10.4 Test all shader kernel dispatches
- [ ] 10.5 Verify shader execution correctness

## 11. Examples and Benchmarks

- [ ] 11.1 Update `examples/metal_basic.rs`
- [ ] 11.2 Update `benches/gpu_operations.rs`
- [ ] 11.3 Test all examples run successfully
- [ ] 11.4 Run benchmarks and compare with baseline

## 12. Testing

- [ ] 12.1 Run unit tests: `cargo test --features metal-native`
- [ ] 12.2 Run integration tests
- [ ] 12.3 Fix any test failures
- [ ] 12.4 Add new tests for objc2-specific patterns
- [ ] 12.5 Verify test coverage maintained at ≥95%
- [ ] 12.6 Test on multiple macOS versions if possible
- [ ] 12.7 Test on different Apple Silicon chips (M1/M2/M3)

## 13. Performance Validation

- [ ] 13.1 Run full benchmark suite
- [ ] 13.2 Compare against baseline metrics
- [ ] 13.3 Investigate any performance regressions
- [ ] 13.4 Document performance changes
- [ ] 13.5 Verify latency targets met (<3ms)
- [ ] 13.6 Verify throughput targets met (>10K ops/sec)

## 14. Documentation

- [ ] 14.1 Create `docs/guides/MIGRATION_METAL_OBJC2.md`
- [ ] 14.2 Update `README.md` with new dependencies
- [ ] 14.3 Update `docs/ARCHITECTURE.md`
- [ ] 14.4 Update API documentation (rustdoc)
- [ ] 14.5 Document objc2 patterns and best practices
- [ ] 14.6 Update `CHANGELOG.md`
- [ ] 14.7 Update examples documentation

## 15. Quality Checks

- [ ] 15.1 Run `cargo fmt --all`
- [ ] 15.2 Run `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] 15.3 Fix all clippy warnings
- [ ] 15.4 Run `cargo build --release`
- [ ] 15.5 Run `cargo doc --no-deps` and fix warnings
- [ ] 15.6 Run `codespell` if configured
- [ ] 15.7 Check for unused dependencies

## 16. Security and Audit

- [ ] 16.1 Run `cargo audit` for vulnerabilities
- [ ] 16.2 Review unsafe code blocks
- [ ] 16.3 Verify memory safety patterns
- [ ] 16.4 Check for resource leaks
- [ ] 16.5 Document security improvements

## 17. Final Validation

- [ ] 17.1 Full CI/CD pipeline passes
- [ ] 17.2 All quality gates passed
- [ ] 17.3 Performance benchmarks acceptable
- [ ] 17.4 Documentation complete and accurate
- [ ] 17.5 No regressions in functionality
- [ ] 17.6 Backward compatibility verified where applicable

## 18. Release Preparation

- [ ] 18.1 Update version to 0.1.8 (or 0.2.0 if breaking)
- [ ] 18.2 Complete CHANGELOG.md entry
- [ ] 18.3 Create git commit with conventional format
- [ ] 18.4 Create git tag for release
- [ ] 18.5 Update ROADMAP.md if needed
- [ ] 18.6 Archive this OpenSpec change

## Notes

**Critical Success Criteria:**
- Zero test failures
- No performance regression
- All documentation updated
- Clean `cargo clippy` with no warnings
- Coverage maintained at ≥95%

**Risk Mitigation:**
- Git tag created for easy rollback
- Incremental migration by module
- Continuous testing after each module
- Performance monitoring throughout

**Estimated Timeline:**
- Days 1-2: Preparation, core context, vector storage (tasks 1-4)
- Day 2: Buffer pool, HNSW, monitoring (tasks 5-7)
- Day 3: Helpers, detection, shaders, examples (tasks 8-11)
- Day 3: Testing, validation, documentation (tasks 12-17)
- Final: Release preparation (task 18)

**Commit Strategy:**
- Commit after each major module migration
- Run quality checks before each commit
- Use conventional commit format
- Reference this OpenSpec change in commits

