## 1. Dependencies
- [ ] 1.1 Add `objc2-metal-performance-shaders` as optional dep in `Cargo.toml`, target-gated to `cfg(target_os = "macos")` and wired into `metal-native` feature
- [ ] 1.2 Verify `cargo check --features metal-native` still passes on macOS
- [ ] 1.3 Verify `cargo check --no-default-features` still passes on non-macOS hosts

## 2. Norm Cache
- [ ] 2.1 Add `norms_sq: Vec<f32>` field to `MetalNativeVectorStorage`
- [ ] 2.2 Push squared L2 norm during `add_vector` and `add_vectors` batch path
- [ ] 2.3 Clear the norm cache in `clear()`
- [ ] 2.4 Ensure buffer expansion preserves norms alignment with device data

## 3. GPU Search via MPS
- [ ] 3.1 Replace the mock loop in `src/metal/vector_storage.rs::search()` with a real dispatch
- [ ] 3.2 Upload the query vector into a transient `MTLBuffer` (shared mode)
- [ ] 3.3 Allocate a scores `MTLBuffer` of length `vector_count * sizeof::<f32>`
- [ ] 3.4 Dispatch `MPSMatrixVectorMultiplication` with transpose=true so output `y[i] = v_i . query`
- [ ] 3.5 Read scores back via blit encoder into a host-visible buffer
- [ ] 3.6 Apply metric post-processing (Cosine normalise, Euclidean derive ||v-q||^2) on CPU
- [ ] 3.7 Respect `removed_indices` when selecting top-K on CPU

## 4. Hygiene
- [ ] 4.1 Remove the mock-score comment and the placeholder marker from `search()`
- [ ] 4.2 Ensure no panic paths; map every MPS / Metal error to `HiveGpuError`
- [ ] 4.3 Re-run `cargo clippy --features metal-native -- -D warnings` to zero warnings
- [ ] 4.4 Re-run `cargo fmt --all --check`

## 5. Tests
- [ ] 5.1 `tests/metal_bruteforce_smoke.rs` mirroring `tests/cuda_smoke.rs` (context info, Cosine, Euclidean, buffer growth)
- [ ] 5.2 `tests/metal_search_accuracy.rs` validating numerical agreement with a CPU reference within 1e-3 over 1000 random queries
- [ ] 5.3 Ensure the existing 72-test Metal suite still passes
- [ ] 5.4 Gate every test behind a graceful exit when Metal is unavailable, same pattern the CUDA suite uses

## 6. Benchmarks
- [ ] 6.1 Extend `benches/gpu_operations.rs` with a real `search_bruteforce` group comparing GPU vs CPU
- [ ] 6.2 Capture baseline numbers on an Apple Silicon host (M1 Pro or M3 Pro class)
- [ ] 6.3 Record the numbers in `docs/benchmarks/PERFORMANCE.md`, replacing the previously fabricated Metal search table

## 7. Tail (mandatory — enforced by rulebook v5.3.0)
- [ ] 7.1 Update or create documentation covering the implementation
- [ ] 7.2 Write tests covering the new behavior
- [ ] 7.3 Run tests and confirm they pass
