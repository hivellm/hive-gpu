# Metal Backend Validation Specification

## ADDED Requirements

### Requirement: Real-Hardware Validation on Apple Silicon

The Metal brute-force search path (phase4a) and the Metal IVF index
(phase4c) SHALL be exercised on real Apple Silicon hardware before the
0.3.0 release is tagged.

#### Scenario: Brute-force suite runs to completion

Given an Apple Silicon Mac on macOS 13 or newer with the current stable
Rust toolchain installed
When the maintainer runs `cargo test --features metal-native --test
metal_bruteforce`
Then all six tests in `tests/metal_bruteforce.rs` pass
And no panic surfaces from `src/metal/vector_storage.rs::run_sgemv_dot`

#### Scenario: IVF suite runs to completion

Given the same host and toolchain
When the maintainer runs `cargo test --features metal-native --test
metal_ivf`
Then all five tests in `tests/metal_ivf.rs` pass
And no panic surfaces from `src/metal/ivf.rs::dispatch_sgemm_dot` or
`src/metal/ivf.rs::MetalIvfIndex::build`

#### Scenario: Existing Metal suite unaffected

Given the same host
When the full Metal suite runs via `cargo test --features metal-native`
Then every test that passed on 0.2.0 continues to pass
And no test that previously passed now fails

### Requirement: Numerical Agreement with CPU Reference

The Metal brute-force search SHALL agree with a CPU reference within the
documented tolerance.

#### Scenario: DotProduct GPU-vs-CPU divergence within tolerance

Given 500 random 32-dimensional vectors and a random 32-dimensional query
When `MetalNativeVectorStorage::search` returns the top-10 and the CPU
reference loop returns the same
Then the returned id sets are identical
And per-element score divergence is below 1e-3

#### Scenario: Cosine self-query returns score ≈ 1.0

Given a `MetalNativeVectorStorage` populated with a single non-zero vector
When the same vector is passed as the query under metric Cosine
Then the top-1 result's score is within 1e-4 of 1.0

### Requirement: IVF Recall Parity with CUDA

The Metal IVF index SHALL achieve recall comparable to the CUDA IVF index
on equivalent workloads.

#### Scenario: Random-data recall at nprobe = n_list / 4

Given 5 000 random 32-dimensional vectors indexed with `n_list = 64` and
`nprobe = 16`
When 30 random queries are searched with `k = 10`
Then mean recall@10 against a CPU brute-force reference is at least 0.65
And the observed value is recorded in `docs/benchmarks/PERFORMANCE.md`

#### Scenario: Full-scan recall

Given the same dataset and a trained `MetalIvfIndex`
When `set_nprobe(n_list)` is applied and the 30 queries re-run
Then mean recall@10 is at least 0.90 — effectively the brute-force baseline

### Requirement: Performance Numbers Captured

Real performance measurements SHALL replace the fabricated figures in
`docs/benchmarks/PERFORMANCE.md`.

#### Scenario: Brute-force search latency recorded

Given the Mac validation host
When `cargo bench --features metal-native --bench gpu_operations` completes
Then the Apple Silicon search-latency table in
`docs/benchmarks/PERFORMANCE.md` carries Criterion medians from that run
And the previous synthetic numbers are removed

#### Scenario: IVF head-to-head recorded

Given the same host
When `cargo bench --features metal-native --bench metal_ivf` completes
Then `docs/benchmarks/PERFORMANCE.md` contains a Metal IVF section with
the brute-force vs. IVF head-to-head numbers at 1 M vectors
And the IVF speedup (or lack thereof) is called out explicitly

### Requirement: Quality Gate Parity

All three quality gates in use for the CUDA backend SHALL also be green
for the Metal backend before the release is tagged.

#### Scenario: Clippy, fmt, and doc are green

Given the Mac validation host
When the maintainer runs
`cargo clippy --features metal-native --lib --tests --benches -- -D
warnings`
And `cargo fmt --all --check`
And `cargo doc --no-deps --features metal-native`
Then all three commands exit 0

### Requirement: Release Artifact Updated

A version bump and CHANGELOG entry SHALL ship alongside the validated
Metal functionality.

#### Scenario: Version bump matches scope

Given brute-force passes but IVF does not
When the release lands
Then `Cargo.toml` advances to 0.2.1
And `CHANGELOG.md` carries a `## [0.2.1]` entry describing the
brute-force fix only

#### Scenario: Both Metal paths land together

Given both brute-force and IVF pass on Mac
When the release lands
Then `Cargo.toml` advances to 0.3.0
And `CHANGELOG.md` carries a `## [0.3.0]` entry describing Metal
brute-force, Metal IVF, and the Apple Silicon performance numbers

## MODIFIED Requirements

### Requirement: Performance Documentation Truthfulness

`docs/benchmarks/PERFORMANCE.md` MUST not contain unmeasured numbers for
any backend the crate ships.

#### Scenario: Every row in the document is reproducible

Given a release containing phase4d's validation
When a reader inspects the Metal section of
`docs/benchmarks/PERFORMANCE.md`
Then every latency / throughput figure cites the Criterion bench that
produced it
And the historical "Apple M1 Pro" fabricated table is removed
