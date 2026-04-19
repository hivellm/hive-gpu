# Metal IVF Index Specification

## ADDED Requirements

### Requirement: Metal IVF Public API Parity with CUDA

`MetalIvfIndex` MUST expose the same public API surface as `CudaIvfIndex`
so that `IvfConfig` and index operations are portable between backends.

#### Scenario: Same IvfConfig works on both backends

Given an `IvfConfig` instance built from user code
When the config is passed to both `CudaIvfIndex::new` and
`MetalIvfIndex::new`
Then both constructors accept it without modification
And semantically equivalent indexes are produced on their respective
backends

### Requirement: K-Means Training on Metal

`MetalIvfIndex::train` SHALL run k-means++ initialization and Lloyd
iterations using MPS SGEMM plus the `ivf_argmin` compute shader.

#### Scenario: Inertia is monotonically non-increasing

Given a synthetic dataset with three well-separated clusters
When `train(&vectors, n_iter=20)` is invoked
Then the final inertia is at most the initial inertia
And each iteration's inertia is at most the previous iteration's inertia
within a 1e-6 tolerance

#### Scenario: Empty clusters are reseeded

Given a training run where at least one centroid ends up with zero
assigned points
When the next Lloyd iteration begins
Then the empty centroid is replaced by the point furthest from its
current assignment

### Requirement: Cluster Assignment at Add Time

`MetalIvfIndex::add_vectors` SHALL assign each new vector to its nearest
centroid and update the inverted lists in Metal private memory.

#### Scenario: Inverted lists remain consistent

Given a trained `MetalIvfIndex` with `n_list` centroids
When `add_vectors(&batch)` returns
Then the sum of cluster sizes equals the total stored vector count
And `cluster_offsets_buf[i + 1] - cluster_offsets_buf[i]` matches the
number of vectors assigned to cluster `i`

### Requirement: ivf_argmin Metal Kernel Correctness

The `ivf_argmin` compute shader SHALL produce the same output as a
straightforward CPU argmin over every row of the input score matrix.

#### Scenario: Kernel matches CPU reference

Given a random `(512, 128)` score matrix
When the `ivf_argmin` kernel runs on that matrix
And the same matrix is argmin-reduced row-wise on the CPU
Then the two outputs are identical

### Requirement: Coarse Cluster Selection via MPS SGEMV

`MetalIvfIndex::search` SHALL compute query-to-centroid scores via a
single `MPSMatrixVectorMultiplication` dispatch and select the top-`nprobe`
clusters on the CPU.

#### Scenario: Single coarse SGEMV per search

Given a `MetalIvfIndex` with `n_list = 256` and `nprobe = 16`
When `search(&query, 10)` runs
Then exactly one `MPSMatrixVectorMultiplication` dispatch runs against
the centroids buffer
And exactly 16 cluster subranges are subsequently searched

### Requirement: Refined Per-Cluster Search

Per-probed-cluster search SHALL reuse the same MPS SGEMV primitive the
brute-force Metal backend uses (introduced in
`phase4a_finish-metal-bruteforce-search`), over contiguous subranges of
the flat vector buffer.

#### Scenario: Per-cluster SGEMV uses contiguous ranges

Given a cluster `i` with start / end offsets stored in
`cluster_offsets_buf`
When the search path enters that cluster
Then the refined SGEMV dispatch reads a contiguous `(end - start)` rows
of the flat vector buffer
And no scatter/gather into a temporary buffer is required

### Requirement: Recall Parity with CUDA

Metal IVF SHALL meet the same recall targets the CUDA backend commits to,
on the same dataset.

#### Scenario: Recall@10 at nprobe = n_list / 16

Given 1 M random 128-dim vectors indexed on both Metal and CUDA with
`n_list = 1024`
When 1000 random queries are searched with `nprobe = 64, k = 10`
Then Metal's mean recall@10 is at least 0.95
And Metal's mean recall@10 differs from CUDA's by at most 0.02

### Requirement: Latency Scaling

Search latency SHALL grow sub-linearly with dataset size at fixed
`nprobe`, matching the CUDA backend's scaling envelope.

#### Scenario: Latency stays within the sub-linear envelope

Given benchmarks run at 100 K, 1 M vectors with `nprobe = n_list / 16`
When latencies are recorded on the reference Apple Silicon host
Then the ratio of latency(1 M) to latency(100 K) is under 5×

### Requirement: Runtime nprobe Tuning

`MetalIvfIndex::set_nprobe` SHALL allow query-time adjustment of `nprobe`
without rebuilding the index.

#### Scenario: nprobe change affects next search

Given a trained `MetalIvfIndex` with `nprobe = 16`
When `set_nprobe(64)` is called
And `search(&query, 10)` runs afterward
Then exactly 64 clusters are probed on that search
And training and inverted lists are untouched

### Requirement: Metric Coverage

Metal IVF SHALL support all three metrics: DotProduct, Cosine, and
Euclidean (L2), all implemented through the MPS SGEMV + cached norms
pattern the brute-force path uses.

#### Scenario: Cosine normalises with cached norms on Metal

Given a Metal IVF index with metric Cosine
When a search is performed
Then the final score per candidate equals
`dot(v, q) / (||v|| * ||q||)` within 1e-4 of a CPU reference

### Requirement: Soft-Deletion Compatibility

`remove_vectors` SHALL continue to work on IVF results and MUST NOT yield
removed items in search output.

#### Scenario: Removed vector never appears in Metal IVF results

Given a `MetalIvfIndex` with vector "x" removed
When a query whose nearest cluster contains "x" is searched
Then "x" does not appear in the returned results

### Requirement: Feature and Platform Gating

Metal IVF code MUST compile only when the `metal-native` feature is
enabled on macOS.

#### Scenario: Non-macOS host builds without Metal IVF

Given a Linux or Windows host building with `--features metal-native`
enabled erroneously
When the project compiles
Then the `MetalIvfIndex` module is excluded via `cfg(target_os =
"macos")`
And no Metal symbols leak into the resulting binary

## MODIFIED Requirements

### Requirement: GpuCapabilities Advertises IVF Support on Metal

`GpuCapabilities` MUST report IVF availability on the Metal backend once
this task lands.

#### Scenario: Metal context reports IVF support

Given a `MetalNativeContext` on an Apple Silicon host
When `supports_operations()` is inspected after this task lands
Then the returned `GpuCapabilities` indicates IVF support is available
