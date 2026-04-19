# CUDA IVF Index Specification

## ADDED Requirements

### Requirement: IVF Configuration Type

The crate MUST expose an `IvfConfig` struct defining the IVF index
hyperparameters with documented defaults.

#### Scenario: IvfConfig default values

Given a call to `IvfConfig::default()`
When the returned config is inspected
Then `n_list` defaults to a non-zero positive value chosen via `sqrt(N)`
heuristic documented in the crate
And `nprobe` defaults to `n_list / 16`
And `kmeans_iters` defaults to 20
And `training_sample_size` defaults to the lesser of 256 * n_list and the
full dataset size

### Requirement: K-Means Training Convergence

`CudaIvfIndex::train` SHALL run k-means++ initialization followed by Lloyd
iterations until inertia stops improving or `kmeans_iters` is reached.

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
And the total centroid count remains equal to `n_list`

### Requirement: Cluster Assignment at Add Time

`add_vectors` SHALL assign each new vector to its nearest centroid and
update the inverted lists in device memory.

#### Scenario: Each added vector lives in exactly one cluster

Given a trained `CudaIvfIndex` with `n_list` centroids
When `add_vectors(&batch)` returns
Then the sum of cluster sizes equals the total stored vector count
And `cluster_offsets[i + 1] - cluster_offsets[i]` matches the number of
vectors assigned to cluster `i`
And every entry in `cluster_member_indices` points at a valid vector

### Requirement: Coarse Cluster Selection

`CudaIvfIndex::search` SHALL compute query-to-centroid scores via cuBLAS
SGEMV and probe the top-`nprobe` clusters only.

#### Scenario: nprobe clusters are probed

Given a `CudaIvfIndex` with `n_list = 256` and `nprobe = 16`
When `search(&query, 10)` runs
Then a single SGEMV is dispatched against the centroids matrix
And exactly 16 inverted lists are subsequently searched
And clusters beyond the top-16 by coarse score are not accessed

### Requirement: Refined Per-Cluster Search

For each probed cluster, the search path SHALL reuse the same cuBLAS SGEMV
primitive the brute-force backend uses, over the cluster's contiguous
index subrange.

#### Scenario: Per-cluster SGEMV uses contiguous ranges

Given a cluster `i` with `cluster_offsets[i] = start` and
`cluster_offsets[i + 1] = end`
When the search path enters that cluster
Then SGEMV runs on the flat vector buffer starting at offset
`start * dimension` for `end - start` rows
And no scatter/gather into a temporary buffer is required

### Requirement: Recall Guarantee

The IVF index MUST achieve documented recall targets against a brute-force
reference on standard datasets.

#### Scenario: Recall@10 at nprobe = n_list / 16

Given a dataset of 1 M random 128-dim vectors
And a trained `CudaIvfIndex` with `n_list = 1024` and `nprobe = 64`
When 1000 random queries are searched with `k = 10`
Then the mean recall@10 compared to a brute-force reference is at least
0.95

#### Scenario: Recall@10 at nprobe = n_list / 4

Given the same dataset and a trained `CudaIvfIndex` with `nprobe = 256`
When the same 1000 queries are searched
Then the mean recall@10 is at least 0.99

### Requirement: Latency Scaling

Search latency SHALL grow sub-linearly with dataset size at fixed
`nprobe`.

#### Scenario: Latency stays within the sub-linear envelope

Given benchmarks run at 100 K, 1 M, and 10 M vectors with fixed `n_list =
sqrt(N)` and `nprobe = n_list / 16`
When latencies are recorded on the reference NVIDIA host
Then the ratio of latency(10 M) to latency(100 K) is under 10×
And the ratio of latency(1 M) to latency(100 K) is under 4×

### Requirement: Runtime nprobe Tuning

`CudaIvfIndex::set_nprobe` SHALL allow query-time adjustment of `nprobe`
without rebuilding the index.

#### Scenario: nprobe change affects next search

Given a trained `CudaIvfIndex` with `nprobe = 16`
When `set_nprobe(64)` is called
And `search(&query, 10)` runs afterward
Then exactly 64 clusters are probed on that search
And training and inverted lists are untouched

### Requirement: Metric Coverage

IVF SHALL support all three metrics the crate exposes: DotProduct, Cosine,
and Euclidean (L2).

#### Scenario: Euclidean uses cached norms

Given an IVF index with metric Euclidean and cached vector and centroid
squared norms
When a search is performed
Then the backend computes `||v - q||^2 = ||v||^2 - 2 v·q + ||q||^2` from
the cached norms
And no redundant full distance kernel is invoked

#### Scenario: Cosine normalises with cached norms

Given an IVF index with metric Cosine
When a search is performed
Then the final score per candidate equals
`dot(v, q) / (||v|| * ||q||)` within 1e-4 of a CPU reference

### Requirement: Soft-Deletion Compatibility

`remove_vectors` SHALL continue to work on IVF results and MUST NOT yield
removed items in search output.

#### Scenario: Removed vector never appears in IVF results

Given an `CudaIvfIndex` with vector "x" removed
When a query whose nearest cluster contains "x" is searched
Then "x" does not appear in the returned results

### Requirement: Feature Flag Isolation

The IVF code MUST compile only when the `cuda` feature is enabled on a
supported target OS, mirroring the brute-force backend gating.

#### Scenario: Build without CUDA feature

Given a project depending on `hive-gpu` without the `cuda` feature
When the project is compiled on any OS
Then no IVF code is pulled in
And no cudarc dependency is linked

## MODIFIED Requirements

### Requirement: GpuCapabilities Advertises IVF Support

`GpuCapabilities` MUST expose whether the backend supports IVF so callers
can pick the right index at runtime.

#### Scenario: CUDA context reports IVF support

Given a `CudaContext` on a host with a CUDA-capable GPU
When `supports_operations()` is inspected after this task lands
Then the returned `GpuCapabilities` indicates IVF support is available
