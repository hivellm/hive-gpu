# Metal Brute-Force Search Specification

## ADDED Requirements

### Requirement: Real GPU-Computed Scores on Metal

`MetalNativeVectorStorage::search` SHALL compute real distances on the GPU
for every supported metric. Mock score paths MUST be removed.

#### Scenario: Cosine search returns GPU-computed scores

Given a `MetalNativeVectorStorage` populated with three unit vectors in
different directions and metric `Cosine`
When `search(&query_aligned_with_first_vector, 3)` is called
Then the first returned result's score is within 1e-4 of 1.0
And the other two results have cosine scores strictly less than the first
And no result carries the synthetic `1.0 - (i * 0.1)` pattern

#### Scenario: Euclidean search ranks by real distance

Given a `MetalNativeVectorStorage` populated with vectors at distances
1, 3, and 9 from the query and metric `Euclidean`
When `search(&query, 3)` is called
Then the closest vector is returned first
And the farthest is returned last
And the similarity score is monotonically decreasing in distance

### Requirement: MPS-Based SGEMV Dispatch

The implementation SHALL use Metal Performance Shaders
(`MPSMatrixVectorMultiplication`) to compute dot products between the query
and every stored vector in a single dispatch.

#### Scenario: One SGEMV call per search

Given a populated `MetalNativeVectorStorage`
When `search(&query, k)` is invoked
Then exactly one `MPSMatrixVectorMultiplication` dispatch executes
And the output buffer has length equal to the stored vector count
And the command buffer is synchronised before top-K selection on the CPU

### Requirement: CPU-Side Norm Cache

For metrics that require vector norms (Cosine, Euclidean), squared norms
SHALL be computed on the CPU at `add_vectors` time and cached in memory
mirroring the stored index order.

#### Scenario: Norms update with add and clear

Given an empty `MetalNativeVectorStorage`
When 100 vectors are added in a single batch
Then the norm cache contains 100 entries matching the dot product of each
vector with itself
When `clear` is called
Then the norm cache is empty

#### Scenario: Cosine score matches CPU formula

Given a `MetalNativeVectorStorage` with any non-zero vector v and query q
When the backend computes Cosine score
Then the returned score equals `dot(v, q) / (||v|| * ||q||)` within 1e-4

### Requirement: Numerical Agreement with CPU Reference

The Metal GPU result set MUST match a CPU reference implementation within
tolerance.

#### Scenario: GPU top-10 equals CPU top-10 on random data

Given 1000 random 128-dim vectors and a random 128-dim query
When `search(&query, 10)` runs on Metal GPU
And the same computation runs on a naive CPU loop
Then the returned id sets are identical
And per-element score divergence is below 1e-3

### Requirement: Soft-Deleted Vectors Excluded

Vectors marked via `remove_vectors` MUST NOT appear in search results, even
though their distance is still computed on the GPU.

#### Scenario: Removed vectors filtered before top-K

Given a populated storage with vector "a" removed
When a query that would rank "a" as the top match is searched
Then "a" does not appear in the returned results
And `vector_count()` excludes "a" from the live count

### Requirement: Error Propagation

Any failure in MPS dispatch, command buffer commit, or readback SHALL be
returned as `HiveGpuError`. Panics are forbidden on the search hot path.

#### Scenario: Command buffer failure surfaces as error

Given a `MetalNativeVectorStorage` whose command queue refuses to create a
command buffer
When `search` is invoked
Then the call returns `Err(HiveGpuError::_)` rather than panicking

### Requirement: Benchmarks Capture Real Numbers

Apple Silicon performance numbers in `docs/benchmarks/PERFORMANCE.md` MUST
reflect actual measurements, not synthetic throughput.

#### Scenario: Recorded numbers originate from criterion

Given the completion of this task
When `docs/benchmarks/PERFORMANCE.md` is updated
Then the Metal search table contains numbers produced by
`cargo bench --features metal-native --bench gpu_operations`
And the previously fabricated figures are removed

## MODIFIED Requirements

### Requirement: GpuVectorStorage Search Contract

The existing `GpuVectorStorage::search` trait contract MUST continue to be
honored on the Metal backend.

#### Scenario: Search returns at most `limit` results

Given a populated `MetalNativeVectorStorage`
When `search(&query, limit)` is called with any `limit`
Then `results.len() <= limit`

#### Scenario: Empty storage returns empty results

Given an empty `MetalNativeVectorStorage`
When `search(&query, k)` is called for any `k > 0`
Then the result is an empty `Vec`
