# Memory Pool Specification

## ADDED Requirements

### Requirement: GPU Memory Pooling
The system SHALL provide a memory pool for reusing GPU buffers to reduce allocation overhead.

#### Scenario: Allocate buffer from pool
- **WHEN** a user requests a GPU buffer from the pool
- **THEN** the pool checks if a free buffer is available
- **AND** returns a reused buffer if available (pool hit)
- **AND** allocates a new buffer if pool is empty (pool miss)
- **AND** tracks statistics for hits and misses

#### Scenario: Return buffer to pool
- **WHEN** a user returns a buffer to the pool
- **THEN** the buffer is marked as available
- **AND** added to the free buffer queue
- **AND** remains allocated for future reuse
- **AND** statistics are updated

#### Scenario: Pool size limit enforcement
- **WHEN** the pool reaches max_buffers capacity
- **THEN** new allocation requests create buffers outside the pool
- **AND** those buffers are freed immediately when returned
- **AND** pool size never exceeds the configured limit
- **AND** applications gracefully handle pool exhaustion

### Requirement: Pool Configuration
Users SHALL be able to configure pool behavior through PoolConfig.

#### Scenario: Create pool with custom configuration
- **WHEN** a user creates a context with pool configuration
- **THEN** the pool is initialized with specified buffer_size
- **AND** pool capacity is set to max_buffers
- **AND** pool pre-allocates buffers if pre_allocate > 0
- **AND** pool is enabled if enable flag is true

#### Scenario: Disable pooling
- **WHEN** a user creates a context without pool configuration
- **THEN** all allocations go directly to GPU (no pooling)
- **AND** no pool overhead is incurred
- **AND** behavior is identical to pre-pool implementation

### Requirement: Pool Statistics
The pool SHALL track usage statistics for monitoring and optimization.

#### Scenario: Query pool statistics
- **WHEN** a user calls `pool_statistics()`
- **THEN** the system returns current statistics including:
  - Total allocations requested
  - Pool hits (buffers reused)
  - Pool misses (new allocations)
  - Current number of buffers in pool
  - Peak pool size reached
  - Total memory currently managed by pool

#### Scenario: Calculate pool hit rate
- **WHEN** pool statistics are available
- **THEN** hit rate can be calculated as `hits / (hits + misses)`
- **AND** high hit rate (>80%) indicates effective pooling
- **AND** low hit rate suggests pool misconfiguration

### Requirement: Thread Safety
Memory pool operations SHALL be thread-safe for concurrent access.

#### Scenario: Concurrent allocations
- **WHEN** multiple threads request buffers simultaneously
- **THEN** the pool uses synchronization (Mutex/RwLock)
- **AND** each thread receives a unique buffer
- **AND** no buffer is returned to multiple threads
- **AND** no race conditions occur

#### Scenario: Concurrent returns
- **WHEN** multiple threads return buffers simultaneously
- **THEN** all buffers are safely added to free queue
- **AND** no buffers are lost or duplicated
- **AND** pool state remains consistent

### Requirement: Memory Safety
Pool SHALL prevent memory leaks and use-after-free errors.

#### Scenario: Automatic cleanup on drop
- **WHEN** a pool goes out of scope
- **THEN** all buffers in the pool are freed
- **AND** GPU memory is properly released
- **AND** no memory leaks occur

#### Scenario: Buffer lifecycle management
- **WHEN** a buffer is allocated from pool
- **THEN** buffer is marked as in-use
- **AND** buffer cannot be returned to another user until returned to pool
- **AND** double-free is prevented through tracking

### Requirement: Backend Integration
Pool SHALL integrate with all GPU backends (Metal, CUDA, ROCm).

#### Scenario: Metal context with pool
- **WHEN** MetalNativeContext is created with pool config
- **THEN** pool uses MTLBuffer for backend-specific buffers
- **AND** Metal allocations use pool when enabled
- **AND** fallback to direct Metal allocation if pool exhausted

#### Scenario: CUDA context with pool
- **WHEN** CudaContext is created with pool config
- **THEN** pool uses cudaMalloc/cudaFree internally
- **AND** CUDA allocations use pool when enabled
- **AND** integrates with CUDA streams for async operations

#### Scenario: ROCm context with pool
- **WHEN** RocmContext is created with pool config
- **THEN** pool uses hipMalloc/hipFree internally
- **AND** HIP allocations use pool when enabled
- **AND** integrates with HIP streams for async operations

### Requirement: Performance Characteristics
Pool SHALL reduce allocation overhead and improve throughput.

#### Scenario: Reduced allocation latency
- **WHEN** pool is enabled and warmed up
- **THEN** buffer allocation latency is <0.01ms (vs 0.1-1ms for new allocation)
- **AND** throughput improves by 50-80% for batch operations
- **AND** CPU time spent in allocator is minimized

#### Scenario: Memory efficiency
- **WHEN** pool is properly configured
- **THEN** memory fragmentation is reduced
- **AND** total memory usage is predictable
- **AND** peak memory usage does not exceed pool capacity + overhead

## Implementation Notes

**Core API**:
```rust
pub struct PoolConfig {
    pub buffer_size: usize,    // Size of each buffer
    pub max_buffers: usize,    // Maximum pool size
    pub pre_allocate: usize,   // Warm-up count
    pub enable: bool,          // Enable/disable
}

pub struct GpuMemoryPool {
    // Thread-safe internal state
}

impl GpuMemoryPool {
    pub fn new(config: PoolConfig) -> Self;
    pub fn allocate(&self) -> Result<GpuBuffer>;
    pub fn deallocate(&self, buffer: GpuBuffer);
    pub fn statistics(&self) -> PoolStatistics;
}
```

**Integration with Contexts**:
```rust
impl MetalNativeContext {
    pub fn with_pool(config: PoolConfig) -> Result<Self>;
}
```

**Performance Targets**:
- Pool allocation: <0.01ms
- Pool hit rate: >80% for typical workloads
- Throughput improvement: 50-80% for batch operations
- Memory overhead: <5% of total managed memory

**Configuration Recommendations**:
- buffer_size: Match typical vector size * dimension
- max_buffers: 10-50 for most workloads
- pre_allocate: 5-10 for reduced startup latency

**Testing Requirements**:
- Unit tests for all pool operations
- Thread safety tests with concurrent access
- Integration tests with all backends
- Performance benchmarks showing improvements
- Memory leak detection
- Pool exhaustion handling

