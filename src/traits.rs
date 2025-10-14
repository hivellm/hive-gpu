//! Core traits for Hive GPU

use crate::types::{GpuVector, GpuSearchResult, GpuDistanceMetric, GpuDeviceInfo, GpuCapabilities, GpuMemoryStats, HnswConfig};
use crate::error::Result;

/// Core GPU backend trait
pub trait GpuBackend {
    /// Get device information
    fn device_info(&self) -> GpuDeviceInfo;
    
    /// Get device capabilities
    fn supports_operations(&self) -> GpuCapabilities;
    
    /// Get memory statistics
    fn memory_stats(&self) -> GpuMemoryStats;
}

/// GPU vector storage trait
pub trait GpuVectorStorage {
    /// Add vectors to storage
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>>;
    
    /// Search for similar vectors
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>>;
    
    /// Remove vectors by IDs
    fn remove_vectors(&mut self, ids: &[String]) -> Result<()>;
    
    /// Get total vector count
    fn vector_count(&self) -> usize;
    
    /// Get vector dimension
    fn dimension(&self) -> usize;
    
    /// Get vector by ID
    fn get_vector(&self, id: &str) -> Result<Option<GpuVector>>;
    
    /// Clear all vectors
    fn clear(&mut self) -> Result<()>;
}

/// GPU context trait for creating storage
pub trait GpuContext {
    /// Create vector storage with default configuration
    fn create_storage(&self, dimension: usize, metric: GpuDistanceMetric) -> Result<Box<dyn GpuVectorStorage>>;
    
    /// Create vector storage with HNSW configuration
    fn create_storage_with_config(&self, dimension: usize, metric: GpuDistanceMetric, config: HnswConfig) -> Result<Box<dyn GpuVectorStorage>>;
    
    /// Get memory statistics
    fn memory_stats(&self) -> GpuMemoryStats;
    
    /// Get device information
    fn device_info(&self) -> GpuDeviceInfo;
}

/// GPU buffer management trait
pub trait GpuBufferManager {
    /// Allocate buffer with specified size
    fn allocate_buffer(&mut self, size: usize) -> Result<GpuBuffer>;
    
    /// Deallocate buffer
    fn deallocate_buffer(&mut self, buffer: GpuBuffer) -> Result<()>;
    
    /// Get buffer pool statistics
    fn pool_stats(&self) -> BufferPoolStats;
}

/// GPU buffer handle
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Buffer ID
    pub id: usize,
    /// Buffer size in bytes
    pub size: usize,
    /// Buffer type
    pub buffer_type: BufferType,
}

/// Buffer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferType {
    /// Vector data buffer
    Vector,
    /// Metadata buffer
    Metadata,
    /// Temporary computation buffer
    Temporary,
    /// HNSW graph buffer
    HnswGraph,
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    /// Total buffers in pool
    pub total_buffers: usize,
    /// Available buffers
    pub available_buffers: usize,
    /// Pool utilization (0.0-1.0)
    pub utilization: f32,
    /// Total memory allocated
    pub total_memory: usize,
}

/// GPU monitoring trait
pub trait GpuMonitor {
    /// Get VRAM usage statistics
    fn get_vram_stats(&self) -> VramStats;
    
    /// Validate VRAM-only operation
    fn validate_all_vram(&self) -> Result<()>;
    
    /// Generate VRAM report
    fn generate_vram_report(&self) -> String;
}

/// VRAM statistics
#[derive(Debug, Clone)]
pub struct VramStats {
    /// Total VRAM in bytes
    pub total_vram: usize,
    /// Allocated VRAM in bytes
    pub allocated_vram: usize,
    /// Available VRAM in bytes
    pub available_vram: usize,
    /// VRAM utilization (0.0-1.0)
    pub utilization: f32,
    /// Number of VRAM buffers
    pub buffer_count: usize,
}

/// VRAM buffer information
#[derive(Debug, Clone)]
pub struct VramBufferInfo {
    /// Buffer ID
    pub buffer_id: usize,
    /// Buffer size in bytes
    pub size: usize,
    /// Buffer type
    pub buffer_type: BufferType,
    /// Allocation timestamp
    pub allocated_at: u64,
}
