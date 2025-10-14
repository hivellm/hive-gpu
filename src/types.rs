//! Core types for Hive GPU

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A GPU vector with its associated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuVector {
    /// Unique identifier for the vector
    pub id: String,
    /// The vector data (always f32 for compatibility)
    pub data: Vec<f32>,
    /// Optional metadata associated with the vector
    pub metadata: HashMap<String, String>,
}

impl GpuVector {
    /// Create a new GPU vector
    pub fn new(id: String, data: Vec<f32>) -> Self {
        Self {
            id,
            data,
            metadata: HashMap::new(),
        }
    }

    /// Create a new GPU vector with metadata
    pub fn with_metadata(id: String, data: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self { id, data, metadata }
    }

    /// Get the dimension of the vector
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Get memory usage in bytes
    pub fn memory_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>() + self.id.len() + self.metadata.len() * 32 // rough estimate
    }
}

impl From<&GpuVector> for Vec<f32> {
    fn from(v: &GpuVector) -> Self {
        v.data.clone()
    }
}

/// Distance metrics for vector similarity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum GpuDistanceMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
    /// Dot product
    DotProduct,
}

impl std::fmt::Display for GpuDistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuDistanceMetric::Cosine => write!(f, "cosine"),
            GpuDistanceMetric::Euclidean => write!(f, "euclidean"),
            GpuDistanceMetric::DotProduct => write!(f, "dot_product"),
        }
    }
}

/// Search result from GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSearchResult {
    /// Vector ID
    pub id: String,
    /// Similarity score
    pub score: f32,
    /// Vector index in storage
    pub index: usize,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device name
    pub name: String,
    /// Device type (Metal, CUDA, etc.)
    pub device_type: String,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Maximum buffer size
    pub max_buffer_size: u64,
    /// Compute capability
    pub compute_capability: Option<String>,
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapabilities {
    /// Supports HNSW operations
    pub supports_hnsw: bool,
    /// Supports batch operations
    pub supports_batch: bool,
    /// Maximum vector dimension
    pub max_dimension: usize,
    /// Maximum vectors per batch
    pub max_batch_size: usize,
}

/// GPU memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Available memory in bytes
    pub available: usize,
    /// Memory utilization percentage (0.0-1.0)
    pub utilization: f32,
    /// Number of active buffers
    pub buffer_count: usize,
}

/// HNSW configuration for GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bidirectional links created for each node
    pub max_connections: usize,
    /// Size of the dynamic list for nearest neighbors (construction)
    pub ef_construction: usize,
    /// Size of the dynamic list for nearest neighbors (search)
    pub ef_search: usize,
    /// Maximum level in the hierarchy
    pub max_level: usize,
    /// Level assignment multiplier
    pub level_multiplier: f32,
    /// Random seed for level assignment
    pub seed: Option<u64>,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 100,
            ef_search: 50,
            max_level: 8,
            level_multiplier: 0.5,
            seed: None,
        }
    }
}

/// Vector metadata for GPU operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Original vector ID
    pub original_id: String,
    /// Index in storage
    pub index: usize,
    /// Timestamp of creation
    pub timestamp: u64,
}
