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
///
/// Provides detailed information about the GPU device including memory,
/// capabilities, and backend-specific details.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(all(target_os = "macos", feature = "metal-native"))]
/// # {
/// use hive_gpu::metal::MetalNativeContext;
/// use hive_gpu::traits::GpuContext;
///
/// let context = MetalNativeContext::new().expect("Failed to create Metal context");
/// let info = context.device_info().expect("Failed to get device info");
///
/// println!("Device: {}", info.name);
/// println!("Backend: {}", info.backend);
/// println!("VRAM: {} MB", info.total_vram_bytes / 1024 / 1024);
/// println!("Usage: {:.1}%", info.vram_usage_percent());
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device name (e.g., "Apple M2 Pro", "NVIDIA RTX 4090")
    pub name: String,

    /// Backend type (e.g., "Metal", "CUDA", "ROCm")
    pub backend: String,

    /// Total VRAM in bytes
    pub total_vram_bytes: u64,

    /// Currently available VRAM in bytes
    pub available_vram_bytes: u64,

    /// Currently used VRAM in bytes (calculated as total - available)
    pub used_vram_bytes: u64,

    /// Driver version string (e.g., "macOS 14.1", "CUDA 12.0", "ROCm 5.4")
    pub driver_version: String,

    /// Compute capability or architecture version
    /// - Metal: None
    /// - CUDA: e.g., "8.9" for sm_89
    /// - ROCm: e.g., "gfx1030"
    pub compute_capability: Option<String>,

    /// Maximum threads per block/workgroup
    pub max_threads_per_block: u32,

    /// Maximum shared memory per block (in bytes)
    pub max_shared_memory_per_block: u64,

    /// Device ID (0-indexed)
    pub device_id: i32,

    /// PCI bus ID (e.g., "0000:01:00.0")
    /// None for Metal (Apple Silicon doesn't expose PCI)
    pub pci_bus_id: Option<String>,
}

impl GpuDeviceInfo {
    /// Calculate VRAM usage percentage (0.0 to 100.0)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hive_gpu::types::GpuDeviceInfo;
    /// # let info = GpuDeviceInfo {
    /// #     name: "Test GPU".to_string(),
    /// #     backend: "Test".to_string(),
    /// #     total_vram_bytes: 16 * 1024 * 1024 * 1024,
    /// #     available_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     used_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     driver_version: "1.0".to_string(),
    /// #     compute_capability: None,
    /// #     max_threads_per_block: 1024,
    /// #     max_shared_memory_per_block: 49152,
    /// #     device_id: 0,
    /// #     pci_bus_id: None,
    /// # };
    /// let usage = info.vram_usage_percent();
    /// assert!(usage >= 0.0 && usage <= 100.0);
    /// ```
    pub fn vram_usage_percent(&self) -> f64 {
        if self.total_vram_bytes == 0 {
            return 0.0;
        }
        (self.used_vram_bytes as f64 / self.total_vram_bytes as f64) * 100.0
    }

    /// Check if there is sufficient available VRAM
    ///
    /// # Arguments
    ///
    /// * `required_bytes` - Minimum required VRAM in bytes
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hive_gpu::types::GpuDeviceInfo;
    /// # let info = GpuDeviceInfo {
    /// #     name: "Test GPU".to_string(),
    /// #     backend: "Test".to_string(),
    /// #     total_vram_bytes: 16 * 1024 * 1024 * 1024,
    /// #     available_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     used_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     driver_version: "1.0".to_string(),
    /// #     compute_capability: None,
    /// #     max_threads_per_block: 1024,
    /// #     max_shared_memory_per_block: 49152,
    /// #     device_id: 0,
    /// #     pci_bus_id: None,
    /// # };
    /// // Check if we have at least 1GB available
    /// if info.has_available_vram(1 * 1024 * 1024 * 1024) {
    ///     println!("Sufficient VRAM available");
    /// }
    /// ```
    pub fn has_available_vram(&self, required_bytes: u64) -> bool {
        self.available_vram_bytes >= required_bytes
    }

    /// Get VRAM available in megabytes (convenience method)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hive_gpu::types::GpuDeviceInfo;
    /// # let info = GpuDeviceInfo {
    /// #     name: "Test GPU".to_string(),
    /// #     backend: "Test".to_string(),
    /// #     total_vram_bytes: 16 * 1024 * 1024 * 1024,
    /// #     available_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     used_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     driver_version: "1.0".to_string(),
    /// #     compute_capability: None,
    /// #     max_threads_per_block: 1024,
    /// #     max_shared_memory_per_block: 49152,
    /// #     device_id: 0,
    /// #     pci_bus_id: None,
    /// # };
    /// println!("Available: {} MB", info.available_vram_mb());
    /// ```
    pub fn available_vram_mb(&self) -> u64 {
        self.available_vram_bytes / (1024 * 1024)
    }

    /// Get total VRAM in megabytes (convenience method)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use hive_gpu::types::GpuDeviceInfo;
    /// # let info = GpuDeviceInfo {
    /// #     name: "Test GPU".to_string(),
    /// #     backend: "Test".to_string(),
    /// #     total_vram_bytes: 16 * 1024 * 1024 * 1024,
    /// #     available_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     used_vram_bytes: 8 * 1024 * 1024 * 1024,
    /// #     driver_version: "1.0".to_string(),
    /// #     compute_capability: None,
    /// #     max_threads_per_block: 1024,
    /// #     max_shared_memory_per_block: 49152,
    /// #     device_id: 0,
    /// #     pci_bus_id: None,
    /// # };
    /// println!("Total: {} MB", info.total_vram_mb());
    /// ```
    pub fn total_vram_mb(&self) -> u64 {
        self.total_vram_bytes / (1024 * 1024)
    }
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

/// Configuration for an IVF (Inverted File) index.
///
/// IVF partitions the vector space into `n_list` Voronoi cells via k-means.
/// At query time, only the `nprobe` cells closest to the query are searched,
/// making search cost roughly `O(n_list + nprobe * N / n_list)` instead of
/// `O(N)`. Higher `nprobe` yields higher recall at the cost of latency.
///
/// Defaults follow the FAISS convention of `n_list ≈ sqrt(N)` and
/// `nprobe = n_list / 16` for a ~0.95 recall@10 starting point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfConfig {
    /// Number of Voronoi cells (centroids).
    pub n_list: usize,
    /// Number of cells to probe per query.
    pub nprobe: usize,
    /// Upper bound on training-sample size used by k-means. The effective
    /// sample is `min(training_sample_size, input.len())`.
    pub training_sample_size: usize,
    /// Maximum Lloyd iterations during training.
    pub kmeans_iters: usize,
    /// Random seed for k-means++ initialisation. `None` ⇒ non-deterministic.
    pub seed: Option<u64>,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_list: 1024,
            nprobe: 64,
            training_sample_size: 256 * 1024,
            kmeans_iters: 20,
            seed: None,
        }
    }
}

impl IvfConfig {
    /// Heuristic `n_list ≈ sqrt(N)` clamped to a reasonable range.
    #[must_use]
    pub fn for_dataset_size(n: usize) -> Self {
        let n_list = (n as f64).sqrt().ceil() as usize;
        let n_list = n_list.clamp(16, 65_536);
        let nprobe = (n_list / 16).max(1);
        Self {
            n_list,
            nprobe,
            training_sample_size: (256 * n_list).min(n),
            kmeans_iters: 20,
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
