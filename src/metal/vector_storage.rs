//! # Metal Native Vector Storage
//!
//! High-performance vector storage using Metal GPU acceleration.
//! All vector data is stored in VRAM for maximum efficiency.

use super::context::MetalNativeContext;
use crate::error::{HiveGpuError, Result};
use crate::traits::GpuVectorStorage;
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
    MTLStorageMode,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Vector metadata structure
#[cfg(all(target_os = "macos", feature = "metal-native"))]
#[derive(Debug, Clone)]
pub struct VectorMetadata {
    pub original_id: String,
    pub index: usize,
    pub timestamp: u64,
}

/// Metal Native Vector Storage
#[cfg(all(target_os = "macos", feature = "metal-native"))]
#[derive(Debug)]
pub struct MetalNativeVectorStorage {
    context: Arc<MetalNativeContext>,
    pub vectors_buffer: Retained<ProtocolObject<dyn MTLBuffer>>, // public for IVF cluster views
    metadata_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    vector_count: usize,
    buffer_capacity: usize, // Total capacity in vectors
    dimension: usize,
    metric: GpuDistanceMetric,
    vector_id_map: HashMap<String, usize>,
    index_to_id: Vec<String>,
    vector_metadata: HashMap<String, VectorMetadata>,
    pub removed_indices: HashSet<usize>,
    vector_payloads: HashMap<String, Option<std::collections::HashMap<String, String>>>,
    /// Squared L2 norms per stored vector, kept in host memory. Cached at
    /// `add_vector` time so `search` can derive Cosine/Euclidean scores
    /// without a second kernel pass.
    norms_sq: Vec<f32>,
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
impl MetalNativeVectorStorage {
    /// Create new Metal native vector storage
    pub fn new(
        context: Arc<MetalNativeContext>,
        dimension: usize,
        metric: GpuDistanceMetric,
    ) -> Result<Self> {
        let device = context.device();

        // Calculate initial capacity (minimum 1024 vectors or 1MB worth)
        let min_vectors = 1024;
        let min_bytes = 1024 * 1024; // 1MB
        let min_vectors_by_bytes = min_bytes / (dimension * std::mem::size_of::<f32>());
        let initial_capacity = min_vectors.max(min_vectors_by_bytes);

        let initial_size = initial_capacity
            .checked_mul(dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                HiveGpuError::Other("Initial buffer size calculation overflow".to_string())
            })?;

        // Create vectors buffer (VRAM only, no CPU access)
        let vectors_buffer = device
            .newBufferWithLength_options(initial_size, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| HiveGpuError::Other("Failed to create vectors buffer".to_string()))?;

        // Create metadata buffer (VRAM only)
        let metadata_buffer = device
            .newBufferWithLength_options(
                initial_capacity * 256, // 256 bytes per vector metadata
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| HiveGpuError::Other("Failed to create metadata buffer".to_string()))?;

        debug!(
            "✅ Metal native vector storage created (VRAM only) with capacity: {}",
            initial_capacity
        );

        Ok(Self {
            context,
            vectors_buffer,
            metadata_buffer,
            vector_count: 0,
            buffer_capacity: initial_capacity,
            dimension,
            metric,
            vector_id_map: HashMap::new(),
            index_to_id: Vec::new(),
            vector_metadata: HashMap::new(),
            removed_indices: HashSet::new(),
            vector_payloads: HashMap::new(),
            norms_sq: Vec::new(),
        })
    }

    /// Add vector to storage (VRAM only)
    pub fn add_vector(&mut self, vector: &GpuVector) -> Result<usize> {
        // Validate vector ID is unique
        if self.vector_id_map.contains_key(&vector.id) {
            return Err(HiveGpuError::Other(format!(
                "Vector with ID '{}' already exists",
                vector.id
            )));
        }

        // Validate vector dimension
        if vector.data.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.data.len(),
            });
        }

        // Validate all values are finite (no NaN/Infinity)
        for (i, &value) in vector.data.iter().enumerate() {
            if !value.is_finite() {
                return Err(HiveGpuError::Other(format!(
                    "Vector contains non-finite value at index {}: {}",
                    i, value
                )));
            }
        }

        // Validate ID length
        if vector.id.len() > 256 {
            return Err(HiveGpuError::Other(
                "Vector ID too long (max 256 chars)".to_string(),
            ));
        }

        // Check if we need to expand buffer
        if self.vector_count >= self.buffer_capacity {
            self.expand_buffer()?;
        }

        let device = self.context.device();
        let queue = self.context.command_queue();

        // Upload new vector data directly to existing buffer
        let vector_data = &vector.data;
        let offset = self
            .vector_count
            .checked_mul(self.dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HiveGpuError::Other("Offset calculation overflow".to_string()))?;

        // Create staging buffer for upload
        let staging_size = self
            .dimension
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HiveGpuError::Other("Staging size calculation overflow".to_string()))?;

        let staging_buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    std::ptr::NonNull::new_unchecked(vector_data.as_ptr() as *mut std::ffi::c_void),
                    staging_size,
                    MTLResourceOptions::StorageModeShared, // CPU accessible for upload
                )
                .ok_or_else(|| HiveGpuError::Other("Failed to create staging buffer".to_string()))?
        };

        // Copy from staging to VRAM buffer
        let command_buffer = queue
            .commandBuffer()
            .ok_or_else(|| HiveGpuError::Other("Failed to create command buffer".to_string()))?;

        let blit_encoder = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| HiveGpuError::Other("Failed to create blit encoder".to_string()))?;

        unsafe {
            blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &staging_buffer,
                0,
                &self.vectors_buffer,
                offset,
                staging_size,
            );
        }

        blit_encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Update state with proper ID tracking
        let index = self.vector_count;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Store metadata
        let metadata = VectorMetadata {
            original_id: vector.id.clone(),
            index,
            timestamp,
        };

        self.vector_id_map.insert(vector.id.clone(), index);
        self.index_to_id.push(vector.id.clone());
        self.vector_metadata.insert(vector.id.clone(), metadata);
        self.vector_payloads
            .insert(vector.id.clone(), Some(vector.metadata.clone()));
        self.norms_sq
            .push(vector.data.iter().map(|&x| x * x).sum::<f32>());
        self.vector_count += 1;

        debug!(
            "✅ Vector added to VRAM: {} (total: {}, has_metadata: {})",
            vector.id,
            self.vector_count,
            !vector.metadata.is_empty()
        );
        Ok(index)
    }

    /// Expand buffer with adaptive growth strategy
    fn expand_buffer(&mut self) -> Result<()> {
        let device = self.context.device();
        let queue = self.context.command_queue();

        // Calculate new capacity with adaptive growth
        let growth_factor = if self.buffer_capacity < 1000 {
            2.0 // Double for small buffers
        } else if self.buffer_capacity < 10000 {
            1.5 // 50% growth for medium buffers
        } else {
            1.2 // 20% growth for large buffers
        };

        let new_capacity = (self.buffer_capacity as f32 * growth_factor).ceil() as usize;
        let new_capacity = new_capacity.max(self.vector_count + 1); // Ensure we can fit at least one more

        // Check VRAM limits (conservative 1GB limit)
        let new_size = new_capacity
            .checked_mul(self.dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                HiveGpuError::Other("New buffer size calculation overflow".to_string())
            })?;

        if new_size > 1024 * 1024 * 1024 {
            // 1GB limit
            return Err(HiveGpuError::VramLimitExceeded {
                requested: new_size,
                limit: 1024 * 1024 * 1024,
            });
        }

        info!(
            "🔄 Expanding Metal buffer: {} -> {} vectors ({} MB)",
            self.buffer_capacity,
            new_capacity,
            new_size / 1024 / 1024
        );

        // Create new larger buffer
        let new_vectors_buffer = device
            .newBufferWithLength_options(new_size, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| {
                HiveGpuError::Other("Failed to create new vectors buffer".to_string())
            })?;

        let new_metadata_buffer = device
            .newBufferWithLength_options(new_capacity * 256, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| {
                HiveGpuError::Other("Failed to create new metadata buffer".to_string())
            })?;

        // Copy existing data to new buffer
        let command_buffer = queue
            .commandBuffer()
            .ok_or_else(|| HiveGpuError::Other("Failed to create command buffer".to_string()))?;

        let blit_encoder = command_buffer
            .blitCommandEncoder()
            .ok_or_else(|| HiveGpuError::Other("Failed to create blit encoder".to_string()))?;

        let current_size = self
            .vector_count
            .checked_mul(self.dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HiveGpuError::Other("Current size calculation overflow".to_string()))?;

        unsafe {
            blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &self.vectors_buffer,
                0,
                &new_vectors_buffer,
                0,
                current_size,
            );
        }

        blit_encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        // Replace old buffer with new one
        self.vectors_buffer = new_vectors_buffer;
        self.metadata_buffer = new_metadata_buffer;
        self.buffer_capacity = new_capacity;

        debug!("✅ Metal buffer expanded to {} vectors", new_capacity);
        Ok(())
    }

    /// Get vector by ID
    pub fn get_vector(&self, id: &str) -> Result<Option<GpuVector>> {
        if let Some(&index) = self.vector_id_map.get(id) {
            if self.removed_indices.contains(&index) {
                return Ok(None);
            }

            // Get vector data from VRAM (this is expensive, so we'll return a placeholder)
            // In practice, you'd implement a method to read from VRAM
            let metadata = self.vector_metadata.get(id).cloned();
            let payload = self.vector_payloads.get(id).cloned().flatten();

            if let Some(meta) = metadata {
                // Create a placeholder vector (in practice, you'd read from VRAM)
                let vector = GpuVector {
                    id: meta.original_id,
                    data: vec![0.0; self.dimension], // Placeholder - would read from VRAM
                    metadata: payload.unwrap_or_default(),
                };
                Ok(Some(vector))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Remove vector by ID
    pub fn remove_vector(&mut self, id: &str) -> Result<()> {
        if let Some(&index) = self.vector_id_map.get(id) {
            self.removed_indices.insert(index);
            self.vector_payloads.remove(id);
            debug!("✅ Vector marked as removed: {} (index: {})", id, index);
            Ok(())
        } else {
            Err(HiveGpuError::VectorNotFound(id.to_string()))
        }
    }

    /// Clear all vectors
    pub fn clear(&mut self) -> Result<()> {
        self.vector_count = 0;
        self.vector_id_map.clear();
        self.index_to_id.clear();
        self.vector_metadata.clear();
        self.removed_indices.clear();
        self.vector_payloads.clear();
        self.norms_sq.clear();

        debug!("✅ All vectors cleared from Metal storage");
        Ok(())
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> StorageStats {
        StorageStats {
            vector_count: self.vector_count,
            buffer_capacity: self.buffer_capacity,
            dimension: self.dimension,
            removed_count: self.removed_indices.len(),
            buffer_size_mb: (self.buffer_capacity * self.dimension * std::mem::size_of::<f32>())
                / 1024
                / 1024,
        }
    }

    /// Run the `sgemv_dot` Metal kernel against the full vector buffer.
    /// Returns a host vector of `vector_count` dot products.
    ///
    /// This is the brute-force primitive used by [`GpuVectorStorage::search`]
    /// and — through `pub(crate)` exposure — by the Metal IVF index's
    /// coarse and refined search paths.
    pub(crate) fn gpu_dot_scores(&self, query: &[f32]) -> Result<Vec<f32>> {
        run_sgemv_dot(
            &self.context,
            &self.vectors_buffer,
            /* matrix_element_offset = */ 0,
            self.vector_count,
            self.dimension,
            query,
        )
    }
}

/// Dispatch `sgemv_dot` for a `(n_vectors, dimension)` row-major matrix
/// stored inside `matrix_buffer` starting at `matrix_element_offset` f32s
/// past the buffer's base address. Returns the `n_vectors`-length score
/// vector read back to host memory.
///
/// Shared between `MetalNativeVectorStorage::gpu_dot_scores` and the IVF
/// index's per-cluster refined search, where the cluster subrange starts
/// at a non-zero element offset inside a larger buffer.
#[cfg(all(target_os = "macos", feature = "metal-native"))]
pub(crate) fn run_sgemv_dot(
    context: &MetalNativeContext,
    matrix_buffer: &ProtocolObject<dyn MTLBuffer>,
    matrix_element_offset: usize,
    n_vectors: usize,
    dimension: usize,
    query: &[f32],
) -> Result<Vec<f32>> {
    if n_vectors == 0 {
        return Ok(Vec::new());
    }

    let device = context.device();
    let queue = context.command_queue();
    let pipeline = context.compute_pipeline("sgemv_dot")?;

    // Query buffer (host-visible so we can fill it without a blit).
    let query_bytes = dimension * std::mem::size_of::<f32>();
    let query_buffer = unsafe {
        device
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new_unchecked(query.as_ptr() as *mut std::ffi::c_void),
                query_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| HiveGpuError::Other("Failed to create query buffer".to_string()))?
    };

    // Output scores buffer (shared so we can read back directly).
    let scores_bytes = n_vectors * std::mem::size_of::<f32>();
    let scores_buffer = device
        .newBufferWithLength_options(scores_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| HiveGpuError::Other("Failed to create scores buffer".to_string()))?;

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| HiveGpuError::Other("Failed to create command buffer".to_string()))?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| HiveGpuError::Other("Failed to create compute encoder".to_string()))?;

    encoder.setComputePipelineState(&pipeline);

    // buffer(0): matrix, with optional element offset converted to bytes.
    let matrix_byte_offset = matrix_element_offset * std::mem::size_of::<f32>();
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(matrix_buffer), matrix_byte_offset, 0);
    }
    // buffer(1): query.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&query_buffer), 0, 1);
    }
    // buffer(2): scores.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&scores_buffer), 0, 2);
    }
    // buffer(3): dimension as u32 inline constant.
    let dim_u32 = dimension as u32;
    let n_u32 = n_vectors as u32;
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&dim_u32 as *const u32 as *mut std::ffi::c_void),
            std::mem::size_of::<u32>(),
            3,
        );
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut std::ffi::c_void),
            std::mem::size_of::<u32>(),
            4,
        );
    }

    // Dispatch: one thread per output row. Threadgroup size capped by
    // the pipeline's max; grid size rounded up.
    let max_tgs = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    let tgs = MTLSize {
        width: max_tgs,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: n_vectors,
        height: 1,
        depth: 1,
    };
    // `dispatchThreads_threadsPerThreadgroup` is available on Apple GPUs
    // (requires macOS 10.15+). Intel Macs on macOS 10.13/14 would need
    // the older `dispatchThreadgroups_threadsPerThreadgroup` API; we target
    // Apple Silicon only.
    unsafe {
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tgs);
    }
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Read scores back out of the shared-mode buffer.
    let mut out = vec![0f32; n_vectors];
    // SAFETY: scores_buffer has `scores_bytes` bytes of host-visible
    // memory; we copy exactly that many bytes into a matching f32 slice.
    unsafe {
        let src = scores_buffer.contents().as_ptr() as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n_vectors);
    }
    Ok(out)
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub vector_count: usize,
    pub buffer_capacity: usize,
    pub dimension: usize,
    pub removed_count: usize,
    pub buffer_size_mb: usize,
}

impl GpuVectorStorage for MetalNativeVectorStorage {
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>> {
        let mut indices = Vec::new();
        for vector in vectors {
            let index = self.add_vector(vector)?;
            indices.push(index);
        }
        Ok(indices)
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
        if limit == 0 || self.vector_count == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        for (i, &v) in query.iter().enumerate() {
            if !v.is_finite() {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite query component at index {i}"
                )));
            }
        }

        // Compute raw dot products against every stored vector on the GPU.
        let raw_scores = self.gpu_dot_scores(query)?;

        // Apply the caller's metric on the host using cached squared norms.
        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        let mut scored: Vec<(usize, f32)> = raw_scores
            .into_iter()
            .enumerate()
            .map(|(i, dot)| {
                let score = match self.metric {
                    GpuDistanceMetric::DotProduct => dot,
                    GpuDistanceMetric::Cosine => {
                        let v_norm = self.norms_sq[i].sqrt();
                        let q_norm = query_norm_sq.sqrt();
                        let denom = v_norm * q_norm;
                        if denom > 0.0 { dot / denom } else { 0.0 }
                    }
                    GpuDistanceMetric::Euclidean => {
                        (self.norms_sq[i] - 2.0 * dot + query_norm_sq).max(0.0)
                    }
                };
                (i, score)
            })
            .collect();

        // Drop removed indices.
        scored.retain(|(idx, _)| !self.removed_indices.contains(idx));

        // Top-K on the CPU.
        match self.metric {
            GpuDistanceMetric::Euclidean => {
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            }
            _ => scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
        }
        scored.truncate(limit);

        Ok(scored
            .into_iter()
            .map(|(index, score)| {
                let id = self.index_to_id[index].clone();
                let similarity = match self.metric {
                    GpuDistanceMetric::Euclidean => 1.0 / (1.0 + score.sqrt()),
                    _ => score,
                };
                GpuSearchResult {
                    id,
                    score: similarity,
                    index,
                }
            })
            .collect())
    }

    fn remove_vectors(&mut self, ids: &[String]) -> Result<()> {
        for id in ids {
            self.remove_vector(id)?;
        }
        Ok(())
    }

    fn vector_count(&self) -> usize {
        self.vector_count
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn get_vector(&self, id: &str) -> Result<Option<GpuVector>> {
        self.get_vector(id)
    }

    fn clear(&mut self) -> Result<()> {
        self.clear()
    }
}
