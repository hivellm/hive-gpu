//! # Metal Native Vector Storage
//!
//! High-performance vector storage using Metal GPU acceleration.
//! All vector data is stored in VRAM for maximum efficiency.

use metal::{Buffer as MetalBuffer, Device as MetalDevice, MTLResourceOptions, MTLStorageMode, MTLCPUCacheMode};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::error::{Result, HiveGpuError};
use crate::types::{GpuVector, GpuSearchResult, GpuDistanceMetric};
use crate::traits::GpuVectorStorage;
use super::context::MetalNativeContext;
use tracing::{info, warn, debug};

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
    pub vectors_buffer: MetalBuffer,  // Made public for GPU search access
    metadata_buffer: MetalBuffer,
    vector_count: usize,
    buffer_capacity: usize, // Total capacity in vectors
    dimension: usize,
    metric: GpuDistanceMetric,
    vector_id_map: HashMap<String, usize>,
    index_to_id: Vec<String>, // Maps index to original ID
    vector_metadata: HashMap<String, VectorMetadata>, // Maps ID to metadata
    pub removed_indices: HashSet<usize>, // Tracks removed vector indices - made public for GPU search
    vector_payloads: HashMap<String, Option<std::collections::HashMap<String, String>>>, // Store payloads in CPU memory
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
impl MetalNativeVectorStorage {
    /// Create new Metal native vector storage
    pub fn new(context: Arc<MetalNativeContext>, dimension: usize, metric: GpuDistanceMetric) -> Result<Self> {
        let device = context.device();
        
        // Calculate initial capacity (minimum 1024 vectors or 1MB worth)
        let min_vectors = 1024;
        let min_bytes = 1024 * 1024; // 1MB
        let min_vectors_by_bytes = min_bytes / (dimension * std::mem::size_of::<f32>());
        let initial_capacity = min_vectors.max(min_vectors_by_bytes);
        
        let initial_size = initial_capacity
            .checked_mul(dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HiveGpuError::Other("Initial buffer size calculation overflow".to_string()))?;
        
        // Create vectors buffer (VRAM only, no CPU access)
        let vectors_buffer = device.new_buffer(
            initial_size as u64,
            MTLResourceOptions::StorageModePrivate, // VRAM only, fastest
        );
        
        // Create metadata buffer (VRAM only)
        let metadata_buffer = device.new_buffer(
            initial_capacity as u64 * 256, // 256 bytes per vector metadata
            MTLResourceOptions::StorageModePrivate, // VRAM only
        );
        
        debug!("✅ Metal native vector storage created (VRAM only) with capacity: {}", initial_capacity);
        
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
        })
    }
    
    /// Add vector to storage (VRAM only)
    pub fn add_vector(&mut self, vector: &GpuVector) -> Result<usize> {
        // Validate vector ID is unique
        if self.vector_id_map.contains_key(&vector.id) {
            return Err(HiveGpuError::Other(format!("Vector with ID '{}' already exists", vector.id)));
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
                return Err(HiveGpuError::Other(format!("Vector contains non-finite value at index {}: {}", i, value)));
            }
        }
        
        // Validate ID length
        if vector.id.len() > 256 {
            return Err(HiveGpuError::Other("Vector ID too long (max 256 chars)".to_string()));
        }
        
        // Check if we need to expand buffer
        if self.vector_count >= self.buffer_capacity {
            self.expand_buffer()?;
        }
        
        let device = self.context.device();
        let queue = self.context.command_queue();
        
        // Upload new vector data directly to existing buffer
        let vector_data = &vector.data;
        let offset = self.vector_count
            .checked_mul(self.dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HiveGpuError::Other("Offset calculation overflow".to_string()))?;
        
        // Create staging buffer for upload
        let staging_size = self.dimension
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HiveGpuError::Other("Staging size calculation overflow".to_string()))?;
        
        let staging_buffer = device.new_buffer_with_data(
            vector_data.as_ptr() as *const std::ffi::c_void,
            staging_size as u64,
            MTLResourceOptions::StorageModeShared, // CPU accessible for upload
        );
        
        // Copy from staging to VRAM buffer
        let command_buffer = queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();
        
        blit_encoder.copy_from_buffer(
            &staging_buffer,
            0,
            &self.vectors_buffer,
            offset as u64,
            staging_size as u64,
        );
        
        blit_encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
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
        self.vector_payloads.insert(vector.id.clone(), Some(vector.metadata.clone())); // Store metadata as payload
        self.vector_count += 1;
        
        debug!("✅ Vector added to VRAM: {} (total: {}, has_metadata: {})", vector.id, self.vector_count, !vector.metadata.is_empty());
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
            .ok_or_else(|| HiveGpuError::Other("New buffer size calculation overflow".to_string()))?;
        
        if new_size > 1024 * 1024 * 1024 { // 1GB limit
            return Err(HiveGpuError::VramLimitExceeded {
                requested: new_size,
                limit: 1024 * 1024 * 1024,
            });
        }
        
        info!("🔄 Expanding Metal buffer: {} -> {} vectors ({} MB)", 
            self.buffer_capacity, new_capacity, new_size / 1024 / 1024);
        
        // Create new larger buffer
        let new_vectors_buffer = device.new_buffer(
            new_size as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        
        let new_metadata_buffer = device.new_buffer(
            new_capacity as u64 * 256,
            MTLResourceOptions::StorageModePrivate,
        );
        
        // Copy existing data to new buffer
        let command_buffer = queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();
        
        let current_size = self.vector_count
            .checked_mul(self.dimension)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HiveGpuError::Other("Current size calculation overflow".to_string()))?;
        
        blit_encoder.copy_from_buffer(
            &self.vectors_buffer,
            0,
            &new_vectors_buffer,
            0,
            current_size as u64,
        );
        
        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
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
            buffer_size_mb: (self.buffer_capacity * self.dimension * std::mem::size_of::<f32>()) / 1024 / 1024,
        }
    }
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
        // Basic implementation for validation
        // TODO: Implement GPU-accelerated search using Metal shaders
        
        if self.vector_count() == 0 {
            return Ok(vec![]);
        }
        
        let mut results = Vec::new();
        
        // For now, return mock results to validate the integration
        // In a real implementation, you'd read from Metal buffers and use GPU shaders
        for i in 0..std::cmp::min(limit, self.vector_count()) {
            if let Some(id) = self.index_to_id.get(i) {
                results.push(GpuSearchResult {
                    id: id.clone(),
                    score: 1.0 - (i as f32 * 0.1), // Mock similarity scores
                    index: i,
                });
            }
        }
        
        Ok(results)
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
