//! Memory management utilities

use crate::error::{HiveGpuError, Result};

/// Memory utility functions
pub mod memory_utils {
    use super::*;

    /// Calculate memory size for vectors
    pub fn calculate_vector_memory_size(dimension: usize, count: usize) -> usize {
        count * dimension * std::mem::size_of::<f32>()
    }

    /// Calculate memory size for metadata
    pub fn calculate_metadata_memory_size(count: usize) -> usize {
        count * 256 // 256 bytes per vector metadata
    }

    /// Check if memory allocation is safe
    pub fn is_memory_allocation_safe(
        requested: usize,
        available: usize,
        safety_margin: f32,
    ) -> bool {
        let safe_limit = (available as f32 * safety_margin) as usize;
        requested <= safe_limit
    }

    /// Calculate optimal buffer size
    pub fn calculate_optimal_buffer_size(
        current_size: usize,
        growth_factor: f32,
        max_size: usize,
    ) -> usize {
        let new_size = (current_size as f32 * growth_factor) as usize;
        new_size.min(max_size)
    }
}

/// Buffer utility functions
pub mod buffer_utils {
    use super::*;

    /// Buffer alignment for optimal GPU performance
    pub const BUFFER_ALIGNMENT: usize = 256;

    /// Align buffer size to GPU requirements
    pub fn align_buffer_size(size: usize) -> usize {
        ((size + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT
    }

    /// Calculate buffer pool size
    pub fn calculate_buffer_pool_size(
        vector_count: usize,
        dimension: usize,
        pool_factor: f32,
    ) -> usize {
        let base_size = memory_utils::calculate_vector_memory_size(dimension, vector_count);
        (base_size as f32 * pool_factor) as usize
    }

    /// Validate buffer parameters
    pub fn validate_buffer_parameters(
        size: usize,
        alignment: usize,
        max_size: usize,
    ) -> Result<()> {
        if size == 0 {
            return Err(HiveGpuError::BufferAllocationFailed(
                "Zero buffer size".to_string(),
            ));
        }

        if size > max_size {
            return Err(HiveGpuError::BufferAllocationFailed(format!(
                "Buffer size {} exceeds maximum {}",
                size, max_size
            )));
        }

        if size % alignment != 0 {
            return Err(HiveGpuError::BufferAllocationFailed(format!(
                "Buffer size {} not aligned to {}",
                size, alignment
            )));
        }

        Ok(())
    }
}
