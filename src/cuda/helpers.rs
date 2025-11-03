//! # CUDA Helpers
//!
//! Helper functions for CUDA GPU operations.

use super::context::CudaContext;
use crate::error::{HiveGpuError, Result};

/// CUDA Helper Functions
#[cfg(feature = "cuda")]
pub struct CudaHelpers;

#[cfg(feature = "cuda")]
impl CudaHelpers {
    /// Calculate optimal block size for CUDA kernels
    pub fn calculate_block_size(
        context: &CudaContext,
        grid_size: (u32, u32, u32),
    ) -> Result<(u32, u32, u32)> {
        // Calculate optimal block size based on device capabilities
        let (major, minor) = context.compute_capability();

        // Block size limits based on compute capability
        let max_threads_per_block = if major >= 7 {
            1024 // Volta and newer
        } else if major >= 6 {
            1024 // Pascal
        } else {
            512 // Older architectures
        };

        // Calculate optimal block size
        let x = grid_size.0.min(max_threads_per_block);
        let y = grid_size.1.min(max_threads_per_block / x);
        let z = grid_size.2.min(max_threads_per_block / (x * y));

        Ok((x, y, z))
    }

    /// Validate CUDA device capabilities
    pub fn validate_device_capabilities(context: &CudaContext) -> Result<()> {
        // Check if device supports required features
        if !context.supports_required_features() {
            return Err(HiveGpuError::Other(
                "CUDA device does not support required features".to_string(),
            ));
        }

        Ok(())
    }
}
