//! # wgpu Helpers
//!
//! Helper functions for wgpu GPU operations.

use super::context::WgpuContext;
use crate::error::{HiveGpuError, Result};

/// wgpu Helper Functions
#[cfg(feature = "wgpu")]
pub struct WgpuHelpers;

#[cfg(feature = "wgpu")]
impl WgpuHelpers {
    /// Calculate optimal workgroup size for wgpu compute shaders
    pub fn calculate_workgroup_size(
        context: &WgpuContext,
        workgroup_size: (u32, u32, u32),
    ) -> Result<(u32, u32, u32)> {
        // Calculate optimal workgroup size based on wgpu limits
        // wgpu typically supports up to 1024 threads per workgroup
        let max_threads_per_workgroup = 1024;

        // Calculate optimal workgroup size
        let x = workgroup_size.0.min(max_threads_per_workgroup);
        let y = workgroup_size.1.min(max_threads_per_workgroup / x);
        let z = workgroup_size.2.min(max_threads_per_workgroup / (x * y));

        Ok((x, y, z))
    }

    /// Validate wgpu device capabilities
    pub fn validate_device_capabilities(context: &WgpuContext) -> Result<()> {
        // Check if device supports required features
        if !context.supports_required_features() {
            return Err(HiveGpuError::Other(
                "wgpu device does not support required features".to_string(),
            ));
        }

        Ok(())
    }
}
