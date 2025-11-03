//! # Metal Helpers
//!
//! Helper functions for Metal GPU operations.

use super::context::MetalNativeContext;
use crate::error::{HiveGpuError, Result};
use std::sync::Arc;

/// Metal Helper Functions
#[cfg(all(target_os = "macos", feature = "metal-native"))]
pub struct MetalHelpers;

#[cfg(all(target_os = "macos", feature = "metal-native"))]
impl MetalHelpers {
    /// Calculate optimal threadgroup size for Metal compute shaders
    pub fn calculate_threadgroup_size(
        context: &MetalNativeContext,
        workgroup_size: (u32, u32, u32),
    ) -> Result<(u32, u32, u32)> {
        let max_size = context.max_threadgroup_size();

        // Ensure workgroup size doesn't exceed device limits
        let x = workgroup_size.0.min(max_size.width as u32);
        let y = workgroup_size.1.min(max_size.height as u32);
        let z = workgroup_size.2.min(max_size.depth as u32);

        Ok((x, y, z))
    }

    /// Validate Metal device capabilities
    pub fn validate_device_capabilities(context: &MetalNativeContext) -> Result<()> {
        // Check if device supports required features
        if !context.supports_mps() {
            return Err(HiveGpuError::Other(
                "Metal Performance Shaders not supported".to_string(),
            ));
        }

        Ok(())
    }
}
