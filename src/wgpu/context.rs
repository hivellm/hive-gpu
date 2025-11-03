//! # wgpu Context
//!
//! Unified wgpu context for all wgpu operations.
//! This module provides a single source of truth for wgpu device and queue management.

use crate::error::{HiveGpuError, Result};
use crate::traits::{GpuBackend, GpuContext};
use crate::types::{GpuCapabilities, GpuDeviceInfo, GpuMemoryStats};
use tracing::{debug, info};

/// wgpu Context - Single source of truth
#[cfg(feature = "wgpu")]
#[derive(Debug, Clone)]
pub struct WgpuContext {
    device_name: String,
    backend: String,
    total_memory: u64,
}

#[cfg(feature = "wgpu")]
impl WgpuContext {
    /// Create new wgpu context
    pub fn new() -> Result<Self> {
        // This is a placeholder implementation
        // In practice, you'd initialize wgpu using wgpu::Instance
        let device_name = "wgpu GPU".to_string();
        let backend = "Vulkan".to_string(); // Example backend
        let total_memory = 1024 * 1024 * 1024; // 1GB placeholder

        debug!("✅ wgpu context created: {} ({})", device_name, backend);

        Ok(Self {
            device_name,
            backend,
            total_memory,
        })
    }

    /// Get device name
    pub fn device_name(&self) -> String {
        self.device_name.clone()
    }

    /// Get backend name
    pub fn backend(&self) -> String {
        self.backend.clone()
    }

    /// Get total memory
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Check if device supports required features
    pub fn supports_required_features(&self) -> bool {
        // Check wgpu features and limits
        true // Placeholder
    }
}

impl GpuBackend for WgpuContext {
    fn device_info(&self) -> GpuDeviceInfo {
        let total_vram = self.total_memory();
        let used_vram = 0; // Placeholder
        let available_vram = total_vram.saturating_sub(used_vram);

        GpuDeviceInfo {
            name: self.device_name(),
            backend: "wgpu".to_string(),
            total_vram_bytes: total_vram,
            available_vram_bytes: available_vram,
            used_vram_bytes: used_vram,
            driver_version: "wgpu".to_string(),
            compute_capability: None,
            max_threads_per_block: 256,
            max_shared_memory_per_block: 16384,
            device_id: 0,
            pci_bus_id: None,
        }
    }

    fn supports_operations(&self) -> GpuCapabilities {
        GpuCapabilities {
            supports_hnsw: true,
            supports_batch: true,
            max_dimension: 512, // Conservative limit
            max_batch_size: 10000,
        }
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        // This is a simplified implementation
        // In practice, you'd query actual VRAM usage
        GpuMemoryStats {
            total_allocated: 0,
            available: self.total_memory() as usize,
            utilization: 0.0,
            buffer_count: 0,
        }
    }
}

impl GpuContext for WgpuContext {
    fn create_storage(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        // This will be implemented when we migrate vector_storage.rs
        Err(HiveGpuError::Other("Not implemented yet".to_string()))
    }

    fn create_storage_with_config(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
        config: crate::types::HnswConfig,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        // This will be implemented when we migrate vector_storage.rs
        Err(HiveGpuError::Other("Not implemented yet".to_string()))
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuBackend::memory_stats(self)
    }

    fn device_info(&self) -> Result<GpuDeviceInfo> {
        Ok(GpuBackend::device_info(self))
    }
}
