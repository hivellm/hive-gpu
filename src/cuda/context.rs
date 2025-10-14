//! # CUDA Context
//!
//! Unified CUDA context for all CUDA operations.
//! This module provides a single source of truth for CUDA device and stream management.

use crate::error::{Result, HiveGpuError};
use crate::types::{GpuDeviceInfo, GpuCapabilities, GpuMemoryStats};
use crate::traits::{GpuBackend, GpuContext};
use tracing::{info, debug};

/// CUDA Context - Single source of truth
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaContext {
    device_id: u32,
    device_name: String,
    compute_capability: (u32, u32),
    total_memory: u64,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    /// Create new CUDA context
    pub fn new() -> Result<Self> {
        // This is a placeholder implementation
        // In practice, you'd initialize CUDA using cudarc or similar
        let device_id = 0;
        let device_name = "NVIDIA GPU".to_string();
        let compute_capability = (7, 5); // Example: RTX 3080
        let total_memory = 1024 * 1024 * 1024; // 1GB placeholder
        
        debug!("✅ CUDA context created: {}", device_name);
        
        Ok(Self {
            device_id,
            device_name,
            compute_capability,
            total_memory,
        })
    }
    
    /// Get device ID
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    
    /// Get device name
    pub fn device_name(&self) -> String {
        self.device_name.clone()
    }
    
    /// Get compute capability
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }
    
    /// Get total memory
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }
    
    /// Check if device supports required features
    pub fn supports_required_features(&self) -> bool {
        // Check compute capability and other requirements
        self.compute_capability.0 >= 6 // Minimum compute capability 6.0
    }
}

impl GpuBackend for CudaContext {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            name: self.device_name(),
            device_type: "CUDA".to_string(),
            memory_bytes: self.total_memory(),
            max_buffer_size: self.total_memory(),
            compute_capability: Some(format!("{}.{}", self.compute_capability.0, self.compute_capability.1)),
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

impl GpuContext for CudaContext {
    fn create_storage(&self, dimension: usize, metric: crate::types::GpuDistanceMetric) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        // This will be implemented when we migrate vector_storage.rs
        Err(HiveGpuError::Other("Not implemented yet".to_string()))
    }

    fn create_storage_with_config(&self, dimension: usize, metric: crate::types::GpuDistanceMetric, config: crate::types::HnswConfig) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        // This will be implemented when we migrate vector_storage.rs
        Err(HiveGpuError::Other("Not implemented yet".to_string()))
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuBackend::memory_stats(self)
    }

    fn device_info(&self) -> GpuDeviceInfo {
        GpuBackend::device_info(self)
    }
}
