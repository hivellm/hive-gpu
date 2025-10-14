//! # Metal Native Context
//!
//! Unified Metal context for all Metal Native operations.
//! This module provides a single source of truth for Metal device and command queue management.

use metal::{Device as MetalDevice, CommandQueue, MTLGPUFamily, MTLSize, Library};
use std::sync::Arc;
use crate::error::{HiveGpuError, Result};
use crate::types::{GpuDeviceInfo, GpuCapabilities, GpuMemoryStats};
use crate::traits::{GpuBackend, GpuContext};
use tracing::{info, debug};

/// Metal Native Context - Single source of truth
#[cfg(all(target_os = "macos", feature = "metal-native"))]
#[derive(Debug, Clone)]
pub struct MetalNativeContext {
    device: MetalDevice,
    command_queue: CommandQueue,
    library: Library,
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
impl MetalNativeContext {
    /// Create new Metal native context
    pub fn new() -> Result<Self> {
        let device = MetalDevice::system_default()
            .ok_or_else(|| HiveGpuError::NoDeviceAvailable)?;

        let command_queue = device.new_command_queue();

        // Load Metal library with HNSW shaders
        let library = Self::load_metal_library(&device)?;

        debug!("✅ Metal native context created: {}", device.name());

        Ok(Self {
            device,
            command_queue,
            library,
        })
    }
    
    /// Get Metal device
    pub fn device(&self) -> &MetalDevice {
        &self.device
    }
    
    /// Get command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }
    
    /// Get device name
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }
    
    /// Check if device supports Metal Performance Shaders
    pub fn supports_mps(&self) -> bool {
        // Check if device supports MPS (Metal Performance Shaders)
        // This is a simplified check - in practice you'd check specific MPS features
        self.device.supports_family(MTLGPUFamily::Apple7)
    }
    
    /// Get maximum threadgroup size for compute shaders
    pub fn max_threadgroup_size(&self) -> MTLSize {
        self.device.max_threads_per_threadgroup()
    }
    
    /// Get maximum buffer size
    pub fn max_buffer_size(&self) -> u64 {
        // Most Metal devices support very large buffers
        // Return a conservative limit of 1GB
        1024 * 1024 * 1024
    }

    /// Get Metal library
    pub fn library(&self) -> &Library {
        &self.library
    }

    /// Load Metal library with HNSW shaders
    fn load_metal_library(device: &MetalDevice) -> Result<Library> {
        // Load the Metal shader source
        let shader_source = include_str!("../shaders/metal_hnsw.metal");

        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .map_err(|e| HiveGpuError::ShaderCompilationFailed(format!("Failed to compile Metal shaders: {:?}", e)))?;

        debug!("✅ Metal library loaded with {} functions", library.function_names().len());
        Ok(library)
    }
}

impl GpuBackend for MetalNativeContext {
    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            name: self.device_name(),
            device_type: "Metal".to_string(),
            memory_bytes: self.max_buffer_size(),
            max_buffer_size: self.max_buffer_size(),
            compute_capability: Some(format!("Apple{}", self.device.supports_family(MTLGPUFamily::Apple7) as u8 + 6)),
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
            available: self.max_buffer_size() as usize,
            utilization: 0.0,
            buffer_count: 0,
        }
    }
}

impl GpuContext for MetalNativeContext {
    fn create_storage(&self, dimension: usize, metric: crate::types::GpuDistanceMetric) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        use crate::metal::vector_storage::MetalNativeVectorStorage;
        let storage = MetalNativeVectorStorage::new(Arc::new(self.clone()), dimension, metric)?;
        Ok(Box::new(storage))
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
