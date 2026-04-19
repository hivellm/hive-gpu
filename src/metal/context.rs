//! # Metal Native Context
//!
//! Unified Metal context for all Metal Native operations.
//! This module provides a single source of truth for Metal device and command queue management.

use crate::error::{HiveGpuError, Result};
use crate::traits::{GpuBackend, GpuContext};
use crate::types::{GpuCapabilities, GpuDeviceInfo, GpuMemoryStats};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLGPUFamily, MTLLibrary, MTLSize,
};
use std::process::Command;
use std::sync::Arc;
use tracing::{debug, warn};

/// Metal Native Context - Single source of truth
#[cfg(all(target_os = "macos", feature = "metal-native"))]
#[derive(Debug, Clone)]
pub struct MetalNativeContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
impl MetalNativeContext {
    /// Create new Metal native context
    pub fn new() -> Result<Self> {
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| HiveGpuError::NoDeviceAvailable)?;

        let command_queue = device
            .newCommandQueue()
            .ok_or_else(|| HiveGpuError::Other("Failed to create command queue".to_string()))?;

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
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Get command queue
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
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
        unsafe { self.device.supportsFamily(MTLGPUFamily::Apple7) }
    }

    /// Get maximum threadgroup size for compute shaders
    pub fn max_threadgroup_size(&self) -> MTLSize {
        self.device.maxThreadsPerThreadgroup()
    }

    /// Get maximum buffer size
    pub fn max_buffer_size(&self) -> u64 {
        // Most Metal devices support very large buffers
        // Return a conservative limit of 1GB
        1024 * 1024 * 1024
    }

    /// Get Metal library
    pub fn library(&self) -> &ProtocolObject<dyn MTLLibrary> {
        &self.library
    }

    /// Build a compute pipeline state for a named kernel function in the
    /// Metal library. Callers typically cache the returned pipeline for
    /// the lifetime of their owner struct.
    pub fn compute_pipeline(
        &self,
        function_name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let ns_name = objc2_foundation::NSString::from_str(function_name);
        let function = self.library.newFunctionWithName(&ns_name).ok_or_else(|| {
            HiveGpuError::ShaderCompilationFailed(format!(
                "Metal function '{function_name}' not found in compiled library",
            ))
        })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                HiveGpuError::ShaderCompilationFailed(format!(
                    "Failed to build pipeline for '{function_name}': {e:?}",
                ))
            })?;
        Ok(pipeline)
    }

    /// Load Metal library with HNSW shaders
    fn load_metal_library(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        // Load the Metal shader source
        let shader_source = include_str!("../shaders/metal_hnsw.metal");

        // Create NSString from shader source
        let ns_source = objc2_foundation::NSString::from_str(shader_source);
        let options = MTLCompileOptions::new();

        let library = unsafe {
            device
                .newLibraryWithSource_options_error(&ns_source, Some(&options))
                .map_err(|e| {
                    HiveGpuError::ShaderCompilationFailed(format!(
                        "Failed to compile Metal shaders: {:?}",
                        e
                    ))
                })?
        };

        debug!(
            "✅ Metal library loaded with {} functions",
            library.functionNames().count()
        );
        Ok(library)
    }

    /// Get macOS version as driver version
    fn get_macos_version() -> Result<String> {
        let output = Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .map_err(|e| {
                HiveGpuError::Other(format!("Failed to execute sw_vers command: {}", e))
            })?;

        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Ok(format!("macOS {}", version))
        } else {
            warn!("Failed to get macOS version, using fallback");
            Ok("macOS Unknown".to_string())
        }
    }
}

impl GpuBackend for MetalNativeContext {
    fn device_info(&self) -> GpuDeviceInfo {
        // Get VRAM information
        let recommended_max = self.device.recommendedMaxWorkingSetSize();
        let current_allocated = self.device.currentAllocatedSize() as u64;
        let available = recommended_max.saturating_sub(current_allocated);

        // Get macOS version
        let driver_version =
            Self::get_macos_version().unwrap_or_else(|_| "macOS Unknown".to_string());

        // Get max threadgroup size
        let max_threads = self.device.maxThreadsPerThreadgroup();
        let max_threads_per_block =
            (max_threads.width * max_threads.height * max_threads.depth) as u32;

        // Max shared memory per threadgroup (LDS on Apple Silicon)
        // Apple Silicon typically has 32KB per threadgroup
        let max_shared_memory = 32 * 1024; // 32 KB

        GpuDeviceInfo {
            name: self.device_name(),
            backend: "Metal".to_string(),
            total_vram_bytes: recommended_max,
            available_vram_bytes: available,
            used_vram_bytes: current_allocated,
            driver_version,
            compute_capability: None, // Metal doesn't expose compute capability like CUDA
            max_threads_per_block,
            max_shared_memory_per_block: max_shared_memory,
            device_id: 0,     // Metal doesn't expose device IDs for Apple Silicon
            pci_bus_id: None, // Metal doesn't expose PCI bus for Apple Silicon
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
    fn create_storage(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        use crate::metal::vector_storage::MetalNativeVectorStorage;
        let storage = MetalNativeVectorStorage::new(Arc::new(self.clone()), dimension, metric)?;
        Ok(Box::new(storage))
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
