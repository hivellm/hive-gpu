//! GPU Backend Detection
//!
//! This module provides detection of available GPU backends and selection
//! of the best backend for the current system.

use crate::error::{HiveGpuError, Result};

/// Available GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendType {
    /// Metal (Apple Silicon)
    Metal,
    /// CUDA (NVIDIA)
    Cuda,
    /// CPU fallback
    Cpu,
}

impl std::fmt::Display for GpuBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackendType::Metal => write!(f, "Metal"),
            GpuBackendType::Cuda => write!(f, "CUDA"),
            GpuBackendType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Detect all available GPU backends on the current system
pub fn detect_available_backends() -> Vec<GpuBackendType> {
    let mut backends = Vec::new();

    // Check Metal availability (macOS only)
    #[cfg(all(target_os = "macos", feature = "metal-native"))]
    {
        if is_metal_available() {
            backends.push(GpuBackendType::Metal);
        }
    }

    // Check CUDA availability
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            backends.push(GpuBackendType::Cuda);
        }
    }

    // CPU is always available as fallback
    backends.push(GpuBackendType::Cpu);

    backends
}

/// Select the best available backend based on performance priority
pub fn select_best_backend() -> Result<GpuBackendType> {
    let available = detect_available_backends();

    // Priority order: Metal > CUDA > CPU
    if available.contains(&GpuBackendType::Metal) {
        Ok(GpuBackendType::Metal)
    } else if available.contains(&GpuBackendType::Cuda) {
        Ok(GpuBackendType::Cuda)
    } else if available.contains(&GpuBackendType::Cpu) {
        Ok(GpuBackendType::Cpu)
    } else {
        Err(HiveGpuError::NoDeviceAvailable)
    }
}

/// Check if Metal is available on the current system
#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn is_metal_available() -> bool {
    use metal::Device;

    Device::system_default().is_some()
}

/// Check if CUDA is available on the current system
#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    // This is a simplified check - in practice you'd check for CUDA runtime
    // and available devices
    std::env::var("CUDA_VISIBLE_DEVICES").is_ok() || std::env::var("CUDA_HOME").is_ok()
}

/// Get backend-specific device information
pub fn get_backend_info(backend: GpuBackendType) -> Result<String> {
    match backend {
        GpuBackendType::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal-native"))]
            {
                use metal::Device;
                if let Some(device) = Device::system_default() {
                    Ok(format!("Metal device: {}", device.name()))
                } else {
                    Err(HiveGpuError::NoDeviceAvailable)
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "metal-native")))]
            {
                Err(HiveGpuError::NoDeviceAvailable)
            }
        }
        GpuBackendType::Cuda => {
            #[cfg(feature = "cuda")]
            {
                // In practice, you'd query CUDA devices here
                Ok("CUDA device available".to_string())
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(HiveGpuError::NoDeviceAvailable)
            }
        }
        GpuBackendType::Cpu => Ok("CPU fallback".to_string()),
    }
}

/// Get performance characteristics for each backend
pub fn get_backend_performance_info(backend: GpuBackendType) -> BackendPerformanceInfo {
    match backend {
        GpuBackendType::Metal => BackendPerformanceInfo {
            name: "Metal".to_string(),
            memory_bandwidth_gbps: 400.0, // Apple Silicon typical
            compute_units: 8,             // M1 Pro example
            memory_size_gb: 16,
            supports_hnsw: true,
            supports_batch: true,
        },
        GpuBackendType::Cuda => BackendPerformanceInfo {
            name: "CUDA".to_string(),
            memory_bandwidth_gbps: 900.0, // RTX 4090 example
            compute_units: 128,           // RTX 4090 example
            memory_size_gb: 24,
            supports_hnsw: true,
            supports_batch: true,
        },
        GpuBackendType::Cpu => BackendPerformanceInfo {
            name: "CPU".to_string(),
            memory_bandwidth_gbps: 50.0, // DDR4-3200 example
            compute_units: 16,           // 16-core example
            memory_size_gb: 32,
            supports_hnsw: false,
            supports_batch: true,
        },
    }
}

/// Backend performance characteristics
#[derive(Debug, Clone)]
pub struct BackendPerformanceInfo {
    pub name: String,
    pub memory_bandwidth_gbps: f32,
    pub compute_units: usize,
    pub memory_size_gb: usize,
    pub supports_hnsw: bool,
    pub supports_batch: bool,
}
