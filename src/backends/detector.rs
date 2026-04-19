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
    /// ROCm / HIP (AMD)
    Rocm,
    /// CPU fallback
    Cpu,
}

impl std::fmt::Display for GpuBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackendType::Metal => write!(f, "Metal"),
            GpuBackendType::Cuda => write!(f, "CUDA"),
            GpuBackendType::Rocm => write!(f, "ROCm"),
            GpuBackendType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Detect all available GPU backends on the current system
#[allow(clippy::vec_init_then_push)] // pushes live inside cfg-gated blocks
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
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    {
        if is_cuda_available() {
            backends.push(GpuBackendType::Cuda);
        }
    }

    // Check ROCm availability (Linux-only)
    #[cfg(all(feature = "rocm", target_os = "linux"))]
    {
        if is_rocm_available() {
            backends.push(GpuBackendType::Rocm);
        }
    }

    // CPU is always available as fallback
    backends.push(GpuBackendType::Cpu);

    backends
}

/// Select the best available backend based on performance priority
pub fn select_best_backend() -> Result<GpuBackendType> {
    let available = detect_available_backends();

    // Priority order: Metal > CUDA > ROCm > CPU
    if available.contains(&GpuBackendType::Metal) {
        Ok(GpuBackendType::Metal)
    } else if available.contains(&GpuBackendType::Cuda) {
        Ok(GpuBackendType::Cuda)
    } else if available.contains(&GpuBackendType::Rocm) {
        Ok(GpuBackendType::Rocm)
    } else if available.contains(&GpuBackendType::Cpu) {
        Ok(GpuBackendType::Cpu)
    } else {
        Err(HiveGpuError::NoDeviceAvailable)
    }
}

/// Check if Metal is available on the current system
#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn is_metal_available() -> bool {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    MTLCreateSystemDefaultDevice().is_some()
}

/// Check if CUDA is available on the current system.
///
/// Uses cudarc's driver-API probe, which dynamically loads the CUDA driver at
/// call time. Returns `false` when the driver is missing or no devices are
/// enumerable — never panics.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn is_cuda_available() -> bool {
    use cudarc::driver::result;
    match result::init() {
        Ok(()) => result::device::get_count()
            .map(|count| count > 0)
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Check if ROCm / HIP is available on the current system.
///
/// Uses the libloading-based probe in `crate::rocm::context::RocmContext`
/// which dlopens `libamdhip64` / `librocblas` at call time. Returns
/// `false` when neither library is reachable.
#[cfg(all(feature = "rocm", target_os = "linux"))]
fn is_rocm_available() -> bool {
    crate::rocm::context::RocmContext::is_available()
}

/// Get backend-specific device information
pub fn get_backend_info(backend: GpuBackendType) -> Result<String> {
    match backend {
        GpuBackendType::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal-native"))]
            {
                use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
                if let Some(device) = MTLCreateSystemDefaultDevice() {
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
        GpuBackendType::Rocm => {
            #[cfg(all(feature = "rocm", target_os = "linux"))]
            {
                Ok("ROCm device available".to_string())
            }
            #[cfg(not(all(feature = "rocm", target_os = "linux")))]
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
        GpuBackendType::Rocm => BackendPerformanceInfo {
            name: "ROCm".to_string(),
            memory_bandwidth_gbps: 960.0, // RX 7900 XTX example
            compute_units: 96,            // RX 7900 XTX example
            memory_size_gb: 24,
            supports_hnsw: false,
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
