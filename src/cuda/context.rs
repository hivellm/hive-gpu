//! # CUDA Context
//!
//! Real CUDA context backed by the [`cudarc`] driver API. Owns a primary
//! context and stream for a single GPU ordinal plus a cuBLAS handle used by
//! the vector-storage search path.
//!
//! Detection is lazy: `CudaContext::is_available()` dlopens the CUDA driver
//! once and returns `false` when the driver is missing or no devices are
//! enumerable — never panics.

use crate::error::{HiveGpuError, Result};
use crate::traits::{GpuBackend, GpuContext};
use crate::types::{GpuCapabilities, GpuDeviceInfo, GpuMemoryStats};
use std::sync::Arc;
use tracing::debug;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::cublas::CudaBlas;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use cudarc::driver::{CudaDevice, result as cuda_result, sys as cuda_sys};

/// CUDA driver context + cuBLAS handle for a single GPU ordinal.
///
/// The type is feature-gated; callers should gate their own code with
/// `#[cfg(feature = "cuda")]` when interacting with it.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Debug, Clone)]
pub struct CudaContext {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl CudaContext {
    /// Create a context on the default device (ordinal 0).
    pub fn new() -> Result<Self> {
        Self::new_with_device(0)
    }

    /// Create a context on a specific device ordinal.
    pub fn new_with_device(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal)
            .map_err(|e| HiveGpuError::CudaError(format!("CudaDevice::new({ordinal}): {e:?}")))?;
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| HiveGpuError::CublasError(format!("CudaBlas::new: {e:?}")))?;

        debug!(
            "cuda context ready: ordinal={} name={:?}",
            ordinal,
            device.name().ok()
        );

        Ok(Self {
            device,
            blas: Arc::new(blas),
        })
    }

    /// Count CUDA devices on the host. Returns 0 when the driver is absent.
    pub fn device_count() -> Result<usize> {
        cuda_result::init().map_err(|e| HiveGpuError::CudaError(format!("cuInit: {e:?}")))?;
        let count = cuda_result::device::get_count()
            .map_err(|e| HiveGpuError::CudaError(format!("cuDeviceGetCount: {e:?}")))?;
        Ok(count.max(0) as usize)
    }

    /// Non-failing availability probe used by the backend detector.
    pub fn is_available() -> bool {
        // cudarc 0.13's `dynamic-linking` feature loads libcuda.so lazily and
        // *panics* if the loader cannot resolve the library — e.g. on a
        // driver-less CI container that ships only the CUDA toolkit. Wrap the
        // first probe in `catch_unwind` so our documented contract ("never
        // panics") is honoured regardless of the host environment.
        std::panic::catch_unwind(|| {
            matches!(cuda_result::init(), Ok(()))
                && cuda_result::device::get_count()
                    .map(|c| c > 0)
                    .unwrap_or(false)
        })
        .unwrap_or(false)
    }

    /// Borrow the underlying `CudaDevice` for memory operations.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Borrow the cuBLAS handle for BLAS calls.
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// Device ordinal in 0..device_count.
    pub fn device_id(&self) -> u32 {
        self.device.ordinal() as u32
    }

    /// Human-readable device name.
    pub fn device_name(&self) -> String {
        self.device.name().unwrap_or_else(|_| "unknown".to_string())
    }

    /// Compute capability `(major, minor)` queried live from the driver.
    pub fn compute_capability(&self) -> (u32, u32) {
        let major = self
            .device
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0) as u32;
        let minor = self
            .device
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0) as u32;
        (major, minor)
    }

    /// Total physical VRAM in bytes.
    pub fn total_memory(&self) -> u64 {
        Self::mem_info(&self.device)
            .map(|(_, total)| total)
            .unwrap_or(0)
    }

    /// Free VRAM in bytes, queried live.
    pub fn available_memory(&self) -> u64 {
        Self::mem_info(&self.device)
            .map(|(free, _)| free)
            .unwrap_or(0)
    }

    /// Check that the device meets the backend's minimum compute capability
    /// (Volta / sm_70 per the CUDA backend spec).
    pub fn supports_required_features(&self) -> bool {
        self.compute_capability().0 >= 7
    }

    /// Query (free, total) VRAM via `cuMemGetInfo`. The primary context set
    /// by `CudaDevice::new` must be current on this thread; binding is
    /// cheap and handled by cudarc.
    fn mem_info(device: &Arc<CudaDevice>) -> Result<(u64, u64)> {
        device
            .bind_to_thread()
            .map_err(|e| HiveGpuError::CudaError(format!("bind_to_thread: {e:?}")))?;
        let (free, total) = cuda_result::mem_get_info()
            .map_err(|e| HiveGpuError::CudaError(format!("cuMemGetInfo: {e:?}")))?;
        Ok((free as u64, total as u64))
    }

    fn pci_bus_id_string(&self) -> Option<String> {
        let bus = self
            .device
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
            .ok()?;
        let dev = self
            .device
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
            .ok()?;
        let domain = self
            .device
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
            .unwrap_or(0);
        Some(format!("{domain:04x}:{bus:02x}:{dev:02x}.0"))
    }

    fn driver_version_string() -> String {
        let mut version: i32 = 0;
        // SAFETY: `cuDriverGetVersion` is a thread-safe driver call that only
        // writes a 32-bit integer through the passed pointer; a mutable stack
        // reference is always a valid target.
        let status = unsafe { cuda_sys::lib().cuDriverGetVersion(&mut version) };
        if status == cuda_sys::CUresult::CUDA_SUCCESS {
            format!("CUDA {}.{}", version / 1000, (version % 1000) / 10)
        } else {
            "CUDA unknown".to_string()
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl GpuBackend for CudaContext {
    fn device_info(&self) -> GpuDeviceInfo {
        let (free, total) = Self::mem_info(&self.device).unwrap_or((0, 0));
        let used = total.saturating_sub(free);
        let (major, minor) = self.compute_capability();

        let max_threads_per_block = self
            .device
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .unwrap_or(0) as u32;
        let max_shared_memory_per_block = self
            .device
            .attribute(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )
            .unwrap_or(0) as u64;

        GpuDeviceInfo {
            name: self.device_name(),
            backend: "CUDA".to_string(),
            total_vram_bytes: total,
            available_vram_bytes: free,
            used_vram_bytes: used,
            driver_version: Self::driver_version_string(),
            compute_capability: Some(format!("{major}.{minor}")),
            max_threads_per_block,
            max_shared_memory_per_block,
            device_id: self.device.ordinal() as i32,
            pci_bus_id: self.pci_bus_id_string(),
        }
    }

    fn supports_operations(&self) -> GpuCapabilities {
        GpuCapabilities {
            supports_hnsw: false, // HNSW deferred to a later task (phase 5)
            supports_batch: true,
            max_dimension: 4096,
            max_batch_size: 100_000,
        }
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        let (free, total) = Self::mem_info(&self.device).unwrap_or((0, 0));
        let used = total.saturating_sub(free) as usize;
        let available = free as usize;
        let utilization = if total == 0 {
            0.0
        } else {
            used as f32 / total as f32
        };
        GpuMemoryStats {
            total_allocated: used,
            available,
            utilization,
            buffer_count: 0,
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl GpuContext for CudaContext {
    fn create_storage(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        use crate::cuda::vector_storage::CudaVectorStorage;
        let storage = CudaVectorStorage::new(Arc::new(self.clone()), dimension, metric)?;
        Ok(Box::new(storage))
    }

    fn create_storage_with_config(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
        _config: crate::types::HnswConfig,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        // HNSW ignored in v1 — brute-force search is used regardless.
        self.create_storage(dimension, metric)
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuBackend::memory_stats(self)
    }

    fn device_info(&self) -> Result<GpuDeviceInfo> {
        Ok(GpuBackend::device_info(self))
    }
}
