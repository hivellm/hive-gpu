//! # ROCm Context
//!
//! Real context backed by `libamdhip64` + `librocblas` loaded dynamically
//! via [`super::ffi::hip_lib`]. Mirrors the shape of
//! [`crate::cuda::CudaContext`] so downstream code stays portable.
//!
//! ⚠️ AUTHORED BLIND — no AMD hardware was available at write time. See
//! the phase3b proposal and `phase4d`-style validation task for the
//! checklist a Linux + AMD maintainer needs to run before archiving.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use super::ffi::{
    self, HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR, HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR,
    HIP_DEVICE_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, HIP_DEVICE_ATTR_MAX_THREADS_PER_BLOCK,
    HIP_DEVICE_ATTR_PCI_BUS_ID, HIP_DEVICE_ATTR_PCI_DEVICE_ID, HIP_DEVICE_ATTR_PCI_DOMAIN_ID,
    HipDevice_t, HipStream_t, RocblasHandle, hip_check, rocblas_check,
};
use crate::error::{HiveGpuError, Result};
use crate::traits::{GpuBackend, GpuContext};
use crate::types::{GpuCapabilities, GpuDeviceInfo, GpuMemoryStats};
use std::os::raw::{c_char, c_int};
use std::sync::Arc;
use tracing::debug;

/// ROCm driver context + rocBLAS handle for a single GPU ordinal.
pub struct RocmContext {
    device_id: i32,
    stream: HipStream_t,
    rocblas_handle: RocblasHandle,
}

// SAFETY: All three fields are opaque device-side handles. Access is
// serialised via the owning stream; we never read the raw pointers from
// multiple threads without going through rocBLAS/HIP (which internally
// synchronise on the stream).
unsafe impl Send for RocmContext {}
unsafe impl Sync for RocmContext {}

impl std::fmt::Debug for RocmContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmContext")
            .field("device_id", &self.device_id)
            .field("stream", &self.stream)
            .field("rocblas_handle", &self.rocblas_handle)
            .finish()
    }
}

impl RocmContext {
    /// Create a context on the default device (ordinal 0).
    pub fn new() -> Result<Arc<Self>> {
        Self::new_with_device(0)
    }

    /// Create a context on a specific device ordinal.
    pub fn new_with_device(ordinal: i32) -> Result<Arc<Self>> {
        let lib = ffi::require_hip_lib()?;

        // `hipInit(0)` is idempotent — the HIP runtime tolerates repeat
        // calls. Safe to call on every context creation.
        // SAFETY: `hip_init` has no preconditions beyond the library
        // being loaded.
        let status = unsafe { (lib.hip_init)(0) };
        hip_check(status, "hipInit")?;

        // SAFETY: `hip_set_device` takes a device ordinal; we validate
        // bounds via `hipGetDeviceCount` implicitly (HIP returns an error
        // status if the ordinal is out of range).
        let status = unsafe { (lib.hip_set_device)(ordinal) };
        hip_check(status, "hipSetDevice")?;

        // Create a stream dedicated to this context. The stream owns all
        // operations dispatched through rocBLAS and HIP memcpy calls.
        let mut stream: HipStream_t = std::ptr::null_mut();
        // SAFETY: We pass a valid out-pointer; HIP writes the newly
        // created stream handle into it on success.
        let status = unsafe { (lib.hip_stream_create)(&mut stream) };
        hip_check(status, "hipStreamCreate")?;

        // rocBLAS handle bound to the stream so every SGEMV/SGEMM runs on
        // our serialised queue.
        let mut rocblas_handle: RocblasHandle = std::ptr::null_mut();
        // SAFETY: `rocblas_create_handle` writes the handle via the
        // out-pointer; callers must pair every successful create with a
        // `rocblas_destroy_handle`. Our `Drop` impl does exactly that.
        let status = unsafe { (lib.rocblas_create_handle)(&mut rocblas_handle) };
        if let Err(e) = rocblas_check(status, "rocblas_create_handle") {
            // Clean up the HIP stream we already created.
            // SAFETY: `stream` is a valid handle we just created; HIP
            // tolerates destroying a stream with queued work.
            unsafe {
                let _ = (lib.hip_stream_destroy)(stream);
            }
            return Err(e);
        }

        // Attach the stream to rocBLAS so every subsequent rocBLAS call
        // queues on the same stream as the HIP memcpy operations.
        // SAFETY: both handles were just created successfully.
        let status = unsafe { (lib.rocblas_set_stream)(rocblas_handle, stream) };
        if let Err(e) = rocblas_check(status, "rocblas_set_stream") {
            unsafe {
                let _ = (lib.rocblas_destroy_handle)(rocblas_handle);
                let _ = (lib.hip_stream_destroy)(stream);
            }
            return Err(e);
        }

        debug!("rocm context ready: ordinal={}", ordinal);
        Ok(Arc::new(Self {
            device_id: ordinal,
            stream,
            rocblas_handle,
        }))
    }

    /// Non-failing availability probe used by the backend detector.
    pub fn is_available() -> bool {
        let Some(lib) = ffi::hip_lib() else {
            return false;
        };
        // SAFETY: the HIP runtime is safe to query for device count even
        // before `hipInit`; if the driver is not loaded the call returns
        // a non-zero status.
        let mut count: c_int = 0;
        let init_status = unsafe { (lib.hip_init)(0) };
        if init_status != 0 {
            return false;
        }
        let status = unsafe { (lib.hip_get_device_count)(&mut count) };
        status == 0 && count > 0
    }

    /// Count HIP devices on the host. Returns 0 when the runtime is
    /// absent.
    pub fn device_count() -> Result<usize> {
        let lib = ffi::require_hip_lib()?;
        // SAFETY: `hip_init` is idempotent; `hip_get_device_count` writes
        // a single int to the provided pointer.
        let mut count: c_int = 0;
        let status = unsafe { (lib.hip_init)(0) };
        hip_check(status, "hipInit")?;
        let status = unsafe { (lib.hip_get_device_count)(&mut count) };
        hip_check(status, "hipGetDeviceCount")?;
        Ok(count.max(0) as usize)
    }

    /// Borrowed rocBLAS handle. Exposed so the vector storage and IVF
    /// index can dispatch rocBLAS calls.
    pub(crate) fn rocblas_handle(&self) -> RocblasHandle {
        self.rocblas_handle
    }

    /// Borrowed HIP stream owned by this context.
    pub(crate) fn stream(&self) -> HipStream_t {
        self.stream
    }

    /// Device ordinal (0-indexed).
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Human-readable device name from `hipDeviceGetName`.
    pub fn device_name(&self) -> String {
        let Ok(lib) = ffi::require_hip_lib() else {
            return "unknown".to_string();
        };
        let mut buf = [0i8; 256];
        // SAFETY: HIP writes a null-terminated UTF-8 string into `buf`;
        // we cap the length so there is no overflow risk.
        let status = unsafe {
            (lib.hip_device_get_name)(
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
                self.device_id,
            )
        };
        if status != 0 {
            return "unknown".to_string();
        }
        // SAFETY: buffer is null-terminated on success.
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                buf.as_ptr() as *const u8,
                buf.iter().position(|&c| c == 0).unwrap_or(buf.len()),
            );
            String::from_utf8_lossy(bytes).to_string()
        }
    }

    /// gfx architecture inferred from compute capability attributes.
    /// ROCm exposes `MAJOR` / `MINOR` that map to the gfx codes most
    /// users recognise (e.g. (10, 3) → gfx1030).
    pub fn compute_capability(&self) -> (u32, u32) {
        let Ok(lib) = ffi::require_hip_lib() else {
            return (0, 0);
        };
        let major = get_attribute(
            lib,
            HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR,
            self.device_id,
        );
        let minor = get_attribute(
            lib,
            HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR,
            self.device_id,
        );
        (major.max(0) as u32, minor.max(0) as u32)
    }

    /// Format the `gfx<major><minor>` string that ROCm ecosystems use.
    pub fn gfx_string(&self) -> String {
        let (maj, min) = self.compute_capability();
        format!("gfx{maj}{min:02}")
    }

    /// Total physical VRAM in bytes.
    pub fn total_memory(&self) -> u64 {
        Self::mem_info(self.device_id)
            .map(|(_, total)| total)
            .unwrap_or(0)
    }

    /// Free VRAM in bytes, queried live.
    pub fn available_memory(&self) -> u64 {
        Self::mem_info(self.device_id)
            .map(|(free, _)| free)
            .unwrap_or(0)
    }

    fn mem_info(device: HipDevice_t) -> Result<(u64, u64)> {
        let lib = ffi::require_hip_lib()?;
        // SAFETY: `hip_set_device` is safe; `hip_mem_get_info` writes two
        // usize values via out-pointers.
        let status = unsafe { (lib.hip_set_device)(device) };
        hip_check(status, "hipSetDevice")?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = unsafe { (lib.hip_mem_get_info)(&mut free, &mut total) };
        hip_check(status, "hipMemGetInfo")?;
        Ok((free as u64, total as u64))
    }

    fn pci_bus_id_string(&self) -> Option<String> {
        let lib = ffi::require_hip_lib().ok()?;
        let bus = get_attribute(lib, HIP_DEVICE_ATTR_PCI_BUS_ID, self.device_id);
        let dev = get_attribute(lib, HIP_DEVICE_ATTR_PCI_DEVICE_ID, self.device_id);
        let domain = get_attribute(lib, HIP_DEVICE_ATTR_PCI_DOMAIN_ID, self.device_id);
        if bus < 0 || dev < 0 {
            return None;
        }
        Some(format!("{domain:04x}:{bus:02x}:{dev:02x}.0"))
    }

    fn driver_version_string() -> String {
        let Ok(lib) = ffi::require_hip_lib() else {
            return "ROCm unknown".to_string();
        };
        let mut version: c_int = 0;
        // SAFETY: `hipDriverGetVersion` is safe; it writes an int.
        let status = unsafe { (lib.hip_driver_get_version)(&mut version) };
        if status != 0 {
            return "ROCm unknown".to_string();
        }
        format!("ROCm {}.{}", version / 1000, (version % 1000) / 10)
    }
}

fn get_attribute(lib: &ffi::HipLib, attr: c_int, device: HipDevice_t) -> c_int {
    let mut value: c_int = 0;
    // SAFETY: `hipDeviceGetAttribute` writes a single int.
    let status = unsafe { (lib.hip_device_get_attribute)(&mut value, attr, device) };
    if status == 0 { value } else { -1 }
}

impl Drop for RocmContext {
    fn drop(&mut self) {
        let Some(lib) = ffi::hip_lib() else {
            return;
        };
        // Destruction order matters: drop rocBLAS first so it can sync
        // any pending work before the stream goes away. Swallow errors
        // on teardown — there is nothing useful a caller can do.
        // SAFETY: both handles were created by us in `new_with_device`
        // and are not borrowed elsewhere at this point (we hold a
        // unique `&mut self`).
        if !self.rocblas_handle.is_null() {
            unsafe {
                let _ = (lib.rocblas_destroy_handle)(self.rocblas_handle);
            }
            self.rocblas_handle = std::ptr::null_mut();
        }
        if !self.stream.is_null() {
            unsafe {
                let _ = (lib.hip_stream_destroy)(self.stream);
            }
            self.stream = std::ptr::null_mut();
        }
    }
}

impl GpuBackend for RocmContext {
    fn device_info(&self) -> GpuDeviceInfo {
        let (free, total) = Self::mem_info(self.device_id).unwrap_or((0, 0));
        let used = total.saturating_sub(free);
        let max_threads_per_block = ffi::hip_lib()
            .map(|l| get_attribute(l, HIP_DEVICE_ATTR_MAX_THREADS_PER_BLOCK, self.device_id))
            .unwrap_or(0) as u32;
        let max_shared_memory_per_block = ffi::hip_lib()
            .map(|l| {
                get_attribute(
                    l,
                    HIP_DEVICE_ATTR_MAX_SHARED_MEMORY_PER_BLOCK,
                    self.device_id,
                )
            })
            .unwrap_or(0) as u64;
        GpuDeviceInfo {
            name: self.device_name(),
            backend: "ROCm".to_string(),
            total_vram_bytes: total,
            available_vram_bytes: free,
            used_vram_bytes: used,
            driver_version: Self::driver_version_string(),
            compute_capability: Some(self.gfx_string()),
            max_threads_per_block,
            max_shared_memory_per_block,
            device_id: self.device_id,
            pci_bus_id: self.pci_bus_id_string(),
        }
    }

    fn supports_operations(&self) -> GpuCapabilities {
        GpuCapabilities {
            supports_hnsw: false,
            supports_batch: true,
            max_dimension: 4096,
            max_batch_size: 100_000,
        }
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        let (free, total) = Self::mem_info(self.device_id).unwrap_or((0, 0));
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

impl GpuContext for RocmContext {
    fn create_storage(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        use super::vector_storage::RocmVectorStorage;
        // The vector storage needs a cloned Arc<Self>. Since `GpuContext`
        // methods take `&self`, we materialise a fresh Arc view that
        // aliases the existing handles. Multiple `Arc<RocmContext>` are
        // safe because `Drop` is tied to the last strong ref.
        let storage = RocmVectorStorage::new(
            // SAFETY: the caller must guarantee `self` is already held in
            // an `Arc`. The `create_storage` trait contract only makes
            // sense when the context is owned that way — the public
            // constructor always returns `Arc<Self>`.
            unsafe {
                let raw = self as *const Self;
                Arc::increment_strong_count(raw);
                Arc::from_raw(raw)
            },
            dimension,
            metric,
        )?;
        Ok(Box::new(storage))
    }

    fn create_storage_with_config(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
        _config: crate::types::HnswConfig,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        self.create_storage(dimension, metric)
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuBackend::memory_stats(self)
    }

    fn device_info(&self) -> Result<GpuDeviceInfo> {
        Ok(GpuBackend::device_info(self))
    }
}
