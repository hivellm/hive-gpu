//! Hand-rolled FFI for the HIP runtime and rocBLAS, loaded dynamically
//! via `libloading` so the crate builds on hosts without ROCm installed.
//!
//! ⚠️ AUTHORED BLIND — no AMD hardware was available at write time. The
//! function signatures below mirror the HIP 6.x and rocBLAS 4.x C APIs
//! documented at https://rocm.docs.amd.com. A maintainer with a
//! supported AMD GPU should run the ROCm suite and fix any signature
//! drift before the phase3b task is archived.
//!
//! Design:
//! - `HipLib` is a singleton that dlopens `libamdhip64.so` /
//!   `librocblas.so` once. Every HIP / rocBLAS call goes through it.
//! - Only the ~30 symbols hive-gpu needs are declared. Adding a new
//!   symbol means adding a typed function pointer field here plus the
//!   corresponding `.get` call inside `HipLib::load`.
//! - Error conversion helpers wrap raw status codes into
//!   `HiveGpuError::HipError` / `HiveGpuError::RocblasError`.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use crate::error::{HiveGpuError, Result};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

// ---------- opaque types -----------------------------------------------

/// HIP status enum. 0 == `hipSuccess`; anything else is an error.
pub type HipError_t = c_int;
/// HIP device ordinal (0-indexed).
pub type HipDevice_t = c_int;
/// Opaque stream pointer (`hipStream_t`).
pub type HipStream_t = *mut c_void;
/// Device memory pointer on the HIP side. `hipMalloc` returns one via
/// a `void**` out-parameter; we carry it as `*mut c_void`.
pub type HipDevicePtr_t = *mut c_void;

/// rocBLAS status enum. 0 == `rocblas_status_success`.
pub type RocblasStatus = c_int;
/// Opaque rocBLAS library handle.
pub type RocblasHandle = *mut c_void;

/// `rocblas_operation` — matches the C enum values:
/// `rocblas_operation_none = 111`,
/// `rocblas_operation_transpose = 112`,
/// `rocblas_operation_conjugate_transpose = 113`.
pub const ROCBLAS_OP_N: c_int = 111;
pub const ROCBLAS_OP_T: c_int = 112;

/// Device attribute ids. These are the HIP enum values that map 1:1 to
/// CUDA `CUdevice_attribute`. Hardcoding the numeric values keeps us
/// independent of the ROCm header version. See
/// https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___device.html
/// Note: ordinal values differ between ROCm versions in some cases.
/// These match ROCm 6.x (stable).
pub const HIP_DEVICE_ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const HIP_DEVICE_ATTR_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const HIP_DEVICE_ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub const HIP_DEVICE_ATTR_PCI_BUS_ID: c_int = 33;
pub const HIP_DEVICE_ATTR_PCI_DEVICE_ID: c_int = 34;
pub const HIP_DEVICE_ATTR_PCI_DOMAIN_ID: c_int = 50;
pub const HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const HIP_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const HIP_DEVICE_ATTR_WARP_SIZE: c_int = 10;

// ---------- loaded library --------------------------------------------

/// Dynamically-loaded HIP + rocBLAS symbol table. Obtain via
/// [`hip_lib`]; every call is a borrow of the static singleton.
pub(crate) struct HipLib {
    // The Libraries must outlive every Symbol pulled out of them. We keep
    // them as fields so the Drop impl runs at process shutdown only.
    _hip_lib: Library,
    _rocblas_lib: Library,

    // --- core HIP -----------------------------------------------------
    pub hip_init: unsafe extern "C" fn(flags: c_uint) -> HipError_t,
    pub hip_get_device_count: unsafe extern "C" fn(count: *mut c_int) -> HipError_t,
    pub hip_set_device: unsafe extern "C" fn(device: HipDevice_t) -> HipError_t,
    pub hip_get_device: unsafe extern "C" fn(device: *mut HipDevice_t) -> HipError_t,
    pub hip_device_get_name:
        unsafe extern "C" fn(name: *mut c_char, len: c_int, device: HipDevice_t) -> HipError_t,
    pub hip_device_get_attribute:
        unsafe extern "C" fn(value: *mut c_int, attr: c_int, device: HipDevice_t) -> HipError_t,
    pub hip_device_total_mem:
        unsafe extern "C" fn(bytes: *mut usize, device: HipDevice_t) -> HipError_t,
    pub hip_mem_get_info: unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> HipError_t,
    pub hip_driver_get_version: unsafe extern "C" fn(version: *mut c_int) -> HipError_t,
    pub hip_runtime_get_version: unsafe extern "C" fn(version: *mut c_int) -> HipError_t,

    // --- streams ------------------------------------------------------
    pub hip_stream_create: unsafe extern "C" fn(stream: *mut HipStream_t) -> HipError_t,
    pub hip_stream_destroy: unsafe extern "C" fn(stream: HipStream_t) -> HipError_t,
    pub hip_stream_synchronize: unsafe extern "C" fn(stream: HipStream_t) -> HipError_t,

    // --- memory -------------------------------------------------------
    pub hip_malloc: unsafe extern "C" fn(ptr: *mut HipDevicePtr_t, size: usize) -> HipError_t,
    pub hip_free: unsafe extern "C" fn(ptr: HipDevicePtr_t) -> HipError_t,
    /// Generic `hipMemcpy` with kind.
    pub hip_memcpy: unsafe extern "C" fn(
        dst: HipDevicePtr_t,
        src: *const c_void,
        size: usize,
        kind: c_int,
    ) -> HipError_t,
    pub hip_memcpy_async: unsafe extern "C" fn(
        dst: HipDevicePtr_t,
        src: *const c_void,
        size: usize,
        kind: c_int,
        stream: HipStream_t,
    ) -> HipError_t,

    // --- rocBLAS ------------------------------------------------------
    pub rocblas_create_handle: unsafe extern "C" fn(handle: *mut RocblasHandle) -> RocblasStatus,
    pub rocblas_destroy_handle: unsafe extern "C" fn(handle: RocblasHandle) -> RocblasStatus,
    pub rocblas_set_stream:
        unsafe extern "C" fn(handle: RocblasHandle, stream: HipStream_t) -> RocblasStatus,
    /// `rocblas_sgemv`: single-precision matrix-vector multiply.
    /// Signature matches cuBLAS's `cublasSgemv` one-for-one in argument
    /// order and pointer semantics.
    #[allow(clippy::type_complexity)]
    pub rocblas_sgemv: unsafe extern "C" fn(
        handle: RocblasHandle,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        x: *const f32,
        incx: c_int,
        beta: *const f32,
        y: *mut f32,
        incy: c_int,
    ) -> RocblasStatus,
    /// `rocblas_sgemm`: single-precision matrix-matrix multiply. Same
    /// argument layout as `cublasSgemm`.
    #[allow(clippy::type_complexity)]
    pub rocblas_sgemm: unsafe extern "C" fn(
        handle: RocblasHandle,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
    ) -> RocblasStatus,
}

// Safe because we only hold function pointers, never tied to the loader's
// lifetime at a type level. The `_hip_lib` and `_rocblas_lib` fields keep
// the dynamic libraries alive for the duration of the process.
unsafe impl Send for HipLib {}
unsafe impl Sync for HipLib {}

/// `hipMemcpyKind` values — same numeric encoding as CUDA.
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

impl HipLib {
    /// Attempt to load `libamdhip64` + `librocblas` from the host's
    /// standard library search path. Returns `None` if either library
    /// cannot be found — callers treat that as "ROCm not available".
    fn try_load() -> Option<Self> {
        // Try the base filename first; fall back to versioned SONAMEs
        // that distros expose (ROCm ships e.g. libamdhip64.so.5,
        // libamdhip64.so.6).
        let hip_candidates = ["libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"];
        let rocblas_candidates = [
            "librocblas.so",
            "librocblas.so.4",
            "librocblas.so.3",
            "librocblas.so.0",
        ];

        let hip = first_loadable(&hip_candidates)?;
        let rocblas = first_loadable(&rocblas_candidates)?;

        // SAFETY: the symbol lookups below rely on the `libamdhip64` /
        // `librocblas` ABI staying stable across ROCm minor versions.
        // Each `get` returns `Err` when the symbol is missing; we surface
        // those as `None` so detection gracefully fails instead of
        // panicking.
        macro_rules! sym {
            ($lib:ident, $name:expr) => {
                unsafe {
                    $lib.get::<unsafe extern "C" fn()>($name)
                        .ok()?
                        .into_raw()
                        .into_raw()
                }
            };
        }

        // The `into_raw().into_raw()` chain is correct: `Symbol` has its
        // own `into_raw` that returns `RawSymbol`, and `RawSymbol` has an
        // inherent `into_raw` that exposes the underlying `*mut c_void`.
        // A simple transmute of the cast pointer lets us store the typed
        // function pointer directly in the struct.
        let hip_init = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(c_uint) -> HipError_t>(sym!(
                hip,
                b"hipInit\0"
            ))
        };
        let hip_get_device_count = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut c_int) -> HipError_t>(sym!(
                hip,
                b"hipGetDeviceCount\0"
            ))
        };
        let hip_set_device = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(HipDevice_t) -> HipError_t>(sym!(
                hip,
                b"hipSetDevice\0"
            ))
        };
        let hip_get_device = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut HipDevice_t) -> HipError_t>(sym!(
                hip,
                b"hipGetDevice\0"
            ))
        };
        let hip_device_get_name = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(*mut c_char, c_int, HipDevice_t) -> HipError_t,
            >(sym!(hip, b"hipDeviceGetName\0"))
        };
        let hip_device_get_attribute = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(*mut c_int, c_int, HipDevice_t) -> HipError_t,
            >(sym!(hip, b"hipDeviceGetAttribute\0"))
        };
        let hip_device_total_mem = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut usize, HipDevice_t) -> HipError_t>(
                sym!(hip, b"hipDeviceTotalMem\0"),
            )
        };
        let hip_mem_get_info = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut usize, *mut usize) -> HipError_t>(
                sym!(hip, b"hipMemGetInfo\0"),
            )
        };
        let hip_driver_get_version = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut c_int) -> HipError_t>(sym!(
                hip,
                b"hipDriverGetVersion\0"
            ))
        };
        let hip_runtime_get_version = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut c_int) -> HipError_t>(sym!(
                hip,
                b"hipRuntimeGetVersion\0"
            ))
        };

        let hip_stream_create = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut HipStream_t) -> HipError_t>(sym!(
                hip,
                b"hipStreamCreate\0"
            ))
        };
        let hip_stream_destroy = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(HipStream_t) -> HipError_t>(sym!(
                hip,
                b"hipStreamDestroy\0"
            ))
        };
        let hip_stream_synchronize = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(HipStream_t) -> HipError_t>(sym!(
                hip,
                b"hipStreamSynchronize\0"
            ))
        };

        let hip_malloc = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut HipDevicePtr_t, usize) -> HipError_t>(
                sym!(hip, b"hipMalloc\0"),
            )
        };
        let hip_free = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(HipDevicePtr_t) -> HipError_t>(sym!(
                hip,
                b"hipFree\0"
            ))
        };
        let hip_memcpy = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(HipDevicePtr_t, *const c_void, usize, c_int) -> HipError_t,
            >(sym!(hip, b"hipMemcpy\0"))
        };
        let hip_memcpy_async = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(
                    HipDevicePtr_t,
                    *const c_void,
                    usize,
                    c_int,
                    HipStream_t,
                ) -> HipError_t,
            >(sym!(hip, b"hipMemcpyAsync\0"))
        };

        let rocblas_create_handle = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus>(
                sym!(rocblas, b"rocblas_create_handle\0"),
            )
        };
        let rocblas_destroy_handle = unsafe {
            std::mem::transmute::<_, unsafe extern "C" fn(RocblasHandle) -> RocblasStatus>(sym!(
                rocblas,
                b"rocblas_destroy_handle\0"
            ))
        };
        let rocblas_set_stream = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(RocblasHandle, HipStream_t) -> RocblasStatus,
            >(sym!(rocblas, b"rocblas_set_stream\0"))
        };
        let rocblas_sgemv = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(
                    RocblasHandle,
                    c_int,
                    c_int,
                    c_int,
                    *const f32,
                    *const f32,
                    c_int,
                    *const f32,
                    c_int,
                    *const f32,
                    *mut f32,
                    c_int,
                ) -> RocblasStatus,
            >(sym!(rocblas, b"rocblas_sgemv\0"))
        };
        let rocblas_sgemm = unsafe {
            std::mem::transmute::<
                _,
                unsafe extern "C" fn(
                    RocblasHandle,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    c_int,
                    *const f32,
                    *const f32,
                    c_int,
                    *const f32,
                    c_int,
                    *const f32,
                    *mut f32,
                    c_int,
                ) -> RocblasStatus,
            >(sym!(rocblas, b"rocblas_sgemm\0"))
        };

        Some(Self {
            _hip_lib: hip,
            _rocblas_lib: rocblas,
            hip_init,
            hip_get_device_count,
            hip_set_device,
            hip_get_device,
            hip_device_get_name,
            hip_device_get_attribute,
            hip_device_total_mem,
            hip_mem_get_info,
            hip_driver_get_version,
            hip_runtime_get_version,
            hip_stream_create,
            hip_stream_destroy,
            hip_stream_synchronize,
            hip_malloc,
            hip_free,
            hip_memcpy,
            hip_memcpy_async,
            rocblas_create_handle,
            rocblas_destroy_handle,
            rocblas_set_stream,
            rocblas_sgemv,
            rocblas_sgemm,
        })
    }
}

/// Try loading each candidate SONAME in order; return the first that
/// succeeds. Suppresses the underlying `dlopen` error — our caller only
/// cares whether *some* ROCm library was reachable.
fn first_loadable(candidates: &[&str]) -> Option<Library> {
    for name in candidates {
        // SAFETY: `Library::new` is safe if the path resolves to a real
        // shared object. We rely on the dynamic linker to reject paths
        // that are not valid shared objects.
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Some(lib);
        }
    }
    None
}

/// Access the process-wide HIP library singleton. Returns `None` when no
/// `libamdhip64` or `librocblas` is reachable — treat that as "ROCm not
/// installed" and fall back to a different backend.
pub(crate) fn hip_lib() -> Option<&'static HipLib> {
    static LIB: OnceLock<Option<HipLib>> = OnceLock::new();
    LIB.get_or_init(HipLib::try_load).as_ref()
}

// ---------- error helpers ---------------------------------------------

/// Convert a raw `hipError_t` into our `Result`. 0 is success; anything
/// else is wrapped as `HiveGpuError::HipError` with the status code
/// stringified (we do not yet resolve to a human message because the
/// stringification symbol `hipGetErrorString` is not bound here).
pub(crate) fn hip_check(status: HipError_t, context: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(HiveGpuError::HipError(format!(
            "{context} failed with hipError_t={status}"
        )))
    }
}

/// Same as [`hip_check`] but for rocBLAS status codes.
pub(crate) fn rocblas_check(status: RocblasStatus, context: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(HiveGpuError::RocblasError(format!(
            "{context} failed with rocblas_status={status}"
        )))
    }
}

/// Prefer this over touching [`hip_lib`] directly — returns a
/// `HiveGpuError::RocmError` when the libraries cannot be loaded,
/// matching the rest of the crate's fallible API.
pub(crate) fn require_hip_lib() -> Result<&'static HipLib> {
    hip_lib().ok_or_else(|| {
        HiveGpuError::RocmError(
            "failed to load libamdhip64 / librocblas — ROCm not installed on this host".to_string(),
        )
    })
}
