//! ROCm / HIP backend for AMD GPUs.
//!
//! Feature-gated behind `rocm` and cfg-gated to `target_os = "linux"`.
//! Built on hand-rolled FFI loaded via `libloading` so the crate compiles
//! cleanly on hosts without ROCm installed.
//!
//! ⚠️ AUTHORED BLIND — see `phase3b_add-rocm-backend` for the validation
//! task that a Linux + AMD maintainer needs to run before the backend
//! ships.

#![cfg(all(feature = "rocm", target_os = "linux"))]

pub mod context;
pub mod ffi;
pub mod ivf;
pub mod vector_storage;

pub use context::RocmContext;
pub use ivf::RocmIvfIndex;
pub use vector_storage::RocmVectorStorage;
