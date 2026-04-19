//! Intel GPU backend via Vulkan Compute.
//!
//! Gated behind the `intel` Cargo feature and compiled on Linux or
//! Windows. The Vulkan loader is dlopened at runtime by `ash`, so the
//! crate builds on any host — the feature only becomes functional when a
//! Vulkan driver is present.
//!
//! ⚠️ AUTHORED BLIND — no Intel Arc / Battlemage hardware was available
//! at write time. See `phase3c_add-intel-backend`.

#![cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))]
// Vulkan dispatch boilerplate benefits from pedantic-style rewrites but
// rewriting every `(x + 255) / 256` to `.div_ceil(256)` and every
// `slice.len() * size_of::<T>()` to `size_of_val` obscures the intent
// for a backend that is still evolving. Re-enable these once the Intel
// maintainer validates the code on real hardware.
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::manual_is_multiple_of)]

pub mod context;
pub mod ivf;
pub mod vector_storage;

pub use context::IntelContext;
pub use ivf::IntelIvfIndex;
pub use vector_storage::IntelVectorStorage;
