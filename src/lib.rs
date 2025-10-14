//! Hive GPU - High-performance GPU acceleration for vector operations
//!
//! This crate provides GPU-accelerated vector operations using Metal (Apple Silicon)
//! and CUDA (NVIDIA) backends for maximum performance in vector similarity search.

#![allow(warnings)]

pub mod error;
pub mod types;
pub mod traits;

// Re-export commonly used types
pub use error::{HiveGpuError, Result};
pub use types::{
    GpuVector, GpuDistanceMetric, GpuSearchResult, GpuDeviceInfo, 
    GpuCapabilities, GpuMemoryStats, HnswConfig, VectorMetadata
};
pub use traits::{
    GpuBackend, GpuVectorStorage, GpuContext, GpuBufferManager, 
    GpuMonitor, GpuBuffer, BufferType, BufferPoolStats, VramStats, VramBufferInfo
};

// Platform-specific modules
#[cfg(all(target_os = "macos", feature = "metal-native"))]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// Backend detection
pub mod backends;

// Monitoring utilities
pub mod monitoring;

// Shader management
pub mod shaders;

// Utility functions
pub mod utils;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Include test modules
// #[cfg(test)]
// mod tests;
