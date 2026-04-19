//! Hive GPU - High-performance GPU acceleration for vector operations
//!
//! This crate provides GPU-accelerated vector operations using Metal (Apple Silicon)
//! and CUDA (NVIDIA) backends for maximum performance in vector similarity search.

pub mod error;
pub mod traits;
pub mod types;

// Re-export commonly used types
pub use error::{HiveGpuError, Result};
pub use traits::{
    BufferPoolStats, BufferType, GpuBackend, GpuBuffer, GpuBufferManager, GpuContext, GpuMonitor,
    GpuVectorStorage, VramBufferInfo, VramStats,
};
pub use types::{
    GpuCapabilities, GpuDeviceInfo, GpuDistanceMetric, GpuMemoryStats, GpuSearchResult, GpuVector,
    HnswConfig, VectorMetadata,
};

// Platform-specific modules
#[cfg(all(target_os = "macos", feature = "metal-native"))]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

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
