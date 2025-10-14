//! GPU Monitoring and VRAM Management
//!
//! This module provides monitoring capabilities for GPU memory usage,
//! VRAM validation, and performance statistics.

pub mod vram_monitor;
pub mod performance_monitor;

pub use vram_monitor::{VramMonitor, VramValidator, VramBenchmarkResult};
pub use crate::traits::{VramStats, VramBufferInfo};
pub use performance_monitor::{PerformanceMonitor, PerformanceStats, BenchmarkResult};
