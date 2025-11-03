//! GPU Monitoring and VRAM Management
//!
//! This module provides monitoring capabilities for GPU memory usage,
//! VRAM validation, and performance statistics.

pub mod performance_monitor;
pub mod vram_monitor;

pub use crate::traits::{VramBufferInfo, VramStats};
pub use performance_monitor::{BenchmarkResult, PerformanceMonitor, PerformanceStats};
pub use vram_monitor::{VramBenchmarkResult, VramMonitor, VramValidator};
