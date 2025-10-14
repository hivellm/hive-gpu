//! GPU Backend Detection and Management
//!
//! This module provides automatic detection and management of different GPU backends:
//! - Metal (Apple Silicon)
//! - CUDA (NVIDIA)
//! - CPU (fallback)

pub mod detector;

pub use detector::{GpuBackendType, detect_available_backends, select_best_backend};
