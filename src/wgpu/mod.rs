//! wgpu GPU Implementation
//!
//! This module provides wgpu GPU acceleration for cross-platform GPU operations.
//! It includes context management, vector storage, HNSW graph operations, and buffer management.

pub mod context;
pub mod vector_storage;
pub mod hnsw_graph;
pub mod buffer_pool;
pub mod vram_monitor;
pub mod helpers;

pub use context::WgpuContext;
pub use vector_storage::WgpuVectorStorage;
pub use hnsw_graph::WgpuHnswGraph;
pub use buffer_pool::WgpuBufferPool;
pub use vram_monitor::WgpuVramMonitor;
