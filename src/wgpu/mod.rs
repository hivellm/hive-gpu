//! wgpu GPU Implementation
//!
//! This module provides wgpu GPU acceleration for cross-platform GPU operations.
//! It includes context management, vector storage, HNSW graph operations, and buffer management.

pub mod buffer_pool;
pub mod context;
pub mod helpers;
pub mod hnsw_graph;
pub mod vector_storage;
pub mod vram_monitor;

pub use buffer_pool::WgpuBufferPool;
pub use context::WgpuContext;
pub use hnsw_graph::WgpuHnswGraph;
pub use vector_storage::WgpuVectorStorage;
pub use vram_monitor::WgpuVramMonitor;
