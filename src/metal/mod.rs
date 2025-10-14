//! Metal Native GPU Implementation
//!
//! This module provides Metal Native GPU acceleration for Apple Silicon devices.
//! It includes context management, vector storage, HNSW graph operations, and buffer management.

pub mod context;
pub mod vector_storage;
pub mod hnsw_graph;
pub mod buffer_pool;
pub mod vram_monitor;
pub mod helpers;

pub use context::MetalNativeContext;
pub use vector_storage::MetalNativeVectorStorage;
pub use hnsw_graph::MetalNativeHnswGraph;
pub use buffer_pool::MetalBufferPool;
pub use vram_monitor::MetalVramMonitor;
