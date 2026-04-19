//! Metal Native GPU Implementation
//!
//! This module provides Metal Native GPU acceleration for Apple Silicon devices.
//! It includes context management, vector storage, HNSW graph operations, and buffer management.

pub mod buffer_pool;
pub mod context;
pub mod helpers;
pub mod hnsw_graph;
pub mod ivf;
pub mod vector_storage;
pub mod vram_monitor;

pub use buffer_pool::MetalBufferPool;
pub use context::MetalNativeContext;
pub use hnsw_graph::MetalNativeHnswGraph;
pub use ivf::MetalIvfIndex;
pub use vector_storage::MetalNativeVectorStorage;
pub use vram_monitor::MetalVramMonitor;
