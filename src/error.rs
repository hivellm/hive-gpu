//! Error types for Hive GPU

use thiserror::Error;

/// Main error type for Hive GPU operations
#[derive(Error, Debug)]
pub enum HiveGpuError {
    /// Invalid vector dimension
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// VRAM limit exceeded
    #[error("VRAM limit exceeded: requested {requested}, limit {limit}")]
    VramLimitExceeded { requested: usize, limit: usize },

    /// GPU operation failed
    #[error("GPU operation failed: {0}")]
    GpuOperationFailed(String),

    /// No GPU device available
    #[error("No GPU device available")]
    NoDeviceAvailable,

    /// Buffer allocation failed
    #[error("Buffer allocation failed: {0}")]
    BufferAllocationFailed(String),

    /// Device initialization failed
    #[error("Device initialization failed: {0}")]
    DeviceInitializationFailed(String),

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),

    /// Search operation failed
    #[error("Search operation failed: {0}")]
    SearchFailed(String),

    /// Vector not found
    #[error("Vector not found: {0}")]
    VectorNotFound(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// CUDA driver or runtime failure
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// cuBLAS call failure
    #[error("cuBLAS error: {0}")]
    CublasError(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

/// Result type alias for Hive GPU operations
pub type Result<T> = std::result::Result<T, HiveGpuError>;
