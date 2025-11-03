//! Metal Shader Management
//!
//! This module provides loading and management of Metal shaders for Apple Silicon.

use crate::error::{HiveGpuError, Result};

/// Metal shader manager for loading and compiling Metal shaders
pub struct MetalShaderManager {
    /// Loaded shader source
    shader_source: String,
}

impl MetalShaderManager {
    /// Create a new Metal shader manager
    pub fn new() -> Self {
        Self {
            shader_source: Self::load_metal_shaders(),
        }
    }

    /// Get the Metal shader source
    pub fn get_shader_source(&self) -> &str {
        &self.shader_source
    }

    /// Load Metal shaders from embedded source
    fn load_metal_shaders() -> String {
        // This will be replaced with the actual Metal shader source
        // when we migrate the shaders from the vectorizer project
        include_str!("../shaders/metal_hnsw.metal").to_string()
    }

    /// Validate shader source
    pub fn validate_shader(&self) -> Result<()> {
        if self.shader_source.is_empty() {
            return Err(HiveGpuError::ShaderCompilationFailed(
                "Empty shader source".to_string(),
            ));
        }

        // Basic validation - check for required functions
        let required_functions = [
            "cosine_similarity",
            "euclidean_distance",
            "dot_product",
            "hnsw_construction",
            "hnsw_search",
        ];

        for function in &required_functions {
            if !self.shader_source.contains(function) {
                return Err(HiveGpuError::ShaderCompilationFailed(format!(
                    "Missing required function: {}",
                    function
                )));
            }
        }

        Ok(())
    }
}

impl Default for MetalShaderManager {
    fn default() -> Self {
        Self::new()
    }
}
