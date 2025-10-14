//! WGSL Shader Management
//!
//! This module provides loading and management of WGSL shaders for cross-platform GPU operations.

use crate::error::{Result, HiveGpuError};

/// WGSL shader manager for loading and compiling WGSL shaders
pub struct WgslShaderManager {
    /// Loaded shader sources
    shader_sources: std::collections::HashMap<String, String>,
}

impl WgslShaderManager {
    /// Create a new WGSL shader manager
    pub fn new() -> Self {
        Self {
            shader_sources: Self::load_wgsl_shaders(),
        }
    }

    /// Get a specific shader source
    pub fn get_shader(&self, name: &str) -> Option<&str> {
        self.shader_sources.get(name).map(|s| s.as_str())
    }

    /// Get all available shader names
    pub fn get_shader_names(&self) -> Vec<String> {
        self.shader_sources.keys().cloned().collect()
    }

    /// Load WGSL shaders from embedded sources
    fn load_wgsl_shaders() -> std::collections::HashMap<String, String> {
        let mut shaders = std::collections::HashMap::new();
        
        // Load individual WGSL shaders
        // These will be populated when we migrate the shaders from vectorizer
        shaders.insert("similarity".to_string(), include_str!("../shaders/similarity.wgsl").to_string());
        shaders.insert("distance".to_string(), include_str!("../shaders/distance.wgsl").to_string());
        shaders.insert("dot_product".to_string(), include_str!("../shaders/dot_product.wgsl").to_string());
        shaders.insert("hnsw_construction".to_string(), include_str!("../shaders/hnsw_construction.wgsl").to_string());
        shaders.insert("hnsw_navigation".to_string(), include_str!("../shaders/hnsw_navigation.wgsl").to_string());
        shaders.insert("batch_construction".to_string(), include_str!("../shaders/batch_construction.wgsl").to_string());
        
        shaders
    }

    /// Validate all shaders
    pub fn validate_all_shaders(&self) -> Result<()> {
        for (name, source) in &self.shader_sources {
            if source.is_empty() {
                return Err(HiveGpuError::ShaderCompilationFailed(
                    format!("Empty shader source for: {}", name)
                ));
            }

            // Basic WGSL validation
            if !source.contains("@compute") && !source.contains("@vertex") && !source.contains("@fragment") {
                return Err(HiveGpuError::ShaderCompilationFailed(
                    format!("Invalid WGSL shader: {} (missing entry point)", name)
                ));
            }
        }

        Ok(())
    }

    /// Get shader compilation info
    pub fn get_compilation_info(&self) -> ShaderCompilationInfo {
        ShaderCompilationInfo {
            total_shaders: self.shader_sources.len(),
            shader_names: self.get_shader_names(),
            total_size: self.shader_sources.values().map(|s| s.len()).sum(),
        }
    }
}

impl Default for WgslShaderManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Shader compilation information
#[derive(Debug, Clone)]
pub struct ShaderCompilationInfo {
    /// Total number of shaders
    pub total_shaders: usize,
    /// Names of all shaders
    pub shader_names: Vec<String>,
    /// Total size of all shaders in bytes
    pub total_size: usize,
}
