//! Shader Management
//!
//! This module provides management and loading of GPU shaders for different platforms.

pub mod metal_shaders;
pub mod wgsl_shaders;

pub use metal_shaders::MetalShaderManager;
pub use wgsl_shaders::WgslShaderManager;
