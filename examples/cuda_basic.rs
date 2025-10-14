//! Basic CUDA example
//!
//! This example demonstrates basic usage of hive-gpu with CUDA
//! on Linux/Windows systems with NVIDIA GPUs.

use hive_gpu::{
    cuda::{CudaContext, CudaVectorStorage},
    GpuVector, GpuDistanceMetric, GpuSearchResult,
    GpuContext,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Hive-GPU CUDA Basic Example");
    
    // Create CUDA context
    let context = CudaContext::new()?;
    println!("✅ CUDA context created: {}", context.device_name());
    println!("🔧 Compute capability: {}.{}", 
        context.compute_capability().0, 
        context.compute_capability().1);
    
    // Create vector storage
    let mut storage = context.create_storage(256, GpuDistanceMetric::Euclidean)?;
    println!("✅ Vector storage created with dimension 256");
    
    // Create some test vectors
    let vectors = vec![
        GpuVector {
            id: "image_1".to_string(),
            data: vec![1.0; 256],
            metadata: {
                let mut map = HashMap::new();
                map.insert("filename".to_string(), "cat.jpg".to_string());
                map.insert("type".to_string(), "image".to_string());
                map
            },
        },
        GpuVector {
            id: "image_2".to_string(),
            data: vec![2.0; 256],
            metadata: {
                let mut map = HashMap::new();
                map.insert("filename".to_string(), "dog.jpg".to_string());
                map.insert("type".to_string(), "image".to_string());
                map
            },
        },
        GpuVector {
            id: "image_3".to_string(),
            data: vec![3.0; 256],
            metadata: {
                let mut map = HashMap::new();
                map.insert("filename".to_string(), "car.jpg".to_string());
                map.insert("type".to_string(), "image".to_string());
                map
            },
        },
    ];
    
    // Add vectors to storage
    let indices = storage.add_vectors(&vectors)?;
    println!("✅ Added {} vectors to storage", indices.len());
    println!("📊 Total vectors in storage: {}", storage.vector_count());
    
    // Search for similar vectors
    let query = vec![1.5; 256]; // Query vector
    let results = storage.search(&query, 5)?;
    
    println!("🔍 Search results for query vector:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, result.id, result.score);
    }
    
    // Test vector retrieval
    if let Some(retrieved_vector) = storage.get_vector("image_1")? {
        println!("📄 Retrieved vector: {}", retrieved_vector.id);
        println!("   Filename: {}", retrieved_vector.metadata.get("filename").unwrap_or(&"N/A".to_string()));
        println!("   Type: {}", retrieved_vector.metadata.get("type").unwrap_or(&"N/A".to_string()));
    }
    
    // Test vector removal
    storage.remove_vectors(&["image_3".to_string()])?;
    println!("🗑️ Removed image_3 from storage");
    
    // Verify removal
    let retrieved_after_removal = storage.get_vector("image_3")?;
    assert!(retrieved_after_removal.is_none());
    println!("✅ Vector removal verified");
    
    // Test batch operations
    let batch_vectors = vec![
        GpuVector {
            id: "batch_1".to_string(),
            data: vec![4.0; 256],
            metadata: HashMap::new(),
        },
        GpuVector {
            id: "batch_2".to_string(),
            data: vec![5.0; 256],
            metadata: HashMap::new(),
        },
    ];
    
    let batch_indices = storage.add_vectors(&batch_vectors)?;
    println!("📦 Added {} vectors in batch", batch_indices.len());
    
    // Final search
    let final_results = storage.search(&vec![2.5; 256], 3)?;
    println!("🔍 Final search results:");
    for (i, result) in final_results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, result.id, result.score);
    }
    
    println!("🎉 CUDA example completed successfully!");
    Ok(())
}
