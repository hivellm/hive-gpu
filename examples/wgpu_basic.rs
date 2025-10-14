//! Basic wgpu example
//!
//! This example demonstrates basic usage of hive-gpu with wgpu
//! for cross-platform GPU acceleration.

use hive_gpu::{
    wgpu::{WgpuContext, WgpuVectorStorage},
    GpuVector, GpuDistanceMetric, GpuSearchResult,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Hive-GPU wgpu Basic Example");
    
    // Create wgpu context
    let context = WgpuContext::new()?;
    println!("✅ wgpu context created: {}", context.device_name());
    println!("🔧 Backend: {}", context.backend());
    
    // Create vector storage
    let mut storage = context.create_storage(128, GpuDistanceMetric::DotProduct)?;
    println!("✅ Vector storage created with dimension 128");
    
    // Create some test vectors
    let vectors = vec![
        GpuVector {
            id: "embedding_1".to_string(),
            data: vec![1.0; 128],
            metadata: {
                let mut map = HashMap::new();
                map.insert("text".to_string(), "Hello world".to_string());
                map.insert("language".to_string(), "en".to_string());
                map
            },
        },
        GpuVector {
            id: "embedding_2".to_string(),
            data: vec![2.0; 128],
            metadata: {
                let mut map = HashMap::new();
                map.insert("text".to_string(), "Machine learning".to_string());
                map.insert("language".to_string(), "en".to_string());
                map
            },
        },
        GpuVector {
            id: "embedding_3".to_string(),
            data: vec![3.0; 128],
            metadata: {
                let mut map = HashMap::new();
                map.insert("text".to_string(), "Bonjour le monde".to_string());
                map.insert("language".to_string(), "fr".to_string());
                map
            },
        },
    ];
    
    // Add vectors to storage
    let indices = storage.add_vectors(&vectors)?;
    println!("✅ Added {} vectors to storage", indices.len());
    println!("📊 Total vectors in storage: {}", storage.vector_count());
    
    // Search for similar vectors
    let query = vec![1.5; 128]; // Query vector
    let results = storage.search(&query, 5)?;
    
    println!("🔍 Search results for query vector:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, result.id, result.score);
    }
    
    // Test vector retrieval
    if let Some(retrieved_vector) = storage.get_vector("embedding_1")? {
        println!("📄 Retrieved vector: {}", retrieved_vector.id);
        println!("   Text: {}", retrieved_vector.metadata.get("text").unwrap_or(&"N/A".to_string()));
        println!("   Language: {}", retrieved_vector.metadata.get("language").unwrap_or(&"N/A".to_string()));
    }
    
    // Test vector removal
    storage.remove_vectors(&["embedding_3".to_string()])?;
    println!("🗑️ Removed embedding_3 from storage");
    
    // Verify removal
    let retrieved_after_removal = storage.get_vector("embedding_3")?;
    assert!(retrieved_after_removal.is_none());
    println!("✅ Vector removal verified");
    
    // Test batch operations
    let batch_vectors = vec![
        GpuVector {
            id: "batch_1".to_string(),
            data: vec![4.0; 128],
            metadata: HashMap::new(),
        },
        GpuVector {
            id: "batch_2".to_string(),
            data: vec![5.0; 128],
            metadata: HashMap::new(),
        },
    ];
    
    let batch_indices = storage.add_vectors(&batch_vectors)?;
    println!("📦 Added {} vectors in batch", batch_indices.len());
    
    // Final search
    let final_results = storage.search(&vec![2.5; 128], 3)?;
    println!("🔍 Final search results:");
    for (i, result) in final_results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, result.id, result.score);
    }
    
    println!("🎉 wgpu example completed successfully!");
    Ok(())
}
