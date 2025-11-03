//! Basic Metal Native example
//!
//! This example demonstrates basic usage of hive-gpu with Metal Native
//! on Apple Silicon devices.

use hive_gpu::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Hive-GPU Metal Native Basic Example");

    // Check if Metal is available
    #[cfg(all(target_os = "macos", feature = "metal-native"))]
    {
        use hive_gpu::GpuContext;
        use hive_gpu::metal::MetalNativeContext;

        // Create Metal context
        let context = MetalNativeContext::new()?;
        println!("✅ Metal context created: {}", context.device_name());

        // Create vector storage
        let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
        println!("✅ Vector storage created with dimension 512");

        // Create some test vectors
        let vectors = vec![
            GpuVector {
                id: "document_1".to_string(),
                data: vec![1.0; 512],
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("title".to_string(), "Introduction to AI".to_string());
                    map.insert("category".to_string(), "technology".to_string());
                    map
                },
            },
            GpuVector {
                id: "document_2".to_string(),
                data: vec![2.0; 512],
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("title".to_string(), "Machine Learning Basics".to_string());
                    map.insert("category".to_string(), "technology".to_string());
                    map
                },
            },
            GpuVector {
                id: "document_3".to_string(),
                data: vec![3.0; 512],
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("title".to_string(), "Cooking Recipes".to_string());
                    map.insert("category".to_string(), "lifestyle".to_string());
                    map
                },
            },
        ];

        // Add vectors to storage
        let indices = storage.add_vectors(&vectors)?;
        println!("✅ Added {} vectors to storage", indices.len());
        println!("📊 Total vectors in storage: {}", storage.vector_count());

        // Search for similar vectors
        let query = vec![1.5; 512]; // Query vector
        let results = storage.search(&query, 5)?;

        println!("🔍 Search results for query vector:");
        for (i, result) in results.iter().enumerate() {
            println!("  {}. {} (score: {:.4})", i + 1, result.id, result.score);
        }

        // Test vector retrieval
        if let Some(retrieved_vector) = storage.get_vector("document_1")? {
            println!("📄 Retrieved vector: {}", retrieved_vector.id);
            println!(
                "   Title: {}",
                retrieved_vector
                    .metadata
                    .get("title")
                    .unwrap_or(&"N/A".to_string())
            );
            println!(
                "   Category: {}",
                retrieved_vector
                    .metadata
                    .get("category")
                    .unwrap_or(&"N/A".to_string())
            );
        }

        // Test vector removal
        storage.remove_vectors(&["document_3".to_string()])?;
        println!("🗑️ Removed document_3 from storage");

        // Verify removal
        let retrieved_after_removal = storage.get_vector("document_3")?;
        assert!(retrieved_after_removal.is_none());
        println!("✅ Vector removal verified");

        // Test batch operations
        let batch_vectors = vec![
            GpuVector {
                id: "batch_1".to_string(),
                data: vec![4.0; 512],
                metadata: HashMap::new(),
            },
            GpuVector {
                id: "batch_2".to_string(),
                data: vec![5.0; 512],
                metadata: HashMap::new(),
            },
        ];

        let batch_indices = storage.add_vectors(&batch_vectors)?;
        println!("📦 Added {} vectors in batch", batch_indices.len());

        // Final search
        let final_results = storage.search(&vec![2.5; 512], 3)?;
        println!("🔍 Final search results:");
        for (i, result) in final_results.iter().enumerate() {
            println!("  {}. {} (score: {:.4})", i + 1, result.id, result.score);
        }

        println!("🎉 Example completed successfully!");
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-native")))]
    {
        println!("⚠️ Metal Native is not available on this platform");
        println!("This example requires macOS with the 'metal-native' feature enabled");
        println!("To run this example:");
        println!("  cargo run --example metal_basic --features metal-native");
    }

    Ok(())
}
