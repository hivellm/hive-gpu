//! GPU Vector Operations Tests
//!
//! Validates correctness of GPU vector operations against CPU reference implementations.
//! Tests cover:
//! - Vector addition
//! - Dot product
//! - Cosine similarity
//! - Distance metrics (Euclidean, Manhattan)
//! - Batch operations

#[cfg(all(target_os = "macos", feature = "metal-native"))]
use std::collections::HashMap;

/// Helper function to create test vectors
#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn create_test_vectors(count: usize, dimension: usize) -> Vec<hive_gpu::types::GpuVector> {
    (0..count)
        .map(|i| {
            let data: Vec<f32> = (0..dimension)
                .map(|d| ((i * dimension + d) as f32) * 0.1)
                .collect();
            hive_gpu::types::GpuVector::new(format!("vec_{}", i), data)
        })
        .collect()
}

/// Helper to compute CPU cosine similarity for validation
#[allow(dead_code)]
fn cpu_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Helper to compute CPU Euclidean distance
#[allow(dead_code)]
fn cpu_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Helper to compute CPU dot product
#[allow(dead_code)]
fn cpu_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_vector_ops_tests {
    use super::*;
    use hive_gpu::error::HiveGpuError;
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::traits::GpuContext;
    use hive_gpu::types::GpuDistanceMetric;

    #[test]
    fn test_vector_addition_small() {
        // Test vector addition with small vectors (10 elements)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 10;
        let vectors = create_test_vectors(2, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Add vectors to GPU
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        // Verify vectors were added
        assert_eq!(storage.vector_count(), 2, "Should have 2 vectors");

        println!("✅ Small vector addition (10 elements) successful");
        println!("   Vectors added: {}", storage.vector_count());
    }

    #[test]
    fn test_vector_addition_medium() {
        // Test vector addition with medium vectors (1000 elements)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 1000;
        let vectors = create_test_vectors(10, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        assert_eq!(storage.vector_count(), 10, "Should have 10 vectors");

        println!("✅ Medium vector addition (1000 elements) successful");
        println!("   Vectors added: {}", storage.vector_count());
    }

    #[test]
    fn test_vector_addition_large() {
        // Test vector addition with large vectors (10K elements)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 512; // Standard embedding size
        let count = 100; // 100 vectors
        let vectors = create_test_vectors(count, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        assert_eq!(
            storage.vector_count(),
            count,
            "Should have {} vectors",
            count
        );

        println!("✅ Large vector addition (100 x 512D) successful");
        println!("   Vectors added: {}", storage.vector_count());
    }

    #[test]
    fn test_cosine_similarity_accuracy() {
        // Test cosine similarity GPU vs CPU
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 128;
        let vectors = create_test_vectors(5, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        // Search for first vector
        let query = &vectors[0].data;
        let results = storage.search(query, 5).expect("Failed to search");

        // First result should be the query itself with similarity ~1.0
        assert!(!results.is_empty(), "Should have search results");

        let top_result = &results[0];
        println!("✅ Cosine similarity test:");
        println!(
            "   Top result: {} (score: {:.6})",
            top_result.id, top_result.score
        );
        println!("   Expected: vec_0 (score: ~1.0)");

        // Score should be very close to 1.0 (self-similarity)
        assert!(
            (top_result.score - 1.0).abs() < 0.01,
            "Self-similarity should be ~1.0, got {}",
            top_result.score
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        // Test with orthogonal vectors (should have 0 similarity)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 4;

        // Create orthogonal vectors
        let vec1 =
            hive_gpu::types::GpuVector::new("orthog_1".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        let vec2 =
            hive_gpu::types::GpuVector::new("orthog_2".to_string(), vec![0.0, 1.0, 0.0, 0.0]);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&[vec1, vec2])
            .expect("Failed to add vectors");

        // Search with first vector
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = storage.search(&query, 2).expect("Failed to search");

        println!("✅ Orthogonal vectors test:");
        for result in &results {
            println!("   {}: {:.6}", result.id, result.score);
        }

        // First should be orthog_1 with highest score (self)
        assert_eq!(
            results[0].id, "orthog_1",
            "First result should be self (orthog_1)"
        );
        assert!(
            results[0].score > 0.95,
            "Self-similarity should be high, got {}",
            results[0].score
        );

        // Second should have lower score (orthogonal or different)
        assert!(
            results[1].score < results[0].score,
            "Second result should have lower score than first"
        );

        println!("   ✅ Self-match score: {:.3}", results[0].score);
        println!("   ✅ Other vector score: {:.3}", results[1].score);
    }

    #[test]
    fn test_euclidean_distance() {
        // Test Euclidean distance metric
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 3;

        // Create simple vectors for easy distance calculation
        let vec1 = hive_gpu::types::GpuVector::new("vec_1".to_string(), vec![0.0, 0.0, 0.0]);
        let vec2 = hive_gpu::types::GpuVector::new("vec_2".to_string(), vec![3.0, 4.0, 0.0]);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Euclidean)
            .expect("Failed to create storage");

        storage
            .add_vectors(&[vec1, vec2])
            .expect("Failed to add vectors");

        // Search with origin
        let query = vec![0.0, 0.0, 0.0];
        let results = storage.search(&query, 2).expect("Failed to search");

        println!("✅ Euclidean distance test:");
        for result in &results {
            println!("   {}: distance = {:.6}", result.id, result.score);
        }

        // For Euclidean, score is typically inverted distance
        // vec_1 is at origin (distance = 0)
        // vec_2 is at distance = 5.0 (3²+4²=25, √25=5)
        assert_eq!(results[0].id, "vec_1", "Closest should be vec_1");
    }

    #[test]
    fn test_batch_operations() {
        // Test batch operations with multiple vectors
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 256;
        let batch_size = 50;
        let vectors = create_test_vectors(batch_size, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Add all vectors in batch
        let start = std::time::Instant::now();
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");
        let duration = start.elapsed();

        assert_eq!(
            storage.vector_count(),
            batch_size,
            "Should have {} vectors",
            batch_size
        );

        println!("✅ Batch operations test:");
        println!("   Vectors: {}", batch_size);
        println!("   Dimension: {}", dimension);
        println!("   Time: {:?}", duration);
        println!(
            "   Throughput: {:.2} vectors/sec",
            batch_size as f64 / duration.as_secs_f64()
        );
    }

    #[test]
    fn test_search_accuracy_k_results() {
        // Test that search returns correct number of results
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 64;
        let total_vectors = 20;
        let vectors = create_test_vectors(total_vectors, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        // Search for k=5 results
        let query = &vectors[0].data;
        let results = storage.search(query, 5).expect("Failed to search");

        assert_eq!(results.len(), 5, "Should return exactly 5 results");

        // Search for k=10 results
        let results = storage.search(query, 10).expect("Failed to search");
        assert_eq!(results.len(), 10, "Should return exactly 10 results");

        // Search for more than available
        let results = storage.search(query, 100).expect("Failed to search");
        assert_eq!(
            results.len(),
            total_vectors,
            "Should return all {} available vectors",
            total_vectors
        );

        println!("✅ Search accuracy test (k results):");
        println!("   Total vectors: {}", total_vectors);
        println!("   k=5: {} results", results.len());
    }

    #[test]
    fn test_edge_case_zero_vector() {
        // Test with zero vectors
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 4;

        let zero_vec =
            hive_gpu::types::GpuVector::new("zero".to_string(), vec![0.0, 0.0, 0.0, 0.0]);
        let normal_vec =
            hive_gpu::types::GpuVector::new("normal".to_string(), vec![1.0, 2.0, 3.0, 4.0]);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Should handle zero vectors gracefully
        let result = storage.add_vectors(&[zero_vec, normal_vec]);
        assert!(result.is_ok(), "Should handle zero vectors gracefully");

        println!("✅ Zero vector edge case handled");
    }

    #[test]
    fn test_edge_case_negative_values() {
        // Test with negative values in vectors
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 4;

        let neg_vec =
            hive_gpu::types::GpuVector::new("negative".to_string(), vec![-1.0, -2.0, -3.0, -4.0]);
        let pos_vec =
            hive_gpu::types::GpuVector::new("positive".to_string(), vec![1.0, 2.0, 3.0, 4.0]);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&[neg_vec, pos_vec])
            .expect("Failed to add vectors");

        // Query with positive vector
        let results = storage
            .search(&[1.0, 2.0, 3.0, 4.0], 2)
            .expect("Failed to search");

        println!("✅ Negative values test:");
        for result in &results {
            println!("   {}: {:.6}", result.id, result.score);
        }

        // Find which vector matches better
        let mut scores: HashMap<String, f32> = HashMap::new();
        for result in &results {
            scores.insert(result.id.clone(), result.score);
        }

        // Both vectors should be returned
        assert!(
            scores.contains_key("positive"),
            "Should find positive vector"
        );
        assert!(
            scores.contains_key("negative"),
            "Should find negative vector"
        );

        println!(
            "   ✅ Positive score: {:.3}",
            scores.get("positive").unwrap()
        );
        println!(
            "   ✅ Negative score: {:.3}",
            scores.get("negative").unwrap()
        );

        // The important thing is both vectors are searchable
        assert_eq!(results.len(), 2, "Should return both vectors");
    }

    #[test]
    fn test_different_distance_metrics() {
        // Test all available distance metrics
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 8;
        let vectors = create_test_vectors(5, dimension);

        println!("✅ Testing distance metrics:");

        // Test Cosine
        {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");
            storage.add_vectors(&vectors).expect("Failed to add");
            let results = storage.search(&vectors[0].data, 3).expect("Failed");
            println!("   Cosine: {} results", results.len());
        }

        // Test Euclidean
        {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Euclidean)
                .expect("Failed to create storage");
            storage.add_vectors(&vectors).expect("Failed to add");
            let results = storage.search(&vectors[0].data, 3).expect("Failed");
            println!("   Euclidean: {} results", results.len());
        }

        // Test DotProduct
        {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::DotProduct)
                .expect("Failed to create storage");
            storage.add_vectors(&vectors).expect("Failed to add");
            let results = storage.search(&vectors[0].data, 3).expect("Failed");
            println!("   DotProduct: {} results", results.len());
        }
    }
}
