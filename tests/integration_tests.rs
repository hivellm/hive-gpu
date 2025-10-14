//! Integration tests for hive-gpu
//!
//! These tests verify that all components work together correctly
//! and that the GPU operations produce expected results.

use hive_gpu::{
    GpuVector, GpuDistanceMetric, GpuSearchResult, HnswConfig,
    GpuBackend, GpuVectorStorage, GpuContext,
};

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_tests {
    use super::*;
    use hive_gpu::metal::{MetalNativeContext, MetalNativeVectorStorage};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_metal_basic_operations() {
        let context = MetalNativeContext::new().expect("Failed to create Metal context");
        let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Create test vectors
        let vectors: Vec<GpuVector> = (0..100)
            .map(|i| GpuVector {
                id: format!("vec_{}", i),
                data: vec![i as f32; 128],
                metadata: HashMap::new(),
            })
            .collect();

        // Add vectors
        let indices = storage.add_vectors(&vectors).expect("Failed to add vectors");
        assert_eq!(indices.len(), 100);
        assert_eq!(storage.vector_count(), 100);

        // Search for similar vectors
        let query = vec![50.0; 128];
        let results = storage.search(&query, 10).expect("Failed to search");
        assert_eq!(results.len(), 10);

        // Verify results are sorted by similarity
        for i in 1..results.len() {
            assert!(results[i-1].score <= results[i].score);
        }

        // Test vector retrieval
        let retrieved = storage.get_vector("vec_50").expect("Failed to get vector");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "vec_50");

        // Test vector removal
        storage.remove_vectors(&["vec_50".to_string()]).expect("Failed to remove vector");
        let retrieved_after_removal = storage.get_vector("vec_50").expect("Failed to get vector");
        assert!(retrieved_after_removal.is_none());
    }

    #[tokio::test]
    async fn test_metal_distance_metrics() {
        let context = MetalNativeContext::new().expect("Failed to create Metal context");
        
        // Test Cosine similarity
        let mut cosine_storage = context.create_storage(3, GpuDistanceMetric::Cosine)
            .expect("Failed to create cosine storage");
        
        let vectors = vec![
            GpuVector {
                id: "vec1".to_string(),
                data: vec![1.0, 0.0, 0.0],
                metadata: HashMap::new(),
            },
            GpuVector {
                id: "vec2".to_string(),
                data: vec![0.0, 1.0, 0.0],
                metadata: HashMap::new(),
            },
        ];
        
        cosine_storage.add_vectors(&vectors).expect("Failed to add vectors");
        
        let query = vec![1.0, 0.0, 0.0];
        let results = cosine_storage.search(&query, 2).expect("Failed to search");
        
        // vec1 should be most similar (cosine similarity = 1.0)
        assert_eq!(results[0].id, "vec1");
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_metal_hnsw_construction() {
        let context = MetalNativeContext::new().expect("Failed to create Metal context");
        
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_connections: 32,
        };
        
        let mut hnsw = context.create_storage_with_config(128, GpuDistanceMetric::Cosine, config)
            .expect("Failed to create HNSW storage");
        
        // Create vectors for HNSW construction
        let vectors: Vec<GpuVector> = (0..1000)
            .map(|i| GpuVector {
                id: format!("hnsw_vec_{}", i),
                data: vec![i as f32; 128],
                metadata: HashMap::new(),
            })
            .collect();
        
        hnsw.add_vectors(&vectors).expect("Failed to add vectors to HNSW");
        
        // Test HNSW search
        let query = vec![500.0; 128];
        let results = hnsw.search(&query, 10).expect("Failed to search HNSW");
        
        assert_eq!(results.len(), 10);
        // Results should be well-distributed around the query
        assert!(results.iter().any(|r| r.id.contains("500")));
    }

    #[tokio::test]
    async fn test_metal_vram_monitoring() {
        let context = MetalNativeContext::new().expect("Failed to create Metal context");
        
        // Get initial memory stats
        let initial_stats = context.memory_stats();
        assert!(initial_stats.available > 0);
        
        // Create storage and add vectors
        let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");
        
        let vectors: Vec<GpuVector> = (0..1000)
            .map(|i| GpuVector {
                id: format!("vram_vec_{}", i),
                data: vec![i as f32; 512],
                metadata: HashMap::new(),
            })
            .collect();
        
        storage.add_vectors(&vectors).expect("Failed to add vectors");
        
        // Check memory usage increased
        let stats_after = context.memory_stats();
        assert!(stats_after.total_allocated > initial_stats.total_allocated);
    }

    #[tokio::test]
    async fn test_metal_error_handling() {
        let context = MetalNativeContext::new().expect("Failed to create Metal context");
        
        // Test dimension mismatch
        let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");
        
        let wrong_dimension_vector = GpuVector {
            id: "wrong_dim".to_string(),
            data: vec![1.0; 64], // Wrong dimension
            metadata: HashMap::new(),
        };
        
        let result = storage.add_vectors(&[wrong_dimension_vector]);
        assert!(result.is_err());
        
        // Test duplicate ID
        let vector1 = GpuVector {
            id: "duplicate".to_string(),
            data: vec![1.0; 128],
            metadata: HashMap::new(),
        };
        
        let vector2 = GpuVector {
            id: "duplicate".to_string(), // Same ID
            data: vec![2.0; 128],
            metadata: HashMap::new(),
        };
        
        storage.add_vectors(&[vector1]).expect("Failed to add first vector");
        let result = storage.add_vectors(&[vector2]);
        assert!(result.is_err());
    }
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use hive_gpu::cuda::{CudaContext, CudaVectorStorage};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_cuda_basic_operations() {
        let context = CudaContext::new().expect("Failed to create CUDA context");
        let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Test basic operations (placeholder implementation)
        let vectors = vec![
            GpuVector {
                id: "cuda_vec1".to_string(),
                data: vec![1.0; 128],
                metadata: HashMap::new(),
            },
        ];

        let indices = storage.add_vectors(&vectors).expect("Failed to add vectors");
        assert_eq!(indices.len(), 1);
        assert_eq!(storage.vector_count(), 1);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use super::*;
    use hive_gpu::wgpu::{WgpuContext, WgpuVectorStorage};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_wgpu_basic_operations() {
        let context = WgpuContext::new().expect("Failed to create wgpu context");
        let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Test basic operations (placeholder implementation)
        let vectors = vec![
            GpuVector {
                id: "wgpu_vec1".to_string(),
                data: vec![1.0; 128],
                metadata: HashMap::new(),
            },
        ];

        let indices = storage.add_vectors(&vectors).expect("Failed to add vectors");
        assert_eq!(indices.len(), 1);
        assert_eq!(storage.vector_count(), 1);
    }
}

// Cross-backend tests
mod cross_backend_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_gpu_vector_creation() {
        let vector = GpuVector {
            id: "test_vector".to_string(),
            data: vec![1.0, 2.0, 3.0],
            metadata: {
                let mut map = HashMap::new();
                map.insert("category".to_string(), "test".to_string());
                map
            },
        };

        assert_eq!(vector.id, "test_vector");
        assert_eq!(vector.data.len(), 3);
        assert_eq!(vector.metadata.get("category"), Some(&"test".to_string()));
    }

    #[test]
    fn test_distance_metrics() {
        // Test metric conversion
        assert_eq!(GpuDistanceMetric::Cosine as u32, 0);
        assert_eq!(GpuDistanceMetric::Euclidean as u32, 1);
        assert_eq!(GpuDistanceMetric::DotProduct as u32, 2);
    }

    #[test]
    fn test_hnsw_config() {
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_connections: 32,
        };

        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
        assert_eq!(config.max_connections, 32);
    }

    #[test]
    fn test_gpu_search_result() {
        let result = GpuSearchResult {
            id: "result_1".to_string(),
            score: 0.95,
            index: 42,
        };

        assert_eq!(result.id, "result_1");
        assert!((result.score - 0.95).abs() < 0.001);
        assert_eq!(result.index, 42);
    }
}
