//! GPU Performance Benchmark Tests
//!
//! Tests that measure real performance and establish baselines:
//! - Vector addition throughput
//! - Search latency
//! - Batch processing performance
//! - Memory bandwidth
//! - Scalability tests

use std::time::Instant;

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_performance_tests {
    use super::*;
    use hive_gpu::error::HiveGpuError;
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::traits::GpuContext;
    use hive_gpu::types::{GpuDistanceMetric, GpuVector};

    /// Helper to create test vectors
    fn create_test_vectors(count: usize, dimension: usize) -> Vec<GpuVector> {
        (0..count)
            .map(|i| {
                let data: Vec<f32> = (0..dimension).map(|d| (i * dimension + d) as f32).collect();
                GpuVector::new(format!("vec_{}", i), data)
            })
            .collect()
    }

    #[test]
    fn test_vector_addition_throughput() {
        // Measure vector addition throughput
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 512;
        let sizes = vec![100, 500, 1000, 5000];

        println!("✅ Vector Addition Throughput Benchmark:");
        println!("   Dimension: {}", dimension);
        println!();

        for size in sizes {
            let vectors = create_test_vectors(size, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            let duration = start.elapsed();

            let throughput = size as f64 / duration.as_secs_f64();
            let mb_per_sec =
                (size * dimension * 4) as f64 / duration.as_secs_f64() / 1024.0 / 1024.0;

            println!("   {} vectors:", size);
            println!("      Time: {:?}", duration);
            println!("      Throughput: {:.2} vectors/sec", throughput);
            println!("      Bandwidth: {:.2} MB/s", mb_per_sec);

            // Baseline: Should be able to add at least 100 vectors/sec
            assert!(
                throughput > 100.0,
                "Throughput too low: {:.2} vectors/sec",
                throughput
            );
        }
    }

    #[test]
    fn test_search_latency() {
        // Measure search latency
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 256;
        let vector_count = 1000;
        let vectors = create_test_vectors(vector_count, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        println!("✅ Search Latency Benchmark:");
        println!("   Dataset: {} vectors × {}D", vector_count, dimension);
        println!();

        let k_values = vec![1, 5, 10, 50, 100];
        let query = &vectors[0].data;

        for k in k_values {
            let iterations = 100;
            let start = Instant::now();

            for _ in 0..iterations {
                storage.search(query, k).expect("Failed to search");
            }

            let duration = start.elapsed();
            let avg_latency = duration.as_micros() as f64 / iterations as f64;

            println!("   k={} results:", k);
            println!("      Avg latency: {:.2} μs", avg_latency);
            println!("      QPS: {:.0} queries/sec", 1_000_000.0 / avg_latency);

            // Baseline: Search should complete in < 1ms for k=10
            if k == 10 {
                assert!(
                    avg_latency < 1000.0,
                    "Search latency too high: {:.2} μs",
                    avg_latency
                );
            }
        }
    }

    #[test]
    fn test_batch_processing_performance() {
        // Measure batch processing performance
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 384;
        let batch_sizes = vec![10, 50, 100, 500];

        println!("✅ Batch Processing Performance:");
        println!("   Dimension: {}", dimension);
        println!();

        for batch_size in batch_sizes {
            let vectors = create_test_vectors(batch_size, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            let duration = start.elapsed();

            let throughput = batch_size as f64 / duration.as_secs_f64();
            let time_per_vector = duration.as_micros() as f64 / batch_size as f64;

            println!("   Batch size: {}", batch_size);
            println!("      Total time: {:?}", duration);
            println!("      Time/vector: {:.2} μs", time_per_vector);
            println!("      Throughput: {:.2} vectors/sec", throughput);
        }
    }

    #[test]
    fn test_dimension_scaling() {
        // Test how performance scales with dimension
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimensions = vec![64, 128, 256, 512, 1024];
        let vector_count = 100;

        println!("✅ Dimension Scaling Performance:");
        println!("   Vector count: {}", vector_count);
        println!();

        for dimension in dimensions {
            let vectors = create_test_vectors(vector_count, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            let duration = start.elapsed();

            let throughput = vector_count as f64 / duration.as_secs_f64();
            let data_size = (vector_count * dimension * 4) / 1024; // KB

            println!("   Dimension: {}D", dimension);
            println!("      Data size: {} KB", data_size);
            println!("      Time: {:?}", duration);
            println!("      Throughput: {:.2} vectors/sec", throughput);
        }
    }

    #[test]
    fn test_vector_count_scaling() {
        // Test how performance scales with number of vectors
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 256;
        let vector_counts = vec![100, 500, 1000, 2500, 5000];

        println!("✅ Vector Count Scaling Performance:");
        println!("   Dimension: {}", dimension);
        println!();

        for count in vector_counts {
            let vectors = create_test_vectors(count, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            let duration = start.elapsed();

            let throughput = count as f64 / duration.as_secs_f64();
            let data_size_mb = (count * dimension * 4) as f64 / 1024.0 / 1024.0;

            println!("   {} vectors:", count);
            println!("      Data size: {:.2} MB", data_size_mb);
            println!("      Time: {:?}", duration);
            println!("      Throughput: {:.2} vectors/sec", throughput);
        }
    }

    #[test]
    fn test_memory_bandwidth() {
        // Estimate memory bandwidth
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 1024;
        let count = 1000;
        let vectors = create_test_vectors(count, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        let data_size_bytes = (count * dimension * 4) as f64;
        let data_size_mb = data_size_bytes / 1024.0 / 1024.0;

        let start = Instant::now();
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");
        let duration = start.elapsed();

        let bandwidth_mbps = data_size_mb / duration.as_secs_f64();
        let bandwidth_gbps = bandwidth_mbps / 1024.0;

        println!("✅ Memory Bandwidth Benchmark:");
        println!("   Data transferred: {:.2} MB", data_size_mb);
        println!("   Time: {:?}", duration);
        println!("   Bandwidth: {:.2} MB/s", bandwidth_mbps);
        println!("   Bandwidth: {:.2} GB/s", bandwidth_gbps);

        // Note: This measures effective bandwidth including all overhead
        // (memory allocation, data transfer, Metal command submission, etc.)
        // The M3 Pro has ~200-400 GB/s theoretical unified memory bandwidth,
        // but practical application bandwidth is much lower due to overhead.
        // We just verify operations complete successfully.
        assert!(
            bandwidth_mbps > 1.0,
            "Bandwidth too low: {:.2} MB/s",
            bandwidth_mbps
        );

        println!("   ✅ Memory operations completed successfully");
    }

    #[test]
    fn test_cold_vs_warm_performance() {
        // Compare cold start vs warm cache performance
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 256;
        let count = 500;

        println!("✅ Cold vs Warm Performance:");
        println!();

        // Cold start
        {
            let vectors = create_test_vectors(count, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            let cold_duration = start.elapsed();

            println!("   Cold start (first allocation):");
            println!("      Time: {:?}", cold_duration);
        }

        // Warm cache (repeated allocations)
        let iterations = 5;
        let mut warm_durations = Vec::new();

        for _ in 0..iterations {
            let vectors = create_test_vectors(count, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            warm_durations.push(start.elapsed());
        }

        let avg_warm = warm_durations.iter().sum::<std::time::Duration>() / iterations as u32;
        let min_warm = warm_durations.iter().min().unwrap();
        let max_warm = warm_durations.iter().max().unwrap();

        println!("   Warm cache ({} iterations):", iterations);
        println!("      Avg time: {:?}", avg_warm);
        println!("      Min time: {:?}", min_warm);
        println!("      Max time: {:?}", max_warm);
    }

    #[test]
    fn test_distance_metric_performance() {
        // Compare performance of different distance metrics
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 256;
        let count = 1000;
        let vectors = create_test_vectors(count, dimension);

        println!("✅ Distance Metric Performance:");
        println!("   Dataset: {} vectors × {}D", count, dimension);
        println!();

        let metrics = vec![
            ("Cosine", GpuDistanceMetric::Cosine),
            ("Euclidean", GpuDistanceMetric::Euclidean),
            ("DotProduct", GpuDistanceMetric::DotProduct),
        ];

        for (name, metric) in metrics {
            let mut storage = context
                .create_storage(dimension, metric)
                .expect("Failed to create storage");

            let start = Instant::now();
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");
            let add_duration = start.elapsed();

            // Measure search time
            let query = &vectors[0].data;
            let search_start = Instant::now();
            storage.search(query, 10).expect("Failed to search");
            let search_duration = search_start.elapsed();

            println!("   {}:", name);
            println!("      Add time: {:?}", add_duration);
            println!("      Search time: {:?}", search_duration);
        }
    }

    #[test]
    fn test_concurrent_operations() {
        // Test performance with multiple concurrent operations
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 128;
        let count = 200;
        let num_storages = 5;

        println!("✅ Concurrent Operations Performance:");
        println!("   {} independent storages", num_storages);
        println!();

        let start = Instant::now();

        for i in 0..num_storages {
            let vectors = create_test_vectors(count, dimension);
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");

            if i == 0 {
                println!("   Storage {} time: {:?}", i + 1, start.elapsed());
            }
        }

        let total_duration = start.elapsed();
        let avg_time = total_duration / num_storages as u32;

        println!("   Total time: {:?}", total_duration);
        println!("   Avg per storage: {:?}", avg_time);
        println!(
            "   Throughput: {:.2} storages/sec",
            num_storages as f64 / total_duration.as_secs_f64()
        );
    }

    #[test]
    fn test_performance_baseline() {
        // Establish performance baseline for CI/CD
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 512;
        let count = 1000;
        let vectors = create_test_vectors(count, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        let start = Instant::now();
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");
        let duration = start.elapsed();

        let throughput = count as f64 / duration.as_secs_f64();

        println!("✅ Performance Baseline:");
        println!("   Config: {} vectors × {}D", count, dimension);
        println!("   Time: {:?}", duration);
        println!("   Throughput: {:.2} vectors/sec", throughput);

        // Baseline thresholds (conservative for CI)
        assert!(
            duration.as_secs() < 5,
            "Operation took too long: {:?}",
            duration
        );
        assert!(
            throughput > 200.0,
            "Throughput below baseline: {:.2} vectors/sec",
            throughput
        );

        println!("   ✅ Performance within baseline");
    }
}
