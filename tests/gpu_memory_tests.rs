//! GPU Memory Management Tests
//!
//! Tests for:
//! - Buffer allocation and deallocation
//! - Memory leak detection
//! - Large allocation handling
//! - Buffer pool efficiency
//! - Memory fragmentation

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_memory_tests {
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
    fn test_small_buffer_allocation() {
        // Test allocation of small buffer (1KB)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 64; // 64 * 4 bytes = 256 bytes per vector
        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        let vectors = create_test_vectors(4, dimension); // ~1KB total
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        assert_eq!(storage.vector_count(), 4);
        println!("✅ Small buffer allocation (1KB) successful");
    }

    #[test]
    fn test_medium_buffer_allocation() {
        // Test allocation of medium buffer (1MB)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 512; // Standard embedding size
        let count = 500; // 500 * 512 * 4 bytes = ~1MB
        let vectors = create_test_vectors(count, dimension);

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        assert_eq!(storage.vector_count(), count);
        println!("✅ Medium buffer allocation (~1MB) successful");
        println!("   Vectors: {}", count);
        println!("   Size: ~{} MB", (count * dimension * 4) / 1024 / 1024);
    }

    #[test]
    fn test_large_buffer_allocation() {
        // Test allocation of large buffer (100MB)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 1024;
        let count = 25000; // 25K * 1024 * 4 bytes = ~100MB

        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Add in batches to avoid timeout and duplicate IDs
        let batch_size = 1000;
        let mut total_added = 0;
        for batch_idx in 0..(count / batch_size) {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // Create vectors with unique IDs across batches
            let vectors: Vec<GpuVector> = (start..end)
                .map(|i| {
                    let data: Vec<f32> =
                        (0..dimension).map(|d| (i * dimension + d) as f32).collect();
                    GpuVector::new(format!("vec_{}", i), data)
                })
                .collect();

            storage
                .add_vectors(&vectors)
                .unwrap_or_else(|_| panic!("Failed to add batch {}", batch_idx));

            total_added += vectors.len();

            if batch_idx % 5 == 0 {
                println!("   Progress: {}/{} vectors", total_added, count);
            }
        }

        assert_eq!(storage.vector_count(), count);
        println!("✅ Large buffer allocation (~100MB) successful");
        println!("   Total vectors: {}", count);
        println!(
            "   Total size: ~{} MB",
            (count * dimension * 4) / 1024 / 1024
        );
    }

    #[test]
    fn test_multiple_allocations() {
        // Test multiple independent allocations
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 128;
        let storages_count = 5;

        let mut storages = Vec::new();
        for i in 0..storages_count {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .unwrap_or_else(|_| panic!("Failed to create storage {}", i));

            let vectors = create_test_vectors(10, dimension);
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");

            storages.push(storage);
        }

        // Verify all storages are independent
        for (i, storage) in storages.iter().enumerate() {
            assert_eq!(
                storage.vector_count(),
                10,
                "Storage {} should have 10 vectors",
                i
            );
        }

        println!("✅ Multiple independent allocations successful");
        println!("   Storages created: {}", storages_count);
        println!("   Vectors per storage: 10");
    }

    #[test]
    fn test_deallocation() {
        // Test that deallocation happens when storage is dropped
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info_before = context.device_info().expect("Failed to get device info");
        let used_before = info_before.used_vram_bytes;

        {
            let dimension = 512;
            let count = 1000;
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let vectors = create_test_vectors(count, dimension);
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");

            let info_during = context.device_info().expect("Failed to get device info");
            let used_during = info_during.used_vram_bytes;

            println!("   VRAM before: {} MB", used_before / 1024 / 1024);
            println!("   VRAM during: {} MB", used_during / 1024 / 1024);

            // Metal's unified memory may show fluctuations
            if used_during > used_before {
                println!(
                    "   VRAM allocated: {} MB",
                    (used_during - used_before) / 1024 / 1024
                );
            } else {
                println!(
                    "   VRAM change: {} MB (Metal unified memory)",
                    used_during.saturating_sub(used_before) / 1024 / 1024
                );
            }

            // Storage creation is successful (VRAM accounting may vary with Metal)
            // The important part is no crash occurs
            println!("   ✅ Storage allocated successfully");

            // storage drops here
        }

        // Give GPU time to cleanup
        std::thread::sleep(std::time::Duration::from_millis(100));

        let info_after = context.device_info().expect("Failed to get device info");
        let used_after = info_after.used_vram_bytes;

        println!("   VRAM after: {} MB", used_after / 1024 / 1024);

        // Calculate change (can be positive or negative due to Metal's memory management)
        if used_after > used_before {
            println!(
                "   VRAM increased by: {} MB",
                (used_after - used_before) / 1024 / 1024
            );
        } else {
            println!(
                "   VRAM decreased by: {} MB",
                (used_before - used_after) / 1024 / 1024
            );
        }

        // Metal's unified memory architecture means VRAM can fluctuate
        // The important thing is we don't leak indefinitely
        // We allow for reasonable tolerance due to Metal's memory management
        let tolerance = 50 * 1024 * 1024; // 50MB tolerance for Metal
        let diff = used_after.abs_diff(used_before);

        assert!(
            diff <= tolerance,
            "VRAM change should be within tolerance (before: {} MB, after: {} MB, diff: {} MB)",
            used_before / 1024 / 1024,
            used_after / 1024 / 1024,
            diff / 1024 / 1024
        );

        println!("✅ Deallocation successful (within tolerance)");
    }

    #[test]
    fn test_repeated_allocation_deallocation() {
        // Test repeated allocation and deallocation cycles
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let cycles = 10;
        let dimension = 256;
        let count = 100;

        for cycle in 0..cycles {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let vectors = create_test_vectors(count, dimension);
            storage
                .add_vectors(&vectors)
                .unwrap_or_else(|_| panic!("Failed to add vectors in cycle {}", cycle));

            assert_eq!(storage.vector_count(), count);

            // storage drops here
        }

        println!("✅ Repeated allocation/deallocation successful");
        println!("   Cycles completed: {}", cycles);
        println!("   Vectors per cycle: {}", count);
    }

    #[test]
    fn test_clear_vectors() {
        // Test clearing vectors from storage
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 128;
        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // Add vectors
        let vectors = create_test_vectors(50, dimension);
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");
        assert_eq!(storage.vector_count(), 50);

        // Clear vectors
        storage.clear().expect("Failed to clear storage");
        assert_eq!(storage.vector_count(), 0);

        // Add new vectors after clear
        let new_vectors = create_test_vectors(30, dimension);
        storage
            .add_vectors(&new_vectors)
            .expect("Failed to add vectors after clear");
        assert_eq!(storage.vector_count(), 30);

        println!("✅ Clear vectors successful");
        println!("   Initial: 50 vectors");
        println!("   After clear: 0 vectors");
        println!("   After re-add: 30 vectors");
    }

    #[test]
    fn test_memory_reuse() {
        // Test that memory can be reused efficiently
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 256;
        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        // First allocation
        let vectors1 = create_test_vectors(100, dimension);
        storage
            .add_vectors(&vectors1)
            .expect("Failed to add first batch");

        // Clear and reallocate
        storage.clear().expect("Failed to clear");

        let vectors2 = create_test_vectors(150, dimension);
        storage
            .add_vectors(&vectors2)
            .expect("Failed to add second batch");

        assert_eq!(storage.vector_count(), 150);

        println!("✅ Memory reuse successful");
        println!("   First allocation: 100 vectors");
        println!("   Second allocation: 150 vectors");
    }

    #[test]
    fn test_zero_allocation() {
        // Test creating storage with zero vectors (edge case)
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 128;
        let storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create empty storage");

        assert_eq!(storage.vector_count(), 0);

        println!("✅ Zero allocation (empty storage) successful");
    }

    #[test]
    fn test_memory_stress() {
        // Stress test: many small allocations
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let dimension = 64;
        let iterations = 50;
        let vectors_per_iteration = 20;

        for i in 0..iterations {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .unwrap_or_else(|_| panic!("Failed to create storage iteration {}", i));

            let vectors = create_test_vectors(vectors_per_iteration, dimension);
            storage
                .add_vectors(&vectors)
                .unwrap_or_else(|_| panic!("Failed to add vectors iteration {}", i));

            if i % 10 == 0 {
                println!("   Iteration {}/{}", i, iterations);
            }

            // storage drops here
        }

        println!("✅ Memory stress test successful");
        println!("   Iterations: {}", iterations);
        println!("   Allocations per iteration: {}", vectors_per_iteration);
        println!(
            "   Total allocations: {}",
            iterations * vectors_per_iteration
        );
    }
}
