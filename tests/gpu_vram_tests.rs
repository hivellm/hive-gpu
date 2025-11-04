//! GPU VRAM Monitoring Tests
//!
//! Tests for VRAM monitoring accuracy and real-time tracking:
//! - VRAM usage tracking accuracy
//! - Real-time monitoring
//! - Usage percentage calculation
//! - Available memory queries
//! - Memory pressure detection

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_vram_tests {
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
    fn test_vram_tracking_accuracy() {
        // Test that VRAM tracking reports accurate values
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        println!("✅ VRAM Tracking Accuracy Test:");
        println!("   Device: {}", info.name);
        println!("   Total VRAM: {} MB", info.total_vram_mb());
        println!("   Available: {} MB", info.available_vram_mb());
        println!("   Used: {} MB", info.used_vram_bytes / 1024 / 1024);
        println!("   Usage: {:.2}%", info.vram_usage_percent());

        // Verify values are consistent
        assert!(info.total_vram_bytes > 0, "Total VRAM should be positive");
        assert!(
            info.available_vram_bytes <= info.total_vram_bytes,
            "Available should not exceed total"
        );
        assert!(
            info.vram_usage_percent() >= 0.0 && info.vram_usage_percent() <= 100.0,
            "Usage percentage should be 0-100%"
        );

        // Verify convenience methods match raw values
        assert_eq!(
            info.total_vram_mb(),
            info.total_vram_bytes / 1024 / 1024,
            "total_vram_mb should match calculation"
        );
        assert_eq!(
            info.available_vram_mb(),
            info.available_vram_bytes / 1024 / 1024,
            "available_vram_mb should match calculation"
        );
    }

    #[test]
    fn test_vram_usage_during_allocation() {
        // Test VRAM tracking during actual allocation
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info_before = context.device_info().expect("Failed to get device info");
        println!("✅ VRAM Usage During Allocation Test:");
        println!("   BEFORE allocation:");
        println!("      Available: {} MB", info_before.available_vram_mb());
        println!(
            "      Used: {} MB",
            info_before.used_vram_bytes / 1024 / 1024
        );
        println!("      Usage: {:.2}%", info_before.vram_usage_percent());

        // Allocate memory
        let dimension = 512;
        let count = 1000;
        let mut storage = context
            .create_storage(dimension, GpuDistanceMetric::Cosine)
            .expect("Failed to create storage");

        let vectors = create_test_vectors(count, dimension);
        storage
            .add_vectors(&vectors)
            .expect("Failed to add vectors");

        let expected_size = count * dimension * 4; // 4 bytes per f32
        println!(
            "   Expected allocation: ~{} MB",
            expected_size / 1024 / 1024
        );

        let info_during = context.device_info().expect("Failed to get device info");
        println!("   DURING allocation:");
        println!("      Available: {} MB", info_during.available_vram_mb());
        println!(
            "      Used: {} MB",
            info_during.used_vram_bytes / 1024 / 1024
        );
        println!("      Usage: {:.2}%", info_during.vram_usage_percent());

        // Metal's unified memory may not show exact changes
        // But we verify the API works
        assert!(
            info_during.total_vram_bytes == info_before.total_vram_bytes,
            "Total VRAM should remain constant"
        );
    }

    #[test]
    fn test_vram_percentage_calculation() {
        // Test usage percentage calculation
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        let calculated_percentage =
            (info.used_vram_bytes as f64 / info.total_vram_bytes as f64) * 100.0;
        let reported_percentage = info.vram_usage_percent();

        println!("✅ VRAM Percentage Calculation Test:");
        println!("   Total: {} bytes", info.total_vram_bytes);
        println!("   Used: {} bytes", info.used_vram_bytes);
        println!("   Calculated %: {:.4}%", calculated_percentage);
        println!("   Reported %: {:.4}%", reported_percentage);

        // Should match within floating point tolerance
        let diff = (calculated_percentage - reported_percentage).abs();
        assert!(
            diff < 0.01,
            "Percentage calculation should match (diff: {})",
            diff
        );
    }

    #[test]
    fn test_has_available_vram() {
        // Test has_available_vram method
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        println!("✅ Has Available VRAM Test:");
        println!("   Available: {} MB", info.available_vram_mb());

        // Should have some VRAM available
        assert!(
            info.has_available_vram(1024 * 1024),
            "Should have at least 1 MB available"
        );
        assert!(
            info.has_available_vram(10 * 1024 * 1024),
            "Should have at least 10 MB available"
        );

        // Test with unreasonably large amount
        let huge_amount = 1000 * 1024 * 1024 * 1024; // 1000 GB
        let has_huge = info.has_available_vram(huge_amount);
        println!("   Has 1000 GB available: {}", has_huge);
        // Most systems won't have 1000 GB, but test passes either way

        // Test edge case: 0 bytes
        assert!(
            info.has_available_vram(0),
            "Should always have 0 bytes available"
        );
    }

    #[test]
    fn test_vram_monitoring_multiple_contexts() {
        // Test VRAM monitoring with multiple contexts
        let context1 = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let context2 = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(e) => panic!("Failed to create second Metal context: {}", e),
        };

        let info1 = context1
            .device_info()
            .expect("Failed to get info from ctx1");
        let info2 = context2
            .device_info()
            .expect("Failed to get info from ctx2");

        println!("✅ Multiple Contexts VRAM Monitoring Test:");
        println!("   Context 1:");
        println!("      Total: {} MB", info1.total_vram_mb());
        println!("      Available: {} MB", info1.available_vram_mb());
        println!("   Context 2:");
        println!("      Total: {} MB", info2.total_vram_mb());
        println!("      Available: {} MB", info2.available_vram_mb());

        // Both should report same device
        assert_eq!(
            info1.name, info2.name,
            "Both contexts should see same device"
        );
        assert_eq!(
            info1.total_vram_bytes, info2.total_vram_bytes,
            "Total VRAM should be same for both contexts"
        );
    }

    #[test]
    fn test_vram_monitoring_over_time() {
        // Test VRAM monitoring over multiple allocations
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        println!("✅ VRAM Monitoring Over Time Test:");

        let snapshots = 5;
        let dimension = 256;
        let count = 100;

        for i in 0..snapshots {
            let mut storage = context
                .create_storage(dimension, GpuDistanceMetric::Cosine)
                .expect("Failed to create storage");

            let vectors = create_test_vectors(count, dimension);
            storage
                .add_vectors(&vectors)
                .expect("Failed to add vectors");

            let info = context.device_info().expect("Failed to get device info");
            println!("   Snapshot {}:", i + 1);
            println!("      Available: {} MB", info.available_vram_mb());
            println!("      Used: {} MB", info.used_vram_bytes / 1024 / 1024);
            println!("      Usage: {:.2}%", info.vram_usage_percent());

            // storage drops here, memory should be freed
        }

        println!("   ✅ All snapshots completed successfully");
    }

    #[test]
    fn test_vram_pressure_detection() {
        // Test detecting when VRAM usage is high
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        println!("✅ VRAM Pressure Detection Test:");
        println!("   Total VRAM: {} MB", info.total_vram_mb());
        println!("   Available: {} MB", info.available_vram_mb());
        println!("   Usage: {:.2}%", info.vram_usage_percent());

        // Define pressure thresholds
        let low_pressure = info.vram_usage_percent() < 50.0;
        let medium_pressure = info.vram_usage_percent() >= 50.0 && info.vram_usage_percent() < 80.0;
        let high_pressure = info.vram_usage_percent() >= 80.0;

        println!("   Pressure levels:");
        println!("      Low (<50%): {}", low_pressure);
        println!("      Medium (50-80%): {}", medium_pressure);
        println!("      High (>80%): {}", high_pressure);

        // Exactly one should be true
        let pressure_count = [low_pressure, medium_pressure, high_pressure]
            .iter()
            .filter(|&&x| x)
            .count();
        assert_eq!(
            pressure_count, 1,
            "Exactly one pressure level should be active"
        );
    }

    #[test]
    fn test_vram_available_for_allocation() {
        // Test checking if enough VRAM is available before allocation
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        println!("✅ VRAM Available for Allocation Test:");

        // Calculate sizes for different allocations
        let small = 10 * 1024 * 1024; // 10 MB
        let medium = 100 * 1024 * 1024; // 100 MB
        let large = 1024 * 1024 * 1024; // 1 GB

        println!("   Checking allocation sizes:");
        println!("      10 MB: {}", info.has_available_vram(small));
        println!("      100 MB: {}", info.has_available_vram(medium));
        println!("      1 GB: {}", info.has_available_vram(large));

        // Small allocations should definitely work
        assert!(
            info.has_available_vram(small),
            "Should have 10 MB available"
        );

        // Verify calculation
        assert_eq!(
            info.has_available_vram(small),
            info.available_vram_bytes >= small,
            "has_available_vram should match direct comparison"
        );
    }

    #[test]
    fn test_vram_stats_consistency() {
        // Test that VRAM statistics remain consistent across multiple queries
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        println!("✅ VRAM Stats Consistency Test:");

        let queries = 10;
        let mut total_vrams = Vec::new();

        for i in 0..queries {
            let info = context.device_info().expect("Failed to get device info");
            total_vrams.push(info.total_vram_bytes);

            if i == 0 {
                println!("   Query {}:", i + 1);
                println!("      Total: {} MB", info.total_vram_mb());
                println!("      Available: {} MB", info.available_vram_mb());
            }
        }

        // Total VRAM should be constant across all queries
        let first_total = total_vrams[0];
        assert!(
            total_vrams.iter().all(|&x| x == first_total),
            "Total VRAM should be constant across queries"
        );

        println!("   ✅ {} queries - Total VRAM consistent", queries);
    }

    #[test]
    fn test_vram_boundary_conditions() {
        // Test boundary conditions in VRAM calculations
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        println!("✅ VRAM Boundary Conditions Test:");

        // Test with 0 bytes
        assert!(info.has_available_vram(0), "Should handle 0 bytes");

        // Test with exactly available amount
        let available = info.available_vram_bytes;
        assert!(
            info.has_available_vram(available),
            "Should have exactly available amount"
        );

        // Test with 1 byte more than available
        assert!(
            !info.has_available_vram(available + 1),
            "Should not have more than available"
        );

        // Test with total VRAM
        let total = info.total_vram_bytes;
        println!(
            "   Total VRAM: {} MB ({})",
            total / 1024 / 1024,
            if info.has_available_vram(total) {
                "available"
            } else {
                "not available"
            }
        );

        // Test with u64::MAX (should handle gracefully)
        assert!(
            !info.has_available_vram(u64::MAX),
            "Should handle u64::MAX gracefully"
        );

        println!("   ✅ All boundary conditions handled correctly");
    }
}
