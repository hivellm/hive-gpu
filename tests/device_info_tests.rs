//! Device info API tests
//!
//! Tests for the GPU device information API

use hive_gpu::error::HiveGpuError;

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_tests {
    use super::*;
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::traits::GpuContext;

    #[test]
    fn test_metal_device_info() {
        // Skip if Metal not available
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        // Verify all fields are populated
        assert!(!info.name.is_empty(), "Device name should not be empty");
        assert_eq!(info.backend, "Metal", "Backend should be 'Metal'");

        // VRAM checks
        assert!(info.total_vram_bytes > 0, "Total VRAM should be positive");
        assert!(
            info.available_vram_bytes <= info.total_vram_bytes,
            "Available VRAM should not exceed total VRAM"
        );
        assert!(
            info.used_vram_bytes == info.total_vram_bytes - info.available_vram_bytes,
            "Used VRAM should equal total - available"
        );

        // Driver version check
        assert!(
            info.driver_version.contains("macOS"),
            "Driver version should contain 'macOS'"
        );

        // Capabilities check
        assert!(
            info.max_threads_per_block > 0,
            "Max threads per block should be positive"
        );
        assert!(
            info.max_shared_memory_per_block > 0,
            "Max shared memory should be positive"
        );

        // Metal-specific checks
        assert_eq!(info.device_id, 0, "Metal should report device_id as 0");
        assert!(
            info.pci_bus_id.is_none(),
            "Metal should not expose PCI bus ID"
        );
        assert!(
            info.compute_capability.is_none(),
            "Metal should not expose compute capability"
        );

        println!("✅ Metal Device Info:");
        println!("   Name: {}", info.name);
        println!("   Backend: {}", info.backend);
        println!("   Total VRAM: {} MB", info.total_vram_mb());
        println!("   Available VRAM: {} MB", info.available_vram_mb());
        println!("   Used VRAM: {} MB", info.used_vram_bytes / (1024 * 1024));
        println!("   Usage: {:.1}%", info.vram_usage_percent());
        println!("   Driver: {}", info.driver_version);
        println!("   Max threads/block: {}", info.max_threads_per_block);
        println!(
            "   Max shared memory: {} KB",
            info.max_shared_memory_per_block / 1024
        );
    }

    #[test]
    fn test_vram_usage_percent() {
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        let usage = info.vram_usage_percent();
        assert!(
            (0.0..=100.0).contains(&usage),
            "VRAM usage percentage should be between 0 and 100, got {}",
            usage
        );

        println!("✅ VRAM usage: {:.1}%", usage);
    }

    #[test]
    fn test_has_available_vram() {
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        // Test with 1GB requirement
        let one_gb = 1024 * 1024 * 1024;
        let has_1gb = info.has_available_vram(one_gb);

        println!("✅ Has 1GB available: {}", has_1gb);

        // Test with more than total VRAM (should be false)
        let huge_amount = info.total_vram_bytes + 1;
        assert!(
            !info.has_available_vram(huge_amount),
            "Should not have more VRAM than total"
        );

        // Test with 0 bytes (should always be true)
        assert!(
            info.has_available_vram(0),
            "Should always have 0 bytes available"
        );
    }

    #[test]
    fn test_vram_convenience_methods() {
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        let total_mb = info.total_vram_mb();
        let available_mb = info.available_vram_mb();

        assert!(total_mb > 0, "Total VRAM in MB should be positive");
        assert!(
            available_mb <= total_mb,
            "Available VRAM in MB should not exceed total"
        );

        // Verify conversion is correct
        assert_eq!(
            total_mb,
            info.total_vram_bytes / (1024 * 1024),
            "MB conversion should be correct"
        );
        assert_eq!(
            available_mb,
            info.available_vram_bytes / (1024 * 1024),
            "MB conversion should be correct"
        );

        println!("✅ Total VRAM: {} MB", total_mb);
        println!("✅ Available VRAM: {} MB", available_mb);
    }
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use hive_gpu::cuda::CudaContext;
    use hive_gpu::traits::GpuContext;

    #[test]
    fn test_cuda_device_info() {
        // Skip if CUDA not available
        if !CudaContext::is_available() {
            println!("⚠️  CUDA not available, skipping test");
            return;
        }

        let context = CudaContext::new().expect("Failed to create CUDA context");
        let info = context.device_info().expect("Failed to get device info");

        // Verify all fields are populated
        assert!(!info.name.is_empty(), "Device name should not be empty");
        assert_eq!(info.backend, "CUDA", "Backend should be 'CUDA'");

        // VRAM checks
        assert!(info.total_vram_bytes > 0, "Total VRAM should be positive");
        assert!(
            info.available_vram_bytes <= info.total_vram_bytes,
            "Available VRAM should not exceed total VRAM"
        );

        // Driver version check
        assert!(
            !info.driver_version.is_empty(),
            "Driver version should not be empty"
        );

        // CUDA-specific checks
        assert!(
            info.compute_capability.is_some(),
            "CUDA should expose compute capability"
        );
        assert!(info.pci_bus_id.is_some(), "CUDA should expose PCI bus ID");

        println!("✅ CUDA Device Info:");
        println!("   Name: {}", info.name);
        println!("   Compute Capability: {:?}", info.compute_capability);
        println!("   Total VRAM: {} MB", info.total_vram_mb());
        println!("   PCI Bus ID: {:?}", info.pci_bus_id);
    }
}

// ROCm tests will be added when ROCm backend is implemented
