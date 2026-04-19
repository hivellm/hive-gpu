//! GPU Hardware Detection Tests
//!
//! Comprehensive tests for GPU hardware detection across all backends:
//! - Metal (macOS)
//! - CUDA (Linux/Windows)
//! - ROCm (Linux)
//! - CPU fallback

#[cfg(all(target_os = "macos", feature = "metal-native"))]
mod metal_detection_tests {
    use hive_gpu::error::HiveGpuError;
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::traits::GpuContext;

    #[test]
    fn test_metal_device_availability() {
        // Test that Metal backend can check device availability
        let result = MetalNativeContext::new();

        match result {
            Ok(context) => {
                println!("✅ Metal device available");
                println!("   Context created successfully");

                // Verify context is functional
                let info = context.device_info().expect("Failed to get device info");
                assert!(!info.name.is_empty(), "Device name should not be empty");
                assert_eq!(info.backend, "Metal", "Backend should be Metal");
            }
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal device not available on this system");
                // This is acceptable - not all systems have Metal
            }
            Err(e) => {
                panic!("Unexpected error during Metal detection: {}", e);
            }
        }
    }

    #[test]
    fn test_metal_device_name_retrieval() {
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        // Verify device name
        assert!(!info.name.is_empty(), "Device name should not be empty");
        println!("✅ Metal device name: {}", info.name);

        // Common Apple Silicon names
        let valid_prefixes = ["Apple", "AMD", "Intel"];
        let has_valid_prefix = valid_prefixes
            .iter()
            .any(|prefix| info.name.starts_with(prefix));
        assert!(
            has_valid_prefix,
            "Device name should start with known vendor: {}",
            info.name
        );
    }

    #[test]
    fn test_metal_device_capabilities() {
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        // Verify capabilities are populated
        assert!(
            info.max_threads_per_block > 0,
            "Max threads should be positive"
        );
        assert!(
            info.max_shared_memory_per_block > 0,
            "Max shared memory should be positive"
        );
        assert!(info.total_vram_bytes > 0, "Total VRAM should be positive");

        println!("✅ Metal capabilities:");
        println!("   Max threads/block: {}", info.max_threads_per_block);
        println!(
            "   Max shared memory: {} KB",
            info.max_shared_memory_per_block / 1024
        );
        println!("   Total VRAM: {} MB", info.total_vram_mb());

        // Sanity checks for Apple Silicon
        // Apple Silicon typically has:
        // - Max threads: 512-1024
        // - Shared memory: 32KB
        // - VRAM: 8GB-128GB (unified memory)
        if info.name.contains("Apple") {
            assert!(
                info.max_threads_per_block >= 512,
                "Apple Silicon should support at least 512 threads per block"
            );
            assert!(
                info.max_shared_memory_per_block >= 16 * 1024,
                "Apple Silicon should have at least 16KB shared memory"
            );
        }
    }

    #[test]
    fn test_metal_multiple_contexts() {
        // Test creating multiple Metal contexts (should work)
        let context1 = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create first Metal context: {}", e),
        };

        let context2 = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(e) => panic!("Failed to create second Metal context: {}", e),
        };

        // Both contexts should be functional
        let info1 = context1
            .device_info()
            .expect("Failed to get info from context 1");
        let info2 = context2
            .device_info()
            .expect("Failed to get info from context 2");

        // Should refer to the same device
        assert_eq!(
            info1.name, info2.name,
            "Both contexts should use same device"
        );
        assert_eq!(
            info1.backend, info2.backend,
            "Both should use Metal backend"
        );

        println!("✅ Multiple Metal contexts created successfully");
        println!("   Context 1: {}", info1.name);
        println!("   Context 2: {}", info2.name);
    }

    #[test]
    fn test_metal_vram_query() {
        let context = match MetalNativeContext::new() {
            Ok(ctx) => ctx,
            Err(HiveGpuError::NoDeviceAvailable) => {
                println!("⚠️  Metal not available, skipping test");
                return;
            }
            Err(e) => panic!("Failed to create Metal context: {}", e),
        };

        let info = context.device_info().expect("Failed to get device info");

        // Verify VRAM information is consistent
        assert!(info.total_vram_bytes > 0, "Total VRAM should be positive");
        assert!(
            info.available_vram_bytes <= info.total_vram_bytes,
            "Available VRAM should not exceed total"
        );
        assert!(
            info.used_vram_bytes == info.total_vram_bytes - info.available_vram_bytes,
            "Used VRAM should equal total - available"
        );

        println!("✅ VRAM information:");
        println!("   Total: {} MB", info.total_vram_mb());
        println!("   Available: {} MB", info.available_vram_mb());
        println!("   Used: {} MB", info.used_vram_bytes / (1024 * 1024));
        println!("   Usage: {:.1}%", info.vram_usage_percent());
    }
}

/*
// CUDA tests - will be uncommented when CUDA backend is implemented

#[cfg(feature = "cuda")]
mod cuda_detection_tests {
    use hive_gpu::cuda::CudaContext;
    use hive_gpu::traits::GpuContext;

    #[test]
    fn test_cuda_device_availability() {
        // Test CUDA backend availability check
        let is_available = CudaContext::is_available();

        if is_available {
            println!("✅ CUDA is available on this system");

            // Try to create context
            let result = CudaContext::new();
            assert!(
                result.is_ok(),
                "CUDA context creation should succeed when available"
            );

            let context = result.unwrap();
            let info = context.device_info().expect("Failed to get device info");
            assert_eq!(info.backend, "CUDA", "Backend should be CUDA");
            println!("   CUDA device: {}", info.name);
        } else {
            println!("⚠️  CUDA not available on this system");

            // Verify that context creation fails gracefully
            let result = CudaContext::new();
            assert!(
                result.is_err(),
                "CUDA context should fail when not available"
            );
        }
    }

    #[test]
    fn test_cuda_device_enumeration() {
        if !CudaContext::is_available() {
            println!("⚠️  CUDA not available, skipping test");
            return;
        }

        // Test enumerating CUDA devices
        let device_count = CudaContext::device_count().expect("Failed to get device count");
        assert!(device_count > 0, "Should have at least one CUDA device");

        println!("✅ CUDA devices found: {}", device_count);

        // Get info for each device
        for i in 0..device_count {
            let context = CudaContext::with_device(i).expect("Failed to create context for device");
            let info = context.device_info().expect("Failed to get device info");

            println!("   Device {}: {}", i, info.name);
            println!("      Compute Capability: {:?}", info.compute_capability);
            println!("      VRAM: {} MB", info.total_vram_mb());
        }
    }

    #[test]
    fn test_cuda_compute_capability() {
        if !CudaContext::is_available() {
            println!("⚠️  CUDA not available, skipping test");
            return;
        }

        let context = CudaContext::new().expect("Failed to create CUDA context");
        let info = context.device_info().expect("Failed to get device info");

        // CUDA should expose compute capability
        assert!(
            info.compute_capability.is_some(),
            "CUDA should report compute capability"
        );

        let cc = info.compute_capability.unwrap();
        println!("✅ CUDA compute capability: {}", cc);

        // Parse compute capability (format: "X.Y")
        let parts: Vec<&str> = cc.split('.').collect();
        assert_eq!(parts.len(), 2, "Compute capability should have format X.Y");

        let major: u32 = parts[0].parse().expect("Major version should be a number");
        let minor: u32 = parts[1].parse().expect("Minor version should be a number");

        // Modern CUDA devices should have at least compute capability 3.5
        assert!(
            major >= 3,
            "CUDA device should have compute capability >= 3.0"
        );
        println!("   Major: {}, Minor: {}", major, minor);
    }

    #[test]
    fn test_cuda_pci_bus_id() {
        if !CudaContext::is_available() {
            println!("⚠️  CUDA not available, skipping test");
            return;
        }

        let context = CudaContext::new().expect("Failed to create CUDA context");
        let info = context.device_info().expect("Failed to get device info");

        // CUDA should expose PCI bus ID
        assert!(info.pci_bus_id.is_some(), "CUDA should report PCI bus ID");

        let pci = info.pci_bus_id.unwrap();
        println!("✅ CUDA PCI bus ID: {}", pci);

        // PCI bus ID should match format: "0000:XX:YY.Z"
        assert!(
            pci.contains(':') && pci.contains('.'),
            "PCI bus ID should have standard format"
        );
    }
}
*/

/*
// ROCm tests - will be uncommented when ROCm backend is implemented

#[cfg(feature = "rocm")]
mod rocm_detection_tests {
    use hive_gpu::rocm::RocmContext;
    use hive_gpu::traits::GpuContext;

    #[test]
    fn test_rocm_device_availability() {
        // Test ROCm backend availability check
        let is_available = RocmContext::is_available();

        if is_available {
            println!("✅ ROCm is available on this system");

            let result = RocmContext::new();
            assert!(
                result.is_ok(),
                "ROCm context creation should succeed when available"
            );

            let context = result.unwrap();
            let info = context.device_info().expect("Failed to get device info");
            assert_eq!(info.backend, "ROCm", "Backend should be ROCm");
            println!("   ROCm device: {}", info.name);
        } else {
            println!("⚠️  ROCm not available on this system");
        }
    }

    #[test]
    fn test_rocm_device_enumeration() {
        if !RocmContext::is_available() {
            println!("⚠️  ROCm not available, skipping test");
            return;
        }

        let device_count = RocmContext::device_count().expect("Failed to get device count");
        assert!(device_count > 0, "Should have at least one ROCm device");

        println!("✅ ROCm devices found: {}", device_count);
    }

    #[test]
    fn test_rocm_architecture() {
        if !RocmContext::is_available() {
            println!("⚠️  ROCm not available, skipping test");
            return;
        }

        let context = RocmContext::new().expect("Failed to create ROCm context");
        let info = context.device_info().expect("Failed to get device info");

        // ROCm should expose architecture (e.g., "gfx1030")
        assert!(
            info.compute_capability.is_some(),
            "ROCm should report architecture"
        );

        let arch = info.compute_capability.unwrap();
        println!("✅ ROCm architecture: {}", arch);
        assert!(
            arch.starts_with("gfx"),
            "ROCm architecture should start with 'gfx'"
        );
    }
}
*/

/// Fallback tests - these run regardless of GPU availability
mod fallback_tests {
    use hive_gpu::backends::detector::{
        GpuBackendType, detect_available_backends, select_best_backend,
    };

    #[test]
    fn test_backend_detection() {
        // Test that backend detection works
        let backends = detect_available_backends();

        println!("✅ Detected backends: {:?}", backends);
        assert!(
            !backends.is_empty(),
            "Should detect at least one backend (CPU)"
        );

        // Backend should include at least CPU
        assert!(
            backends.contains(&GpuBackendType::Cpu),
            "CPU should always be available as fallback"
        );

        // Print all detected backends
        for backend in &backends {
            match backend {
                GpuBackendType::Metal => {
                    println!("   Metal backend available");
                    #[cfg(not(target_os = "macos"))]
                    panic!("Metal should only be detected on macOS");
                }
                GpuBackendType::Cuda => {
                    println!("   CUDA backend available");
                }
                GpuBackendType::Rocm => {
                    println!("   ROCm backend available");
                }
                GpuBackendType::Intel => {
                    println!("   Intel backend available");
                }
                GpuBackendType::Cpu => {
                    println!("   CPU backend available");
                }
            }
        }
    }

    #[test]
    fn test_best_backend_selection() {
        // Test selecting the best available backend
        let best = select_best_backend().expect("Should always find a backend");

        println!("✅ Best backend selected: {:?}", best);

        // Best backend should be one of the detected ones
        let available = detect_available_backends();
        assert!(
            available.contains(&best),
            "Best backend should be in available backends"
        );

        // Priority should be: Metal > CUDA > CPU
        #[cfg(all(target_os = "macos", feature = "metal-native"))]
        {
            if available.contains(&GpuBackendType::Metal) {
                assert_eq!(
                    best,
                    GpuBackendType::Metal,
                    "Metal should be preferred on macOS"
                );
            }
        }
    }

    #[test]
    fn test_graceful_fallback_no_gpu() {
        // This test ensures graceful behavior when no GPU is available

        println!("✅ Testing graceful fallback behavior");

        // Should always be able to select a backend (CPU fallback)
        let backend = select_best_backend().expect("Should fallback to CPU if no GPU");

        // Should always return a valid backend
        println!("   Fallback backend: {:?}", backend);

        // Get backend info
        let info = hive_gpu::backends::detector::get_backend_info(backend);
        if let Ok(info_str) = info {
            println!("   Info: {}", info_str);
        }
    }

    #[test]
    fn test_backend_performance_info() {
        // Test getting performance characteristics for each backend
        let backends = detect_available_backends();

        println!("✅ Backend performance characteristics:");
        for backend in backends {
            let perf = hive_gpu::backends::detector::get_backend_performance_info(backend);
            println!("   {} ({}):", perf.name, backend);
            println!(
                "      Memory bandwidth: {:.1} GB/s",
                perf.memory_bandwidth_gbps
            );
            println!("      Compute units: {}", perf.compute_units);
            println!("      Memory size: {} GB", perf.memory_size_gb);
            println!("      HNSW support: {}", perf.supports_hnsw);
            println!("      Batch support: {}", perf.supports_batch);
        }
    }
}
