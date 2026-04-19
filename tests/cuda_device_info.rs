//! Device info API tests on a real CUDA device.

#![cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]

use hive_gpu::cuda::CudaContext;
use hive_gpu::traits::{GpuBackend, GpuContext};

fn skip_if_no_gpu() -> bool {
    if !CudaContext::is_available() {
        eprintln!("[cuda_device_info] no CUDA device detected; test is a no-op");
        return true;
    }
    false
}

#[test]
fn device_count_is_positive_when_available() {
    if skip_if_no_gpu() {
        return;
    }
    let count = CudaContext::device_count().expect("device_count");
    assert!(count >= 1, "at least one CUDA device must be reported");
}

#[test]
fn is_available_is_idempotent() {
    for _ in 0..3 {
        let _ = CudaContext::is_available();
    }
}

#[test]
fn device_info_fields_are_all_populated() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let info = GpuBackend::device_info(&ctx);

    assert_eq!(info.backend, "CUDA");
    assert!(!info.name.is_empty());
    assert!(info.total_vram_bytes > 0);
    assert!(info.available_vram_bytes <= info.total_vram_bytes);
    assert_eq!(
        info.used_vram_bytes,
        info.total_vram_bytes
            .saturating_sub(info.available_vram_bytes)
    );
    assert!(info.driver_version.starts_with("CUDA"));

    let cc = info
        .compute_capability
        .as_ref()
        .expect("CUDA backend populates compute capability");
    assert!(cc.contains('.'));
    let (major_str, minor_str) = cc.split_once('.').unwrap();
    let major: u32 = major_str.parse().unwrap();
    let minor: u32 = minor_str.parse().unwrap();
    assert!((3..=15).contains(&major), "major {major} out of range");
    assert!(minor < 10, "minor {minor} out of range");

    assert!(info.max_threads_per_block >= 512);
    assert!(info.max_shared_memory_per_block >= 16_384);
    assert!(info.device_id >= 0);
    if let Some(pci) = &info.pci_bus_id {
        assert!(pci.len() == 12, "pci_bus_id format: {pci}");
        assert!(pci.contains(':'));
    }
}

#[test]
fn memory_stats_agree_with_device_info() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let info = GpuBackend::device_info(&ctx);
    let stats = GpuBackend::memory_stats(&ctx);

    assert!(stats.available as u64 <= info.total_vram_bytes);
    assert!(stats.utilization >= 0.0 && stats.utilization <= 1.0);
}

#[test]
fn vram_helpers_return_sensible_ranges() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let info = GpuContext::device_info(&ctx).unwrap();

    let total_mb = info.total_vram_mb();
    assert!(total_mb >= 512, "any modern NVIDIA GPU has >=512 MB VRAM");
    let usage = info.vram_usage_percent();
    assert!((0.0..=100.0).contains(&usage));
    assert!(info.has_available_vram(1024 * 1024));
}
