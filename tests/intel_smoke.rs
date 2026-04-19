//! Smoke tests for the Intel (Vulkan Compute) backend.
//!
//! ⚠️ NOT YET VALIDATED ON REAL HARDWARE — see `phase3c_add-intel-backend`.
//! Tests only compile on Linux or Windows with the `intel` feature
//! enabled and should be run on an Intel Arc / Battlemage host — or any
//! Vulkan-capable GPU in universal-fallback mode — before the task is
//! archived.
//!
//! Tests are a no-op when no Vulkan loader is reachable, so the suite
//! stays green on CI runners without a GPU.

#![cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))]

use hive_gpu::intel::IntelContext;
use hive_gpu::traits::{GpuBackend, GpuContext};
use hive_gpu::types::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;

fn skip_if_no_gpu() -> bool {
    if !IntelContext::is_available() {
        eprintln!("[intel_smoke] no Vulkan-capable device detected; test is a no-op");
        return true;
    }
    false
}

fn mk(id: &str, data: Vec<f32>) -> GpuVector {
    GpuVector {
        id: id.to_string(),
        data,
        metadata: HashMap::new(),
    }
}

#[test]
fn context_creation_reports_real_device_info() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx =
        IntelContext::new_with_preference(true).expect("construct context on any Vulkan device");
    let info = GpuBackend::device_info(&*ctx);
    assert_eq!(info.backend, "Intel");
    assert!(!info.name.is_empty());
    assert!(info.driver_version.starts_with("Vulkan"));
    assert!(info.max_threads_per_block >= 1);
    // total_vram_bytes may be zero on integrated GPUs that share host
    // memory; we just check it is well-defined.
    let _ = info.total_vram_bytes;
}

#[test]
fn batch_add_then_cosine_search_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = IntelContext::new_with_preference(true).unwrap();
    let mut storage = ctx
        .create_storage(8, GpuDistanceMetric::Cosine)
        .expect("create_storage");

    let vectors = vec![
        mk("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mk("b", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mk("c", vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mk("mix", vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    ];
    storage.add_vectors(&vectors).unwrap();
    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let r = storage.search(&query, 4).unwrap();
    assert_eq!(r[0].id, "a");
    assert!((r[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn euclidean_ranks_nearest_first() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = IntelContext::new_with_preference(true).unwrap();
    let mut storage = ctx.create_storage(4, GpuDistanceMetric::Euclidean).unwrap();
    storage
        .add_vectors(&[
            mk("near", vec![1.0, 1.0, 1.0, 1.0]),
            mk("far", vec![9.0, 9.0, 9.0, 9.0]),
            mk("mid", vec![3.0, 3.0, 3.0, 3.0]),
        ])
        .unwrap();
    let r = storage.search(&[1.1, 1.1, 1.1, 1.1], 3).unwrap();
    assert_eq!(r[0].id, "near");
    assert_eq!(r[1].id, "mid");
    assert_eq!(r[2].id, "far");
}

#[test]
fn dotproduct_matches_cpu_reference_on_random_batch() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = IntelContext::new_with_preference(true).unwrap();
    let dim = 32;
    let n = 500;

    let mut rng_state: u32 = 0xC0FFEE;
    let mut rng = || {
        rng_state = rng_state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let vectors: Vec<GpuVector> = (0..n)
        .map(|i| mk(&format!("v{i}"), (0..dim).map(|_| rng()).collect()))
        .collect();
    let cpu_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();

    let mut storage = ctx
        .create_storage(dim, GpuDistanceMetric::DotProduct)
        .unwrap();
    storage.add_vectors(&vectors).unwrap();

    let q: Vec<f32> = (0..dim).map(|_| rng()).collect();
    let gpu = storage.search(&q, 10).unwrap();
    let mut cpu: Vec<(usize, f32)> = cpu_data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    cpu.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    cpu.truncate(10);

    let gpu_ids: Vec<String> = gpu.iter().map(|r| r.id.clone()).collect();
    let cpu_ids: Vec<String> = cpu.iter().map(|(i, _)| format!("v{i}")).collect();
    assert_eq!(gpu_ids, cpu_ids);

    for (g, (_, c)) in gpu.iter().zip(cpu.iter()) {
        assert!(
            (g.score - c).abs() < 1e-3,
            "score divergence: gpu={}, cpu={}",
            g.score,
            c
        );
    }
}

#[test]
fn removed_vectors_are_excluded_from_search() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = IntelContext::new_with_preference(true).unwrap();
    let mut storage = ctx
        .create_storage(2, GpuDistanceMetric::DotProduct)
        .unwrap();
    storage
        .add_vectors(&[
            mk("a", vec![1.0, 0.0]),
            mk("b", vec![0.9, 0.1]),
            mk("c", vec![0.0, 1.0]),
        ])
        .unwrap();
    storage.remove_vectors(&["a".to_string()]).unwrap();
    let r = storage.search(&[1.0, 0.0], 3).unwrap();
    assert!(r.iter().all(|x| x.id != "a"));
    assert_eq!(r[0].id, "b");
}
