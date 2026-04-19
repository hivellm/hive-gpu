//! Smoke tests for the ROCm backend.
//!
//! ⚠️ NOT YET VALIDATED ON REAL HARDWARE — see the phase3b proposal.
//! Tests only compile on Linux with the `rocm` feature enabled and should
//! be run on an AMD GPU host before the task is archived. Tests are a
//! no-op when no HIP runtime is reachable, so the suite stays green on
//! CI hosts without ROCm installed.

#![cfg(all(feature = "rocm", target_os = "linux"))]

use hive_gpu::rocm::RocmContext;
use hive_gpu::traits::{GpuBackend, GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;

fn skip_if_no_gpu() -> bool {
    if !RocmContext::is_available() {
        eprintln!("[rocm_smoke] no ROCm device detected; test is a no-op");
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
    let ctx = RocmContext::new().expect("RocmContext::new on a host with a ROCm device");
    let info = GpuBackend::device_info(&*ctx);
    assert_eq!(info.backend, "ROCm");
    assert!(!info.name.is_empty(), "device name must be populated");
    assert!(info.total_vram_bytes > 0, "total VRAM must be > 0");
    assert!(
        info.available_vram_bytes <= info.total_vram_bytes,
        "available <= total invariant"
    );
    assert!(info.driver_version.starts_with("ROCm"));
    let cc = info
        .compute_capability
        .as_ref()
        .expect("ROCm backend populates gfx string");
    assert!(cc.starts_with("gfx"), "gfx prefix expected, got {cc}");
}

#[test]
fn batch_add_then_cosine_search_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = RocmContext::new().unwrap();
    let mut storage = ctx
        .create_storage(8, GpuDistanceMetric::Cosine)
        .expect("create_storage");

    let vectors = vec![
        mk("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mk("b", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mk("c", vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mk("mix", vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    ];
    storage.add_vectors(&vectors).expect("add_vectors");
    assert_eq!(storage.vector_count(), 4);

    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = storage.search(&query, 4).expect("search");
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].id, "a");
    assert!((results[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn euclidean_returns_nearest_neighbour_first() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = RocmContext::new().unwrap();
    let mut storage = ctx.create_storage(4, GpuDistanceMetric::Euclidean).unwrap();
    storage
        .add_vectors(&[
            mk("near", vec![1.0, 1.0, 1.0, 1.0]),
            mk("far", vec![9.0, 9.0, 9.0, 9.0]),
            mk("mid", vec![3.0, 3.0, 3.0, 3.0]),
        ])
        .unwrap();
    let q = vec![1.1, 1.1, 1.1, 1.1];
    let r = storage.search(&q, 3).unwrap();
    assert_eq!(r[0].id, "near");
    assert_eq!(r[1].id, "mid");
    assert_eq!(r[2].id, "far");
}

#[test]
fn buffer_growth_preserves_existing_data() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = RocmContext::new().unwrap();
    let mut storage = ctx
        .create_storage(16, GpuDistanceMetric::DotProduct)
        .unwrap();

    let first_batch: Vec<GpuVector> = (0..1200)
        .map(|i| {
            let mut data = vec![0.0f32; 16];
            data[i % 16] = (i as f32) + 1.0;
            mk(&format!("v{i}"), data)
        })
        .collect();
    storage.add_vectors(&first_batch).unwrap();
    assert_eq!(storage.vector_count(), 1200);

    let second_batch: Vec<GpuVector> = (1200..2400)
        .map(|i| {
            let mut data = vec![0.0f32; 16];
            data[i % 16] = (i as f32) + 1.0;
            mk(&format!("v{i}"), data)
        })
        .collect();
    storage.add_vectors(&second_batch).unwrap();
    assert_eq!(storage.vector_count(), 2400);

    let mut query = vec![0.0f32; 16];
    query[5] = 100.0;
    let results = storage.search(&query, 1).unwrap();
    // Same check as the CUDA smoke test — the vector with the largest
    // value at index 5 is v2389 (i=2389, i%16=5, value 2390).
    assert_eq!(results[0].id, "v2389");
}

#[test]
fn dotproduct_matches_cpu_reference_on_random_batch() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = RocmContext::new().unwrap();
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
    assert_eq!(gpu_ids, cpu_ids, "GPU top-10 must match CPU top-10");

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
    let ctx = RocmContext::new().unwrap();
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
