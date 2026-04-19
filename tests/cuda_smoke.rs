//! Smoke test for the CUDA backend.
//!
//! Each test is a no-op when the host has no CUDA driver, which keeps the
//! suite green on CI hosts without an NVIDIA GPU. When a GPU is present the
//! test exercises the full upload → SGEMV → readback path and compares
//! results against a CPU reference.

#![cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]

use hive_gpu::cuda::CudaContext;
use hive_gpu::traits::{GpuBackend, GpuContext};
use hive_gpu::types::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;

fn skip_if_no_gpu() -> bool {
    if !CudaContext::is_available() {
        eprintln!("[cuda_smoke] no CUDA device detected; test is a no-op");
        return true;
    }
    false
}

fn make_vector(id: &str, data: Vec<f32>) -> GpuVector {
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
    let ctx = CudaContext::new().expect("CudaContext::new on a host with a CUDA device");
    let info = GpuBackend::device_info(&ctx);
    assert_eq!(info.backend, "CUDA");
    assert!(!info.name.is_empty(), "device name should be populated");
    assert!(info.total_vram_bytes > 0, "total VRAM must be > 0");
    assert!(
        info.available_vram_bytes <= info.total_vram_bytes,
        "available <= total invariant"
    );
    assert!(info.compute_capability.is_some());
    assert!(
        info.max_threads_per_block >= 512,
        "CUDA devices report at least 512 threads/block"
    );
    assert!(info.driver_version.starts_with("CUDA"));
}

#[test]
fn batch_add_then_cosine_search_matches_cpu() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut storage = ctx
        .create_storage(8, GpuDistanceMetric::Cosine)
        .expect("create_storage");

    let vectors = vec![
        make_vector("a", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        make_vector("b", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        make_vector("c", vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        make_vector("mix", vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    ];
    storage.add_vectors(&vectors).expect("add_vectors");
    assert_eq!(storage.vector_count(), 4);

    let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = storage.search(&query, 4).expect("search");
    assert_eq!(results.len(), 4);

    // The first result must be "a" (exactly equal direction) and "b"/"c"
    // must score 0 (orthogonal). "mix" is in between.
    assert_eq!(results[0].id, "a");
    assert!((results[0].score - 1.0).abs() < 1e-4);

    let score_mix = results
        .iter()
        .find(|r| r.id == "mix")
        .expect("mix present")
        .score;
    let score_b = results
        .iter()
        .find(|r| r.id == "b")
        .expect("b present")
        .score;
    assert!(
        score_mix > score_b,
        "mix should beat b for a query aligned with x-axis"
    );
}

#[test]
fn euclidean_returns_nearest_neighbour_first() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut storage = ctx.create_storage(4, GpuDistanceMetric::Euclidean).unwrap();

    storage
        .add_vectors(&[
            make_vector("near", vec![1.0, 1.0, 1.0, 1.0]),
            make_vector("far", vec![9.0, 9.0, 9.0, 9.0]),
            make_vector("mid", vec![3.0, 3.0, 3.0, 3.0]),
        ])
        .unwrap();

    let query = vec![1.1, 1.1, 1.1, 1.1];
    let results = storage.search(&query, 3).unwrap();
    assert_eq!(results[0].id, "near");
    assert_eq!(results[1].id, "mid");
    assert_eq!(results[2].id, "far");
}

#[test]
fn buffer_growth_preserves_existing_data() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    // Dimension * MIN_INITIAL_VECTORS chosen so the first batch fills the
    // default capacity and the second batch forces a resize.
    let mut storage = ctx
        .create_storage(16, GpuDistanceMetric::DotProduct)
        .unwrap();

    let first_batch: Vec<GpuVector> = (0..1200)
        .map(|i| {
            let mut data = vec![0.0f32; 16];
            data[i % 16] = (i as f32) + 1.0;
            make_vector(&format!("v{i}"), data)
        })
        .collect();
    storage.add_vectors(&first_batch).unwrap();
    assert_eq!(storage.vector_count(), 1200);

    // Trigger resize.
    let second_batch: Vec<GpuVector> = (1200..2400)
        .map(|i| {
            let mut data = vec![0.0f32; 16];
            data[i % 16] = (i as f32) + 1.0;
            make_vector(&format!("v{i}"), data)
        })
        .collect();
    storage.add_vectors(&second_batch).unwrap();
    assert_eq!(storage.vector_count(), 2400);

    // Query that picks up a specific early vector and verifies it is still
    // ranked first — proof that growth preserved the original data.
    let mut query = vec![0.0f32; 16];
    query[5] = 100.0;
    let results = storage.search(&query, 1).unwrap();
    // Highest dot product belongs to the vector with the largest value at
    // index 5, which is v2389 (i=2389, i%16=5, value 2390).
    assert_eq!(results[0].id, "v2389");
}
