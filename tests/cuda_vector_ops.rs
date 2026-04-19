//! Vector storage operations tests on a real CUDA device.

#![cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]

use hive_gpu::cuda::CudaContext;
use hive_gpu::traits::GpuContext;
use hive_gpu::types::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;

fn skip_if_no_gpu() -> bool {
    if !CudaContext::is_available() {
        eprintln!("[cuda_vector_ops] no CUDA device detected; test is a no-op");
        return true;
    }
    false
}

fn vec_with(id: &str, data: Vec<f32>) -> GpuVector {
    GpuVector {
        id: id.to_string(),
        data,
        metadata: HashMap::new(),
    }
}

#[test]
fn single_add_increments_count() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(4, GpuDistanceMetric::DotProduct)
        .unwrap();
    s.add_vectors(&[vec_with("v1", vec![1.0, 2.0, 3.0, 4.0])])
        .unwrap();
    assert_eq!(s.vector_count(), 1);
    assert_eq!(s.dimension(), 4);
}

#[test]
fn batch_add_with_duplicate_inside_batch_fails_atomically() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(2, GpuDistanceMetric::DotProduct)
        .unwrap();
    let batch = vec![
        vec_with("x", vec![1.0, 1.0]),
        vec_with("x", vec![2.0, 2.0]), // duplicate
    ];
    let err = s.add_vectors(&batch).expect_err("duplicate must fail");
    assert!(format!("{err}").contains("duplicate"));
    assert_eq!(s.vector_count(), 0, "failed batch must not mutate state");
}

#[test]
fn batch_add_with_bad_dimension_fails() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(3, GpuDistanceMetric::DotProduct)
        .unwrap();
    let err = s
        .add_vectors(&[vec_with("oops", vec![1.0, 2.0])])
        .expect_err("dimension mismatch");
    assert!(format!("{err}").contains("Dimension"));
}

#[test]
fn non_finite_component_rejected() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(2, GpuDistanceMetric::DotProduct)
        .unwrap();
    let err = s
        .add_vectors(&[vec_with("nan", vec![1.0, f32::NAN])])
        .expect_err("NaN rejected");
    assert!(format!("{err}").contains("non-finite"));
}

#[test]
fn remove_then_search_skips_removed() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(2, GpuDistanceMetric::DotProduct)
        .unwrap();
    s.add_vectors(&[
        vec_with("a", vec![1.0, 0.0]),
        vec_with("b", vec![0.9, 0.1]),
        vec_with("c", vec![0.0, 1.0]),
    ])
    .unwrap();
    s.remove_vectors(&["a".to_string()]).unwrap();
    assert_eq!(s.vector_count(), 2);

    let res = s.search(&[1.0, 0.0], 3).unwrap();
    assert!(res.iter().all(|r| r.id != "a"));
    assert_eq!(res[0].id, "b");
}

#[test]
fn clear_resets_counts() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(2, GpuDistanceMetric::DotProduct)
        .unwrap();
    s.add_vectors(&[vec_with("a", vec![1.0, 0.0]), vec_with("b", vec![0.0, 1.0])])
        .unwrap();
    s.clear().unwrap();
    assert_eq!(s.vector_count(), 0);

    // Adding after clear works and uses fresh IDs.
    s.add_vectors(&[vec_with("a", vec![1.0, 0.0])]).unwrap();
    assert_eq!(s.vector_count(), 1);
}

#[test]
fn get_vector_roundtrips_through_gpu_memory() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let mut s = ctx
        .create_storage(4, GpuDistanceMetric::DotProduct)
        .unwrap();
    let original = vec![0.25, -0.5, 1.5, -2.0];
    s.add_vectors(&[vec_with("round", original.clone())])
        .unwrap();

    let fetched = s
        .get_vector("round")
        .unwrap()
        .expect("stored vector must be retrievable");
    assert_eq!(fetched.id, "round");
    assert_eq!(fetched.data, original);
}

#[test]
fn search_matches_cpu_reference_for_large_random_batch() {
    if skip_if_no_gpu() {
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let dim = 64;
    let n = 2000;

    let mut rng_state: u32 = 0x1234_5678;
    let mut rng = || {
        rng_state = rng_state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let vectors: Vec<GpuVector> = (0..n)
        .map(|i| {
            let data = (0..dim).map(|_| rng()).collect::<Vec<_>>();
            vec_with(&format!("v{i}"), data)
        })
        .collect();
    let cpu_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();

    let mut s = ctx
        .create_storage(dim, GpuDistanceMetric::DotProduct)
        .unwrap();
    s.add_vectors(&vectors).unwrap();
    let query: Vec<f32> = (0..dim).map(|_| rng()).collect();

    let gpu = s.search(&query, 10).unwrap();

    // CPU reference: raw dot products.
    let mut cpu: Vec<(usize, f32)> = cpu_data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    cpu.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    cpu.truncate(10);
    let cpu_ids: Vec<String> = cpu.iter().map(|(i, _)| format!("v{i}")).collect();
    let gpu_ids: Vec<String> = gpu.iter().map(|r| r.id.clone()).collect();

    assert_eq!(gpu_ids, cpu_ids, "GPU top-10 must match CPU top-10");

    // Numerical agreement within 1e-3 per element (allows for fma fusion).
    for (gpu_res, (_, cpu_score)) in gpu.iter().zip(cpu.iter()) {
        assert!(
            (gpu_res.score - cpu_score).abs() < 1e-3,
            "score divergence: gpu={}, cpu={}",
            gpu_res.score,
            cpu_score
        );
    }
}
