//! Metal brute-force search tests.
//!
//! ⚠️ NOT YET VALIDATED ON REAL HARDWARE — see commit message of
//! phase4a_finish-metal-bruteforce-search. Tests are written to mirror
//! `tests/cuda_ivf.rs` in scope. They will only compile on macOS with the
//! `metal-native` feature enabled, and should be run on Apple Silicon
//! hardware before the task is marked complete.

#![cfg(all(target_os = "macos", feature = "metal-native"))]

use hive_gpu::metal::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;
use std::sync::Arc;

fn skip_if_no_device() -> bool {
    if MetalNativeContext::new().is_err() {
        eprintln!("[metal_bruteforce] no Metal device detected; test is a no-op");
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

struct SeededRng(u64);
impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
    }
    fn gen_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

#[test]
fn self_query_returns_exact_match_cosine() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let mut storage = ctx.create_storage(8, GpuDistanceMetric::Cosine).unwrap();
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    storage.add_vectors(&[mk("a", v.clone())]).unwrap();
    let results = storage.search(&v, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "a");
    assert!((results[0].score - 1.0).abs() < 1e-4);
}

#[test]
fn euclidean_ranks_nearest_first() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
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
fn dotproduct_matches_cpu_reference_on_random_batch() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let dim = 32;
    let n = 500;
    let mut rng = SeededRng::new(0xC0FFEE);

    let vectors: Vec<GpuVector> = (0..n)
        .map(|i| mk(&format!("v{i}"), rng.gen_vec(dim)))
        .collect();
    let cpu_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();

    let mut storage = ctx
        .create_storage(dim, GpuDistanceMetric::DotProduct)
        .unwrap();
    storage.add_vectors(&vectors).unwrap();

    let q = rng.gen_vec(dim);

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
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
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

#[test]
fn empty_storage_returns_empty_results() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let storage = ctx
        .create_storage(4, GpuDistanceMetric::DotProduct)
        .unwrap();
    let r = storage.search(&[1.0, 2.0, 3.0, 4.0], 10).unwrap();
    assert!(r.is_empty());
}

#[test]
fn dimension_mismatch_returns_error() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let storage = ctx
        .create_storage(4, GpuDistanceMetric::DotProduct)
        .unwrap();
    let err = storage.search(&[1.0, 2.0], 1).expect_err("dim mismatch");
    assert!(format!("{err}").contains("Dimension"));
}
