//! Metal IVF index tests.
//!
//! ⚠️ NOT YET VALIDATED ON REAL HARDWARE — see the commit message of
//! phase4c_add-metal-ivf-index. Tests mirror `tests/cuda_ivf.rs`. They
//! will only compile on macOS with the `metal-native` feature enabled and
//! should be run on Apple Silicon hardware before the task is archived.

#![cfg(all(target_os = "macos", feature = "metal-native"))]

use hive_gpu::metal::{MetalIvfIndex, MetalNativeContext};
use hive_gpu::types::{GpuDistanceMetric, GpuVector, IvfConfig};
use std::collections::HashMap;
use std::sync::Arc;

fn skip_if_no_device() -> bool {
    if MetalNativeContext::new().is_err() {
        eprintln!("[metal_ivf] no Metal device detected; test is a no-op");
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

fn cpu_top_k_dot(vectors: &[GpuVector], query: &[f32], k: usize) -> Vec<String> {
    let mut scored: Vec<(&str, f32)> = vectors
        .iter()
        .map(|v| {
            let dot = v.data.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
            (v.id.as_str(), dot)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
        .into_iter()
        .take(k)
        .map(|(id, _)| id.to_string())
        .collect()
}

fn recall(returned: &[String], ground_truth: &[String]) -> f32 {
    let gt: std::collections::HashSet<&str> = ground_truth.iter().map(|s| s.as_str()).collect();
    let hits = returned
        .iter()
        .filter(|id| gt.contains(id.as_str()))
        .count();
    hits as f32 / ground_truth.len() as f32
}

#[test]
fn new_rejects_bad_config() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    assert!(
        MetalIvfIndex::new(
            ctx.clone(),
            0,
            GpuDistanceMetric::DotProduct,
            IvfConfig::default()
        )
        .is_err()
    );
    assert!(
        MetalIvfIndex::new(
            ctx.clone(),
            16,
            GpuDistanceMetric::DotProduct,
            IvfConfig {
                n_list: 0,
                ..IvfConfig::default()
            }
        )
        .is_err()
    );
    assert!(
        MetalIvfIndex::new(
            ctx,
            16,
            GpuDistanceMetric::DotProduct,
            IvfConfig {
                n_list: 8,
                nprobe: 16,
                ..IvfConfig::default()
            }
        )
        .is_err()
    );
}

#[test]
fn build_rejects_empty_and_too_small_inputs() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let mut idx = MetalIvfIndex::new(
        ctx,
        4,
        GpuDistanceMetric::DotProduct,
        IvfConfig {
            n_list: 16,
            nprobe: 4,
            ..IvfConfig::default()
        },
    )
    .unwrap();
    assert!(idx.build(&[]).is_err());
    let too_few: Vec<GpuVector> = (0..8)
        .map(|i| mk(&format!("v{i}"), vec![i as f32; 4]))
        .collect();
    assert!(idx.build(&too_few).is_err());
}

#[test]
fn set_nprobe_validates() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let dim = 4;
    let n = 64;
    let mut rng = SeededRng::new(1);
    let vectors: Vec<GpuVector> = (0..n)
        .map(|i| mk(&format!("v{i}"), rng.gen_vec(dim)))
        .collect();

    let mut idx = MetalIvfIndex::new(
        ctx,
        dim,
        GpuDistanceMetric::DotProduct,
        IvfConfig {
            n_list: 8,
            nprobe: 2,
            training_sample_size: n,
            kmeans_iters: 5,
            seed: Some(3),
        },
    )
    .unwrap();
    idx.build(&vectors).unwrap();

    assert_eq!(idx.nprobe(), 2);
    idx.set_nprobe(4).unwrap();
    assert_eq!(idx.nprobe(), 4);
    assert!(idx.set_nprobe(0).is_err());
    assert!(idx.set_nprobe(9).is_err());
}

#[test]
fn recall_at_10_against_bruteforce_dotproduct() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let n = 5_000;
    let dim = 32;
    let mut rng = SeededRng::new(0xABCD);
    let vectors: Vec<GpuVector> = (0..n)
        .map(|i| mk(&format!("v{i}"), rng.gen_vec(dim)))
        .collect();

    let n_list = 64;
    let mut idx = MetalIvfIndex::new(
        ctx,
        dim,
        GpuDistanceMetric::DotProduct,
        IvfConfig {
            n_list,
            nprobe: n_list / 4,
            training_sample_size: 1_024,
            kmeans_iters: 12,
            seed: Some(7),
        },
    )
    .unwrap();
    idx.build(&vectors).unwrap();

    let mut total = 0.0f32;
    let queries = 30;
    for _ in 0..queries {
        let q = rng.gen_vec(dim);
        let gt = cpu_top_k_dot(&vectors, &q, 10);
        let got: Vec<String> = idx
            .search(&q, 10)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        total += recall(&got, &gt);
    }
    let mean = total / queries as f32;
    assert!(
        mean >= 0.65,
        "recall@10 was {mean:.3}, expected >= 0.65 for nprobe = n_list/4"
    );
    eprintln!("[metal_ivf] recall@10 = {mean:.3}");
}

#[test]
fn higher_nprobe_increases_recall() {
    if skip_if_no_device() {
        return;
    }
    let ctx = Arc::new(MetalNativeContext::new().unwrap());
    let n = 2_000;
    let dim = 16;
    let mut rng = SeededRng::new(99);
    let vectors: Vec<GpuVector> = (0..n)
        .map(|i| mk(&format!("v{i}"), rng.gen_vec(dim)))
        .collect();

    let n_list = 32;
    let mut idx = MetalIvfIndex::new(
        ctx,
        dim,
        GpuDistanceMetric::DotProduct,
        IvfConfig {
            n_list,
            nprobe: 2,
            training_sample_size: n,
            kmeans_iters: 10,
            seed: Some(2024),
        },
    )
    .unwrap();
    idx.build(&vectors).unwrap();

    let q = rng.gen_vec(dim);
    let gt = cpu_top_k_dot(&vectors, &q, 10);

    let low: Vec<String> = idx
        .search(&q, 10)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    let low_rec = recall(&low, &gt);

    idx.set_nprobe(n_list).unwrap();
    let full: Vec<String> = idx
        .search(&q, 10)
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    let full_rec = recall(&full, &gt);

    assert!(full_rec >= low_rec, "full-scan recall must be ≥ restricted");
    assert!(
        full_rec >= 0.9,
        "nprobe = n_list should yield near-perfect recall, got {full_rec:.3}"
    );
}
