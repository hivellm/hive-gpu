//! IVF benchmarks: build time vs `n_list`, search latency vs `nprobe`,
//! IVF vs brute-force head-to-head.
//!
//! Runs only when compiled with `--features cuda` on Linux/Windows and a
//! CUDA device is reachable.

#![cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hive_gpu::cuda::{CudaContext, CudaIvfIndex};
use hive_gpu::traits::GpuContext;
use hive_gpu::types::{GpuDistanceMetric, GpuVector, IvfConfig};
use std::collections::HashMap;
use std::sync::Arc;

fn seeded_rng(seed: u32) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn make_batch(n: usize, dim: usize, seed: u32) -> Vec<GpuVector> {
    let mut rng = seeded_rng(seed);
    (0..n)
        .map(|i| GpuVector {
            id: format!("v{i}"),
            data: (0..dim).map(|_| rng()).collect(),
            metadata: HashMap::new(),
        })
        .collect()
}

fn bench_build(c: &mut Criterion) {
    if !CudaContext::is_available() {
        eprintln!("[cuda_ivf bench] no CUDA device — skipping build benches");
        return;
    }
    let ctx = Arc::new(CudaContext::new().unwrap());

    let mut group = c.benchmark_group("cuda_ivf/build");
    for &n in &[10_000usize, 100_000usize] {
        let dim = 128;
        let vectors = make_batch(n, dim, 1);
        let n_list = 1 << ((n as f64).log2() / 2.0).ceil() as u32;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, _| {
            b.iter_batched(
                || {
                    CudaIvfIndex::new(
                        ctx.clone(),
                        dim,
                        GpuDistanceMetric::DotProduct,
                        IvfConfig {
                            n_list: n_list as usize,
                            nprobe: (n_list as usize / 16).max(1),
                            training_sample_size: (n / 4).min(8192),
                            kmeans_iters: 10,
                            seed: Some(1),
                        },
                    )
                    .unwrap()
                },
                |mut idx| {
                    idx.build(&vectors).unwrap();
                    black_box(idx.vector_count());
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn bench_search_vs_nprobe(c: &mut Criterion) {
    if !CudaContext::is_available() {
        eprintln!("[cuda_ivf bench] no CUDA device — skipping search benches");
        return;
    }
    let ctx = Arc::new(CudaContext::new().unwrap());
    let dim = 128;
    let n = 100_000;
    let vectors = make_batch(n, dim, 7);
    let n_list = 256;

    let mut rng = seeded_rng(99);
    let query: Vec<f32> = (0..dim).map(|_| rng()).collect();

    let mut idx = CudaIvfIndex::new(
        ctx.clone(),
        dim,
        GpuDistanceMetric::DotProduct,
        IvfConfig {
            n_list,
            nprobe: 1,
            training_sample_size: 8_192,
            kmeans_iters: 15,
            seed: Some(1),
        },
    )
    .unwrap();
    idx.build(&vectors).unwrap();

    let mut group = c.benchmark_group("cuda_ivf/search_dotproduct_100k");
    group.throughput(Throughput::Elements(n as u64));
    for &nprobe in &[1usize, 4, 16, 64, 256] {
        idx.set_nprobe(nprobe).unwrap();
        group.bench_with_input(BenchmarkId::new("nprobe", nprobe), &nprobe, |b, _| {
            b.iter(|| {
                let r = idx.search(&query, 10).unwrap();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_ivf_vs_bruteforce_1m(c: &mut Criterion) {
    if !CudaContext::is_available() {
        eprintln!("[cuda_ivf bench] no CUDA device — skipping head-to-head");
        return;
    }
    let ctx = Arc::new(CudaContext::new().unwrap());
    let dim = 128;
    let n = 1_000_000;
    let vectors = make_batch(n, dim, 11);
    let mut rng = seeded_rng(123);
    let query: Vec<f32> = (0..dim).map(|_| rng()).collect();

    // IVF index.
    let n_list = 1024;
    let mut ivf = CudaIvfIndex::new(
        ctx.clone(),
        dim,
        GpuDistanceMetric::DotProduct,
        IvfConfig {
            n_list,
            nprobe: 64,
            training_sample_size: 32_768,
            kmeans_iters: 12,
            seed: Some(2),
        },
    )
    .unwrap();
    ivf.build(&vectors).unwrap();

    // Brute-force baseline.
    let ctx_bf = CudaContext::new().unwrap();
    let mut bf = ctx_bf
        .create_storage(dim, GpuDistanceMetric::DotProduct)
        .unwrap();
    bf.add_vectors(&vectors).unwrap();

    let mut group = c.benchmark_group("cuda/1m_search_dotproduct");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("ivf_nprobe_64", |b| {
        b.iter(|| {
            let r = ivf.search(&query, 10).unwrap();
            black_box(r);
        })
    });
    group.bench_function("brute_force", |b| {
        b.iter(|| {
            let r = bf.search(&query, 10).unwrap();
            black_box(r);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_search_vs_nprobe,
    bench_ivf_vs_bruteforce_1m
);
criterion_main!(benches);
