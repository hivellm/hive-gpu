//! CUDA backend throughput benchmarks.
//!
//! Runs only when compiled with `--features cuda` on Linux/Windows and when a
//! CUDA device is actually present. The benches compare raw CPU reference
//! throughput against the GPU-accelerated path for add and search.

#![cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use hive_gpu::cuda::CudaContext;
use hive_gpu::traits::GpuContext;
use hive_gpu::types::{GpuDistanceMetric, GpuVector};
use std::collections::HashMap;

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

fn cpu_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored
}

fn bench_add_vectors(c: &mut Criterion) {
    if !CudaContext::is_available() {
        eprintln!("[cuda_ops bench] no CUDA device available — skipping add benches");
        return;
    }
    let ctx = CudaContext::new().unwrap();

    let mut group = c.benchmark_group("cuda/add_vectors");
    for &n in &[1_000usize, 10_000usize] {
        let dim = 128;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let storage = ctx
                        .create_storage(dim, GpuDistanceMetric::DotProduct)
                        .unwrap();
                    let batch = make_batch(n, dim, 42);
                    (storage, batch)
                },
                |(mut storage, batch)| {
                    storage.add_vectors(&batch).unwrap();
                    black_box(storage.vector_count());
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn bench_search(c: &mut Criterion) {
    if !CudaContext::is_available() {
        eprintln!("[cuda_ops bench] no CUDA device available — skipping search benches");
        return;
    }
    let ctx = CudaContext::new().unwrap();
    let dim = 128;

    let mut group = c.benchmark_group("cuda/search_dotproduct");
    for &n in &[1_000usize, 10_000usize, 100_000usize] {
        // Prepare once per input size.
        let mut storage = ctx
            .create_storage(dim, GpuDistanceMetric::DotProduct)
            .unwrap();
        let batch = make_batch(n, dim, 7);
        let cpu_data: Vec<Vec<f32>> = batch.iter().map(|v| v.data.clone()).collect();
        storage.add_vectors(&batch).unwrap();
        let mut rng = seeded_rng(99);
        let query: Vec<f32> = (0..dim).map(|_| rng()).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |b, _| {
            b.iter(|| {
                let r = storage.search(&query, 10).unwrap();
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("cpu_reference", n), &n, |b, _| {
            b.iter(|| {
                let r = cpu_search(&cpu_data, &query, 10);
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_add_vectors, bench_search);
criterion_main!(benches);
