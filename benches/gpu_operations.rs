//! Benchmarks for hive-gpu operations
//!
//! These benchmarks measure the performance of GPU operations
//! and compare them with CPU implementations.

// criterion_main is used by the macro expansion
#[allow(unused_imports)]
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hive_gpu::GpuVector;
use std::collections::HashMap;

// Helper function to create test vectors (used by all benchmarks)
fn create_test_vectors(count: usize, dimension: usize) -> Vec<GpuVector> {
    (0..count)
        .map(|i| GpuVector {
            id: format!("vec_{}", i),
            data: vec![i as f32; dimension],
            metadata: HashMap::new(),
        })
        .collect()
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn bench_metal_vector_addition(c: &mut Criterion) {
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::{GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("metal_vector_addition");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let context = MetalNativeContext::new().unwrap();
            let mut storage = context
                .create_storage(512, GpuDistanceMetric::Cosine)
                .unwrap();
            let vectors = create_test_vectors(size, 512);

            b.iter(|| {
                storage.add_vectors(&vectors).unwrap();
            });
        });
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn bench_metal_vector_search(c: &mut Criterion) {
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::{GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("metal_vector_search");

    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let context = MetalNativeContext::new().unwrap();
            let mut storage = context
                .create_storage(512, GpuDistanceMetric::Cosine)
                .unwrap();

            // Pre-populate with vectors
            let vectors = create_test_vectors(size, 512);
            storage.add_vectors(&vectors).unwrap();

            let query = vec![500.0; 512];

            b.iter(|| {
                storage.search(&query, 10).unwrap();
            });
        });
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn bench_metal_hnsw_construction(c: &mut Criterion) {
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::{GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("metal_hnsw_construction");

    for size in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let context = MetalNativeContext::new().unwrap();
            let config = hive_gpu::HnswConfig {
                max_connections: 16,
                ef_construction: 200,
                ef_search: 50,
                max_level: 10,
                level_multiplier: 1.0,
                seed: Some(42),
            };
            let mut storage = context
                .create_storage_with_config(512, GpuDistanceMetric::Cosine, config)
                .unwrap();
            let vectors = create_test_vectors(size, 512);

            b.iter(|| {
                storage.add_vectors(&vectors).unwrap();
            });
        });
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn bench_metal_batch_operations(c: &mut Criterion) {
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::{GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("metal_batch_operations");

    for batch_size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &batch_size| {
                let context = MetalNativeContext::new().unwrap();
                let mut storage = context
                    .create_storage(512, GpuDistanceMetric::Cosine)
                    .unwrap();
                let vectors = create_test_vectors(batch_size, 512);

                b.iter(|| {
                    // Test batch add
                    storage.add_vectors(&vectors).unwrap();

                    // Test batch search
                    let query = vec![250.0; 512];
                    storage.search(&query, 10).unwrap();
                });
            },
        );
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn bench_metal_distance_metrics(c: &mut Criterion) {
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::{GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("metal_distance_metrics");

    let metrics = [
        ("cosine", GpuDistanceMetric::Cosine),
        ("euclidean", GpuDistanceMetric::Euclidean),
        ("dot_product", GpuDistanceMetric::DotProduct),
    ];

    for (name, metric) in metrics.iter() {
        group.bench_with_input(BenchmarkId::new("search", name), name, |b, _| {
            let context = MetalNativeContext::new().unwrap();
            let mut storage = context.create_storage(512, *metric).unwrap();

            // Pre-populate with vectors
            let vectors = create_test_vectors(10000, 512);
            storage.add_vectors(&vectors).unwrap();

            let query = vec![5000.0; 512];

            b.iter(|| {
                storage.search(&query, 10).unwrap();
            });
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_vector_operations(c: &mut Criterion) {
    use hive_gpu::cuda::CudaContext;
    use hive_gpu::{GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("cuda_vector_operations");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let context = CudaContext::new().unwrap();
            let mut storage = context
                .create_storage(512, GpuDistanceMetric::Cosine)
                .unwrap();
            let vectors = create_test_vectors(size, 512);

            b.iter(|| {
                storage.add_vectors(&vectors).unwrap();
            });
        });
    }

    group.finish();
}

// CPU baseline benchmarks for comparison
fn bench_cpu_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vector_operations");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let vectors = create_test_vectors(size, 512);

            b.iter(|| {
                // Simulate CPU vector operations
                let mut results = Vec::new();
                for vector in &vectors {
                    let score = vector.data.iter().sum::<f32>();
                    results.push(score);
                }
                results.sort_by(|a, b| b.partial_cmp(a).unwrap());
                results.truncate(10);
            });
        });
    }

    group.finish();
}

// Memory usage benchmarks
#[cfg(all(target_os = "macos", feature = "metal-native"))]
fn bench_metal_memory_usage(c: &mut Criterion) {
    use hive_gpu::metal::MetalNativeContext;
    use hive_gpu::{GpuBackend, GpuContext, GpuDistanceMetric};

    let mut group = c.benchmark_group("metal_memory_usage");

    for size in [1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let context = MetalNativeContext::new().unwrap();
            let mut storage = context
                .create_storage(512, GpuDistanceMetric::Cosine)
                .unwrap();
            let vectors = create_test_vectors(size, 512);

            b.iter(|| {
                storage.add_vectors(&vectors).unwrap();

                // Measure memory usage
                let stats = GpuBackend::memory_stats(&context);
                assert!(stats.total_allocated > 0);
            });
        });
    }

    group.finish();
}

// Configure benchmark groups
#[cfg(all(target_os = "macos", feature = "metal-native"))]
criterion_group!(
    metal_benches,
    bench_metal_vector_addition,
    bench_metal_vector_search,
    bench_metal_hnsw_construction,
    bench_metal_batch_operations,
    bench_metal_distance_metrics,
    bench_metal_memory_usage
);

#[cfg(feature = "cuda")]
criterion_group!(cuda_benches, bench_cuda_vector_operations);

criterion_group!(cpu_benches, bench_cpu_vector_operations);

// Main benchmark runner
#[cfg(all(target_os = "macos", feature = "metal-native"))]
criterion_main!(metal_benches, cpu_benches);

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
criterion_main!(cuda_benches, cpu_benches);

// Fallback: CPU-only benchmarks for unsupported configurations
// (e.g., Windows/Linux without CUDA, or any platform without GPU features)
#[cfg(not(any(
    all(target_os = "macos", feature = "metal-native"),
    all(not(target_os = "macos"), feature = "cuda")
)))]
criterion_main!(cpu_benches);
