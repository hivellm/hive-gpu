//! Timing and performance utilities

use std::time::{Duration, Instant};

/// Timing utility functions
pub mod timing_utils {
    use super::*;

    /// Measure execution time of a closure
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Measure execution time of an async closure
    pub async fn measure_time_async<F, Fut, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed();
        (result, duration)
    }

    /// Format duration for display
    pub fn format_duration(duration: Duration) -> String {
        if duration.as_secs() > 0 {
            format!("{:.2}s", duration.as_secs_f64())
        } else if duration.as_millis() > 0 {
            format!("{:.2}ms", duration.as_millis() as f64)
        } else {
            format!("{:.2}μs", duration.as_micros() as f64)
        }
    }
}

/// Performance utility functions
pub mod performance_utils {
    use super::*;

    /// Calculate throughput (operations per second)
    pub fn calculate_throughput(operations: usize, duration: Duration) -> f64 {
        if duration.as_secs_f64() > 0.0 {
            operations as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate latency (average time per operation)
    pub fn calculate_latency(duration: Duration, operations: usize) -> Duration {
        if operations > 0 {
            Duration::from_nanos(duration.as_nanos() as u64 / operations as u64)
        } else {
            Duration::ZERO
        }
    }

    /// Calculate efficiency (actual vs theoretical performance)
    pub fn calculate_efficiency(
        actual_throughput: f64,
        theoretical_throughput: f64
    ) -> f32 {
        if theoretical_throughput > 0.0 {
            (actual_throughput / theoretical_throughput) as f32
        } else {
            0.0
        }
    }

    /// Benchmark a function multiple times
    pub fn benchmark_function<F, R>(
        f: F,
        iterations: usize,
        warmup_iterations: usize
    ) -> BenchmarkResult
    where
        F: Fn() -> R,
    {
        // Warmup
        for _ in 0..warmup_iterations {
            let _ = f();
        }

        // Benchmark
        let mut durations = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let (_, duration) = timing_utils::measure_time(&f);
            durations.push(duration);
        }

        // Calculate statistics
        durations.sort();
        let min_duration = durations[0];
        let max_duration = durations[iterations - 1];
        let avg_duration = durations.iter().sum::<Duration>() / iterations as u32;
        let median_duration = durations[iterations / 2];

        BenchmarkResult {
            iterations,
            min_duration,
            max_duration,
            avg_duration,
            median_duration,
            total_duration: durations.iter().sum(),
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of iterations
    pub iterations: usize,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Average duration
    pub avg_duration: Duration,
    /// Median duration
    pub median_duration: Duration,
    /// Total duration
    pub total_duration: Duration,
}
