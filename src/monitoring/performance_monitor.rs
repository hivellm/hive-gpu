//! Performance Monitoring and Benchmarking
//!
//! This module provides performance monitoring, benchmarking, and statistics
//! collection for GPU operations.

use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance Monitor for tracking GPU operation performance
pub struct PerformanceMonitor {
    /// Operation statistics
    stats: HashMap<String, OperationStats>,
    /// Start time for current operation
    current_operation: Option<(String, Instant)>,
}

/// Statistics for a specific operation
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation name
    pub operation: String,
    /// Total number of operations
    pub count: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Average duration
    pub avg_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Operations per second
    pub ops_per_second: f64,
}

/// Overall performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total operations performed
    pub total_operations: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Average operations per second
    pub avg_ops_per_second: f64,
    /// Operation breakdown
    pub operation_stats: Vec<OperationStats>,
}

/// Benchmark result for a specific operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            current_operation: None,
        }
    }

    /// Start timing an operation
    pub fn start_operation(&mut self, operation: String) {
        self.current_operation = Some((operation, Instant::now()));
    }

    /// End timing the current operation
    pub fn end_operation(&mut self) -> Result<Duration> {
        if let Some((operation, start_time)) = self.current_operation.take() {
            let duration = start_time.elapsed();
            self.record_operation(operation, duration);
            Ok(duration)
        } else {
            Err(crate::error::HiveGpuError::Other(
                "No operation in progress".to_string(),
            ))
        }
    }

    /// Record an operation with its duration
    pub fn record_operation(&mut self, operation: String, duration: Duration) {
        let stats = self
            .stats
            .entry(operation.clone())
            .or_insert(OperationStats {
                operation: operation.clone(),
                count: 0,
                total_duration: Duration::ZERO,
                avg_duration: Duration::ZERO,
                min_duration: Duration::MAX,
                max_duration: Duration::ZERO,
                ops_per_second: 0.0,
            });

        stats.count += 1;
        stats.total_duration += duration;
        stats.avg_duration = stats.total_duration / stats.count as u32;
        stats.min_duration = stats.min_duration.min(duration);
        stats.max_duration = stats.max_duration.max(duration);
        stats.ops_per_second = stats.count as f64 / stats.total_duration.as_secs_f64();
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        let total_operations: usize = self.stats.values().map(|s| s.count).sum();
        let total_duration: Duration = self.stats.values().map(|s| s.total_duration).sum();
        let avg_ops_per_second = if total_duration.as_secs_f64() > 0.0 {
            total_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        PerformanceStats {
            total_operations,
            total_duration,
            avg_ops_per_second,
            operation_stats: self.stats.values().cloned().collect(),
        }
    }

    /// Get statistics for a specific operation
    pub fn get_operation_stats(&self, operation: &str) -> Option<&OperationStats> {
        self.stats.get(operation)
    }

    /// Clear all statistics
    pub fn clear_stats(&mut self) {
        self.stats.clear();
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let stats = self.get_performance_stats();
        let mut report = String::new();

        report.push_str("Performance Report:\n");
        report.push_str(&format!("  Total Operations: {}\n", stats.total_operations));
        report.push_str(&format!(
            "  Total Duration: {:.2}s\n",
            stats.total_duration.as_secs_f64()
        ));
        report.push_str(&format!(
            "  Average Ops/sec: {:.2}\n",
            stats.avg_ops_per_second
        ));

        report.push_str("\nOperation Breakdown:\n");
        for op_stats in &stats.operation_stats {
            report.push_str(&format!("  {}:\n", op_stats.operation));
            report.push_str(&format!("    Count: {}\n", op_stats.count));
            report.push_str(&format!(
                "    Avg Duration: {:.2}ms\n",
                op_stats.avg_duration.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "    Min Duration: {:.2}ms\n",
                op_stats.min_duration.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "    Max Duration: {:.2}ms\n",
                op_stats.max_duration.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!("    Ops/sec: {:.2}\n", op_stats.ops_per_second));
        }

        report
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
