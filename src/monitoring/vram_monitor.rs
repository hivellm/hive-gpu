//! VRAM Monitoring and Validation
//!
//! This module provides VRAM usage monitoring, validation, and optimization
//! for GPU operations.

use crate::error::{HiveGpuError, Result};
use crate::traits::{GpuMonitor, VramBufferInfo, VramStats};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// VRAM Monitor for tracking GPU memory usage
pub struct VramMonitor {
    /// Total VRAM capacity in bytes
    total_vram: usize,
    /// Currently allocated VRAM in bytes
    allocated_vram: usize,
    /// Buffer information map
    buffers: HashMap<usize, VramBufferInfo>,
    /// Next buffer ID
    next_buffer_id: usize,
}

impl VramMonitor {
    /// Create a new VRAM monitor
    pub fn new(total_vram: usize) -> Self {
        Self {
            total_vram,
            allocated_vram: 0,
            buffers: HashMap::new(),
            next_buffer_id: 0,
        }
    }

    /// Allocate VRAM for a buffer
    pub fn allocate_buffer(
        &mut self,
        size: usize,
        buffer_type: crate::traits::BufferType,
    ) -> Result<usize> {
        if self.allocated_vram + size > self.total_vram {
            return Err(HiveGpuError::VramLimitExceeded {
                requested: self.allocated_vram + size,
                limit: self.total_vram,
            });
        }

        let buffer_id = self.next_buffer_id;
        self.next_buffer_id += 1;

        let buffer_info = VramBufferInfo {
            buffer_id,
            size,
            buffer_type,
            allocated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.buffers.insert(buffer_id, buffer_info);
        self.allocated_vram += size;

        Ok(buffer_id)
    }

    /// Deallocate VRAM for a buffer
    pub fn deallocate_buffer(&mut self, buffer_id: usize) -> Result<()> {
        if let Some(buffer_info) = self.buffers.remove(&buffer_id) {
            self.allocated_vram = self.allocated_vram.saturating_sub(buffer_info.size);
            Ok(())
        } else {
            Err(HiveGpuError::Other(format!(
                "Buffer {} not found",
                buffer_id
            )))
        }
    }

    /// Get current VRAM statistics
    pub fn get_vram_stats(&self) -> VramStats {
        VramStats {
            total_vram: self.total_vram,
            allocated_vram: self.allocated_vram,
            available_vram: self.total_vram - self.allocated_vram,
            utilization: self.allocated_vram as f32 / self.total_vram as f32,
            buffer_count: self.buffers.len(),
        }
    }

    /// Validate that all operations are in VRAM
    pub fn validate_all_vram(&self) -> Result<()> {
        let stats = self.get_vram_stats();

        if stats.utilization > 0.95 {
            return Err(HiveGpuError::VramLimitExceeded {
                requested: stats.allocated_vram,
                limit: (self.total_vram as f32 * 0.95) as usize,
            });
        }

        Ok(())
    }

    /// Generate detailed VRAM report
    pub fn generate_vram_report(&self) -> String {
        let stats = self.get_vram_stats();
        let mut report = String::new();

        report.push_str("VRAM Report:\n");
        report.push_str(&format!(
            "  Total VRAM: {:.2} MB\n",
            stats.total_vram as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!(
            "  Allocated: {:.2} MB\n",
            stats.allocated_vram as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!(
            "  Available: {:.2} MB\n",
            stats.available_vram as f64 / 1024.0 / 1024.0
        ));
        report.push_str(&format!(
            "  Utilization: {:.1}%\n",
            stats.utilization * 100.0
        ));
        report.push_str(&format!("  Buffer Count: {}\n", stats.buffer_count));

        // Buffer breakdown by type
        let mut type_counts: HashMap<crate::traits::BufferType, (usize, usize)> = HashMap::new();
        for buffer in self.buffers.values() {
            let entry = type_counts.entry(buffer.buffer_type).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += buffer.size;
        }

        report.push_str("\nBuffer Breakdown:\n");
        for (buffer_type, (count, size)) in type_counts {
            report.push_str(&format!(
                "  {:?}: {} buffers, {:.2} MB\n",
                buffer_type,
                count,
                size as f64 / 1024.0 / 1024.0
            ));
        }

        report
    }
}

impl GpuMonitor for VramMonitor {
    fn get_vram_stats(&self) -> VramStats {
        self.get_vram_stats()
    }

    fn validate_all_vram(&self) -> Result<()> {
        self.validate_all_vram()
    }

    fn generate_vram_report(&self) -> String {
        self.generate_vram_report()
    }
}

/// VRAM Validator for ensuring VRAM-only operations
pub struct VramValidator {
    monitor: VramMonitor,
    max_utilization: f32,
}

impl VramValidator {
    /// Create a new VRAM validator
    pub fn new(total_vram: usize, max_utilization: f32) -> Self {
        Self {
            monitor: VramMonitor::new(total_vram),
            max_utilization,
        }
    }

    /// Validate VRAM allocation request
    pub fn validate_allocation(&self, size: usize) -> Result<()> {
        let stats = self.monitor.get_vram_stats();
        let new_utilization = (stats.allocated_vram + size) as f32 / stats.total_vram as f32;

        if new_utilization > self.max_utilization {
            return Err(HiveGpuError::VramLimitExceeded {
                requested: stats.allocated_vram + size,
                limit: (stats.total_vram as f32 * self.max_utilization) as usize,
            });
        }

        Ok(())
    }
}

/// VRAM Benchmark Result
#[derive(Debug, Clone)]
pub struct VramBenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Memory allocated in bytes
    pub memory_allocated: usize,
    /// Memory utilization percentage
    pub utilization: f32,
    /// Throughput (operations per second)
    pub throughput: f64,
}
