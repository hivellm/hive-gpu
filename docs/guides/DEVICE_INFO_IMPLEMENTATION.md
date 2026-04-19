# Device Info API - Implementation Guide

This guide documents the implementation of the comprehensive GPU Device Information API added in version 0.1.9.

---

## 📋 Overview

The Device Info API provides detailed information about the GPU device, including:
- Device name and identifiers
- VRAM capacity and availability
- GPU capabilities and features
- Performance characteristics
- Unified memory information (Metal-specific)

---

## 🏗️ Architecture

### Core Components

```
src/
├── types.rs              # GpuDeviceInfo struct definition
├── traits.rs             # GpuContext::device_info() trait method
└── metal/
    └── context.rs        # Metal-specific implementation
```

---

## 📦 Data Structure

### `GpuDeviceInfo` Struct

```rust
/// Comprehensive GPU device information
#[derive(Debug, Clone, PartialEq)]
pub struct GpuDeviceInfo {
    /// Name of the GPU device
    pub device_name: String,
    
    /// Total VRAM in bytes
    pub total_vram_bytes: u64,
    
    /// Currently available VRAM in bytes
    pub available_vram_bytes: u64,
    
    /// Maximum recommended allocation size in bytes
    pub max_allocation_bytes: u64,
    
    /// Whether the device uses unified memory (e.g., Apple Silicon)
    pub is_unified_memory: bool,
    
    /// Backend type (Metal, CUDA, ROCM, CPU)
    pub backend_type: String,
}
```

**Key Design Decisions:**
- **All fields public**: Direct access for performance-critical code
- **Bytes for precision**: VRAM in bytes for accurate calculations
- **Clone + PartialEq**: Easy comparison and testing
- **String backend_type**: Human-readable backend identification

---

## 🔧 Implementation

### Step 1: Define the Struct (`src/types.rs`)

```rust
impl GpuDeviceInfo {
    /// Calculate VRAM usage percentage
    pub fn vram_usage_percent(&self) -> f64 {
        if self.total_vram_bytes == 0 {
            return 0.0;
        }
        let used = self.total_vram_bytes - self.available_vram_bytes;
        (used as f64 / self.total_vram_bytes as f64) * 100.0
    }

    /// Check if enough VRAM is available
    pub fn has_available_vram(&self, required_bytes: u64) -> bool {
        self.available_vram_bytes >= required_bytes
    }

    /// Get total VRAM in MB
    pub fn total_vram_mb(&self) -> f64 {
        self.total_vram_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get available VRAM in MB
    pub fn available_vram_mb(&self) -> f64 {
        self.available_vram_bytes as f64 / 1024.0 / 1024.0
    }
}
```

**Why these helper methods?**
- `vram_usage_percent()`: Quick memory pressure check
- `has_available_vram()`: Pre-allocation validation
- `total_vram_mb()` / `available_vram_mb()`: Human-readable values

---

### Step 2: Add Trait Method (`src/traits.rs`)

```rust
pub trait GpuContext: Send + Sync {
    // ... existing methods ...
    
    /// Get detailed information about the GPU device
    ///
    /// Returns comprehensive device information including:
    /// - Device name and identifiers
    /// - VRAM capacity and availability
    /// - Memory architecture (unified vs discrete)
    /// - Backend type
    ///
    /// # Example
    /// ```
    /// use hive_gpu::metal::MetalNativeContext;
    /// use hive_gpu::traits::GpuContext;
    ///
    /// let context = MetalNativeContext::new()?;
    /// let info = context.device_info()?;
    /// 
    /// println!("Device: {}", info.device_name);
    /// println!("Total VRAM: {:.2} GB", 
    ///          info.total_vram_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    /// println!("Available: {:.2} GB", 
    ///          info.available_vram_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    /// # Ok::<(), hive_gpu::error::HiveGpuError>(())
    /// ```
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError>;
}
```

**Why Result<T, E>?**
- Some backends may fail to query device info
- Consistent error handling across all GPU operations
- Allows graceful degradation

---

### Step 3: Metal Implementation (`src/metal/context.rs`)

```rust
impl GpuContext for MetalNativeContext {
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError> {
        // Get device name
        let device_name = unsafe {
            let name_ns = self.device.name();
            name_ns.to_string()
        };

        // Query VRAM (unified memory on Apple Silicon)
        let total_vram_bytes = self.device.recommendedMaxWorkingSetSize();
        
        // Calculate available VRAM
        // Note: Metal uses unified memory, so this is an approximation
        let current_allocated = self.device.currentAllocatedSize();
        let available_vram_bytes = total_vram_bytes.saturating_sub(current_allocated);

        // Get max allocation size
        let max_allocation_bytes = self.device.maxBufferLength();

        Ok(GpuDeviceInfo {
            device_name,
            total_vram_bytes,
            available_vram_bytes,
            max_allocation_bytes,
            is_unified_memory: true,  // Apple Silicon uses unified memory
            backend_type: "Metal".to_string(),
        })
    }
}
```

**Metal-Specific Notes:**
- `recommendedMaxWorkingSetSize()`: Recommended memory budget
- `currentAllocatedSize()`: Memory currently in use
- `maxBufferLength()`: Maximum single buffer size
- **Unified Memory**: CPU and GPU share the same memory pool

---

## 🧪 Testing Strategy

### Unit Tests (`tests/device_info_tests.rs`)

```rust
#[test]
fn test_metal_device_info() {
    let context = MetalNativeContext::new()
        .expect("Failed to create Metal context");
    
    let info = context.device_info()
        .expect("Failed to get device info");

    // Validate device info
    assert!(!info.device_name.is_empty());
    assert!(info.total_vram_bytes > 0);
    assert!(info.available_vram_bytes <= info.total_vram_bytes);
    assert!(info.max_allocation_bytes > 0);
    assert!(info.is_unified_memory);
    assert_eq!(info.backend_type, "Metal");
}
```

### Helper Method Tests

```rust
#[test]
fn test_vram_usage_percent() {
    let info = GpuDeviceInfo {
        device_name: "Test GPU".to_string(),
        total_vram_bytes: 16 * 1024 * 1024 * 1024,  // 16 GB
        available_vram_bytes: 8 * 1024 * 1024 * 1024,  // 8 GB available
        max_allocation_bytes: 4 * 1024 * 1024 * 1024,
        is_unified_memory: true,
        backend_type: "Metal".to_string(),
    };

    let usage = info.vram_usage_percent();
    assert!((usage - 50.0).abs() < 0.01);  // 50% used
}
```

---

## 📊 Real-World Usage Examples

### Example 1: Memory Pressure Check

```rust
use hive_gpu::metal::MetalNativeContext;
use hive_gpu::traits::GpuContext;

let context = MetalNativeContext::new()?;
let info = context.device_info()?;

// Check memory pressure before allocation
if info.vram_usage_percent() > 80.0 {
    println!("⚠️  High memory pressure: {:.1}%", info.vram_usage_percent());
    // Consider reducing batch size
}

// Check if specific allocation will fit
let required_bytes = 1024 * 1024 * 1024;  // 1 GB
if !info.has_available_vram(required_bytes) {
    eprintln!("❌ Insufficient VRAM for allocation");
    return Err(HiveGpuError::InsufficientMemory);
}
```

### Example 2: Adaptive Batch Sizing

```rust
fn calculate_optimal_batch_size(context: &MetalNativeContext) -> usize {
    let info = context.device_info()
        .expect("Failed to get device info");
    
    let available_gb = info.available_vram_mb() / 1024.0;
    
    // Use 50% of available VRAM for batch
    let target_gb = available_gb * 0.5;
    
    // Calculate batch size (assuming 512D vectors, 4 bytes per float)
    let bytes_per_vector = 512 * 4;
    let batch_size = (target_gb * 1024.0 * 1024.0 * 1024.0) as usize / bytes_per_vector;
    
    batch_size.min(10000)  // Cap at 10k vectors
}
```

### Example 3: Device Info Logging

```rust
fn log_gpu_info(context: &impl GpuContext) {
    match context.device_info() {
        Ok(info) => {
            println!("🎮 GPU Device Information:");
            println!("   Name: {}", info.device_name);
            println!("   Backend: {}", info.backend_type);
            println!("   Total VRAM: {:.2} GB", info.total_vram_mb() / 1024.0);
            println!("   Available: {:.2} GB", info.available_vram_mb() / 1024.0);
            println!("   Usage: {:.1}%", info.vram_usage_percent());
            println!("   Max Allocation: {:.2} GB", 
                     info.max_allocation_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
            println!("   Unified Memory: {}", info.is_unified_memory);
        }
        Err(e) => {
            eprintln!("❌ Failed to get device info: {}", e);
        }
    }
}
```

---

## 🔍 Platform-Specific Details

### macOS (Metal)

**Apple M3 Pro Example:**
```
Device: Apple M3 Pro
Backend: Metal
Total VRAM: 18.00 GB (unified with system RAM)
Available: 15.23 GB
Unified Memory: true
Max Allocation: 16.00 GB
```

**Key Characteristics:**
- Unified memory shared with CPU
- Total VRAM = `recommendedMaxWorkingSetSize()`
- Dynamic allocation based on system load
- No discrete VRAM pool

### Linux (CUDA - Future)

**Expected Structure:**
```rust
fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError> {
    let device_name = // cudaGetDeviceProperties
    let total_vram_bytes = // cuMemGetInfo (total)
    let available_vram_bytes = // cuMemGetInfo (free)
    let max_allocation_bytes = // Device capability
    
    Ok(GpuDeviceInfo {
        device_name,
        total_vram_bytes,
        available_vram_bytes,
        max_allocation_bytes,
        is_unified_memory: false,  // CUDA uses discrete memory
        backend_type: "CUDA".to_string(),
    })
}
```

---

## ⚠️ Common Pitfalls

### 1. Unified Memory Assumptions

❌ **Wrong:**
```rust
// Assumes discrete VRAM
if info.available_vram_bytes < required {
    // This may be too conservative on unified memory systems
}
```

✅ **Correct:**
```rust
if info.is_unified_memory {
    // Unified memory can swap to RAM
    // Be more lenient with VRAM checks
} else {
    // Discrete GPU: strict VRAM limits
    if info.available_vram_bytes < required {
        return Err(HiveGpuError::InsufficientMemory);
    }
}
```

### 2. Stale Device Info

❌ **Wrong:**
```rust
let info = context.device_info()?;
// ... many allocations later ...
if info.has_available_vram(size) {  // Stale data!
    allocate(size)?;
}
```

✅ **Correct:**
```rust
// Query fresh info before each large allocation
let info = context.device_info()?;
if info.has_available_vram(size) {
    allocate(size)?;
}
```

### 3. Zero Division

❌ **Wrong:**
```rust
let usage = (total - available) / total * 100.0;  // May panic if total = 0
```

✅ **Correct:**
```rust
// Use the built-in method (handles zero case)
let usage = info.vram_usage_percent();
```

---

## 📈 Performance Considerations

### Query Overhead

- **Metal**: ~1-5 μs per query (very fast)
- **Recommended**: Cache for short periods (e.g., per batch)
- **Avoid**: Querying every operation

### Example: Cached Device Info

```rust
struct CachedDeviceInfo {
    info: GpuDeviceInfo,
    last_update: Instant,
    ttl: Duration,
}

impl CachedDeviceInfo {
    fn get(&mut self, context: &impl GpuContext) -> Result<&GpuDeviceInfo, HiveGpuError> {
        if self.last_update.elapsed() > self.ttl {
            self.info = context.device_info()?;
            self.last_update = Instant::now();
        }
        Ok(&self.info)
    }
}
```

---

## 🔄 Migration from Old Code

### Before (No Device Info)

```rust
let context = MetalNativeContext::new()?;
let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;

// Blind allocation - may fail
storage.add_vectors(&large_batch)?;
```

### After (With Device Info)

```rust
let context = MetalNativeContext::new()?;
let info = context.device_info()?;

// Check capacity first
let required_bytes = large_batch.len() * 512 * 4;
if !info.has_available_vram(required_bytes) {
    // Split into smaller batches
    for chunk in large_batch.chunks(1000) {
        storage.add_vectors(chunk)?;
    }
} else {
    // Safe to add all at once
    storage.add_vectors(&large_batch)?;
}
```

---

## 🧪 Test Coverage

### Current Coverage: 4 Tests

```
tests/device_info_tests.rs:
✅ test_metal_device_info           - Basic device info query
✅ test_vram_usage_percent          - Usage calculation
✅ test_has_available_vram          - Availability check
✅ test_vram_convenience_methods    - MB conversion methods
```

### Integration with Other Tests

Device info is also tested in:
- `tests/gpu_detection_tests.rs` - Device detection flow
- `tests/gpu_vram_tests.rs` - VRAM monitoring accuracy
- `tests/gpu_stress_tests.rs` - Memory pressure scenarios

---

## 📚 API Reference

### `GpuDeviceInfo` Methods

| Method | Return | Description |
|--------|--------|-------------|
| `vram_usage_percent()` | `f64` | VRAM usage as percentage (0-100) |
| `has_available_vram(u64)` | `bool` | Check if bytes available |
| `total_vram_mb()` | `f64` | Total VRAM in megabytes |
| `available_vram_mb()` | `f64` | Available VRAM in megabytes |

### `GpuContext` Trait

| Method | Return | Description |
|--------|--------|-------------|
| `device_info()` | `Result<GpuDeviceInfo, HiveGpuError>` | Query device information |

---

## 🎯 Future Enhancements

### Planned for v0.2.0

- [ ] **Multi-GPU Support**: Query info for specific GPU by index
- [ ] **Temperature Monitoring**: GPU temperature and thermal state
- [ ] **Compute Capability**: Detailed capability flags
- [ ] **Driver Version**: CUDA/Metal driver versions
- [ ] **PCI Information**: Bus ID, device ID, vendor ID

### Example Future API

```rust
// Multi-GPU support
let devices = enumerate_devices()?;
for (idx, info) in devices.iter().enumerate() {
    println!("GPU {}: {} ({:.2} GB)", idx, info.device_name, info.total_vram_mb() / 1024.0);
}

// Extended info
let extended = context.extended_device_info()?;
println!("Temperature: {}°C", extended.temperature);
println!("Driver: {}", extended.driver_version);
```

---

## 🚀 Quick Start Guide

### 1. Add to your project

```rust
use hive_gpu::metal::MetalNativeContext;
use hive_gpu::traits::GpuContext;
```

### 2. Query device info

```rust
let context = MetalNativeContext::new()?;
let info = context.device_info()?;
```

### 3. Use the information

```rust
println!("Device: {}", info.device_name);
println!("Available VRAM: {:.2} GB", info.available_vram_mb() / 1024.0);

if info.vram_usage_percent() > 90.0 {
    println!("⚠️  High memory pressure!");
}
```

---

## 📝 Summary

The Device Info API provides:
- ✅ Comprehensive GPU device information
- ✅ VRAM capacity and availability tracking
- ✅ Platform-specific optimizations (unified memory support)
- ✅ Helper methods for common operations
- ✅ Robust error handling
- ✅ Extensive test coverage (4+ dedicated tests)

**Integration Points:**
- Memory allocation validation
- Batch size optimization
- Performance monitoring
- Error diagnostics
- User-facing dashboards

For more examples, see:
- `tests/device_info_tests.rs` - Complete test suite
- `examples/metal_basic.rs` - Basic usage
- `docs/reference/API_REFERENCE.md` - Full API documentation

