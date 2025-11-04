# Migration Guide: metal-rs to objc2-metal

## Overview

This guide documents the migration from the deprecated `metal-rs` crate to the modern `objc2-metal` framework for the Hive-GPU Metal backend implementation.

## Table of Contents

- [Background](#background)
- [Why Migrate](#why-migrate)
- [Key Changes](#key-changes)
- [Migration Steps](#migration-steps)
- [API Mapping](#api-mapping)
- [Common Patterns](#common-patterns)
- [Testing Strategy](#testing-strategy)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Background

### metal-rs (Deprecated)

The `metal-rs` crate provided Rust bindings for Apple's Metal framework but has been discontinued due to lack of maintenance in the `objc` 0.2 ecosystem.

```toml
# OLD (Deprecated)
metal = "0.27"
objc = "0.2"
```

### objc2-metal (Current)

The `objc2-metal` framework provides modern, actively maintained bindings built on `objc2`, offering better safety, performance, and alignment with current Rust practices.

```toml
# NEW (Recommended)
objc2-metal = "0.2"
objc2-foundation = "0.2"
objc2 = "0.5"
```

## Why Migrate

### Critical Reasons

1. **Security**: No security patches for metal-rs
2. **Maintenance**: No active development or bug fixes
3. **Features**: Missing access to newer Metal APIs
4. **Ecosystem**: Rust-objc community has moved to objc2
5. **Safety**: objc2 provides better type safety and lifetime management

### Benefits

- ✅ **Active Maintenance**: Regular updates with latest Metal features
- ✅ **Modern Rust**: Follows current Rust idioms and patterns
- ✅ **Better Safety**: Improved type-safe bindings
- ✅ **Performance**: Potential optimizations in newer bindings
- ✅ **Future-Proof**: Aligned with ecosystem direction
- ✅ **MPS Support**: Full access to Metal Performance Shaders (148K+ examples)

## Key Changes

### Dependency Changes

```diff
  [dependencies]
- metal = { version = "0.27", optional = true }
- objc = { version = "0.2", optional = true }
+ objc2-metal = { version = "0.2", optional = true }
+ objc2-foundation = { version = "0.2", optional = true }
+ objc2 = { version = "0.5", optional = true }
```

### Import Changes

```diff
- use metal::{Device, CommandQueue, Buffer, MTLSize};
- use objc::rc::StrongPtr;
+ use objc2_metal::{MTLDevice, MTLCommandQueue, MTLBuffer, MTLSize};
+ use objc2::rc::Retained;
+ use objc2_foundation::NSString;
```

### Naming Convention Changes

objc2-metal uses more explicit Objective-C naming:

- `Device` → `MTLDevice`
- `CommandQueue` → `MTLCommandQueue`
- `Buffer` → `MTLBuffer`
- `Library` → `MTLLibrary`
- Methods follow Objective-C conventions more closely

## Migration Steps

### Step 1: Update Dependencies

Update `Cargo.toml`:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
objc2-metal = { version = "0.2", optional = true }
objc2-foundation = { version = "0.2", optional = true }
objc2 = { version = "0.5", optional = true }

[features]
metal-native = ["objc2-metal", "objc2-foundation", "objc2"]
```

### Step 2: Update Imports

Replace all metal-rs imports with objc2-metal equivalents:

```rust
// Before
use metal::{Device, CommandQueue, Buffer, MTLResourceOptions};

// After
use objc2_metal::{MTLDevice, MTLCommandQueue, MTLBuffer, MTLResourceOptions};
```

### Step 3: Update Type Names

Rename types to follow objc2-metal conventions:

```rust
// Before
let device: Device = ...;
let queue: CommandQueue = ...;

// After
let device: &MTLDevice = ...;
let queue: &MTLCommandQueue = ...;
```

### Step 4: Update Method Calls

Method names follow Objective-C conventions more closely:

```rust
// Before (metal-rs)
let device = metal::Device::system_default()?;
let queue = device.new_command_queue();

// After (objc2-metal)
let device = MTLCreateSystemDefaultDevice().ok_or(...)?;
let queue = device.newCommandQueue();
```

### Step 5: Update Buffer Creation

Buffer creation uses different patterns:

```rust
// Before (metal-rs)
let buffer = device.new_buffer(
    size,
    metal::MTLResourceOptions::StorageModePrivate
);

// After (objc2-metal)
let buffer = device.newBufferWithLength_options(
    size,
    MTLResourceOptions::StorageModePrivate
);
```

### Step 6: Update String Handling

Use `NSString` from objc2-foundation:

```rust
// Before (metal-rs)
let shader_source = "...metal shader code...";

// After (objc2-metal)
use objc2_foundation::NSString;
let shader_source = NSString::from_str("...metal shader code...");
```

### Step 7: Update Library Compilation

Shader compilation uses new APIs:

```rust
// Before (metal-rs)
let library = device
    .new_library_with_source(shader_source, &options)?;

// After (objc2-metal)
let library = device
    .newLibraryWithSource_options_error(&shader_source, None)?;
```

## API Mapping

### Device Creation

```rust
// metal-rs
let device = metal::Device::system_default()
    .ok_or(HiveGpuError::NoDeviceAvailable)?;

// objc2-metal
use objc2_metal::MTLCreateSystemDefaultDevice;
let device = MTLCreateSystemDefaultDevice()
    .ok_or(HiveGpuError::NoDeviceAvailable)?;
```

### Command Queue Creation

```rust
// metal-rs
let queue = device.new_command_queue();

// objc2-metal
let queue = device.newCommandQueue();
```

### Buffer Creation

```rust
// metal-rs
let buffer = device.new_buffer(
    size as u64,
    MTLResourceOptions::StorageModePrivate
);

// objc2-metal
let buffer = device.newBufferWithLength_options(
    size as u64,
    MTLResourceOptions::StorageModePrivate
);
```

### Library Compilation

```rust
// metal-rs
let options = metal::CompileOptions::new();
let library = device
    .new_library_with_source(shader_source, &options)
    .map_err(|e| HiveGpuError::ShaderCompilationFailed(...))?;

// objc2-metal
use objc2_metal::MTLCompileOptions;
let options = MTLCompileOptions::new();
let library = device
    .newLibraryWithSource_options_error(&shader_source, Some(&options))
    .map_err(|e| HiveGpuError::ShaderCompilationFailed(...))?;
```

### GPU Family Checks

```rust
// metal-rs
device.supports_family(MTLGPUFamily::Apple7)

// objc2-metal (same)
device.supportsFamily(MTLGPUFamily::Apple7)
```

### VRAM Queries

```rust
// metal-rs
let max_vram = device.recommended_max_working_set_size();
let used_vram = device.current_allocated_size();

// objc2-metal
let max_vram = device.recommendedMaxWorkingSetSize();
let used_vram = device.currentAllocatedSize();
```

## Common Patterns

### Pattern 1: Device Initialization

```rust
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
use std::sync::Arc;

pub struct MetalContext {
    device: Retained<MTLDevice>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or(HiveGpuError::NoDeviceAvailable)?;
        
        Ok(Self { device })
    }
    
    pub fn device(&self) -> &MTLDevice {
        &self.device
    }
}
```

### Pattern 2: Buffer Allocation with Staging

```rust
use objc2_metal::{MTLBuffer, MTLResourceOptions};

fn allocate_buffer(
    device: &MTLDevice,
    data: &[f32],
) -> Result<Retained<MTLBuffer>> {
    let size = data.len() * std::mem::size_of::<f32>();
    
    // Create staging buffer
    let staging = device.newBufferWithBytes_length_options(
        data.as_ptr() as *const _,
        size as u64,
        MTLResourceOptions::StorageModeShared
    );
    
    // Create GPU-only buffer
    let gpu_buffer = device.newBufferWithLength_options(
        size as u64,
        MTLResourceOptions::StorageModePrivate
    );
    
    // Copy via command buffer
    let queue = device.newCommandQueue();
    let cmd_buffer = queue.commandBuffer();
    let blit = cmd_buffer.blitCommandEncoder();
    
    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
        &staging,
        0,
        &gpu_buffer,
        0,
        size as u64
    );
    
    blit.endEncoding();
    cmd_buffer.commit();
    cmd_buffer.waitUntilCompleted();
    
    Ok(gpu_buffer)
}
```

### Pattern 3: Compute Pipeline Creation

```rust
use objc2_metal::{MTLComputePipelineDescriptor, MTLFunction};
use objc2_foundation::NSString;

fn create_compute_pipeline(
    device: &MTLDevice,
    library: &MTLLibrary,
    function_name: &str,
) -> Result<Retained<MTLComputePipelineState>> {
    let name = NSString::from_str(function_name);
    let function = library.newFunctionWithName(&name)
        .ok_or(HiveGpuError::ShaderNotFound)?;
    
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| HiveGpuError::PipelineCreationFailed(...))?;
    
    Ok(pipeline)
}
```

### Pattern 4: Safe Command Encoding

```rust
fn dispatch_compute(
    queue: &MTLCommandQueue,
    pipeline: &MTLComputePipelineState,
    buffers: &[&MTLBuffer],
    grid_size: MTLSize,
) -> Result<()> {
    let cmd_buffer = queue.commandBuffer();
    let encoder = cmd_buffer.computeCommandEncoder();
    
    encoder.setComputePipelineState(pipeline);
    
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.setBuffer_offset_atIndex(buffer, 0, index as u64);
    }
    
    let threadgroup_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    
    encoder.dispatchThreads_threadsPerThreadgroup(
        grid_size,
        threadgroup_size
    );
    
    encoder.endEncoding();
    cmd_buffer.commit();
    cmd_buffer.waitUntilCompleted();
    
    Ok(())
}
```

## Testing Strategy

### Unit Tests

Test individual components with objc2-metal:

```rust
#[cfg(all(test, target_os = "macos", feature = "metal-native"))]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;
    
    #[test]
    fn test_device_creation() {
        let device = MTLCreateSystemDefaultDevice();
        assert!(device.is_some());
    }
    
    #[test]
    fn test_buffer_allocation() {
        let device = MTLCreateSystemDefaultDevice().unwrap();
        let buffer = device.newBufferWithLength_options(
            1024,
            MTLResourceOptions::StorageModePrivate
        );
        assert_eq!(buffer.length(), 1024);
    }
}
```

### Integration Tests

Validate end-to-end workflows:

```rust
#[test]
fn test_vector_storage_with_objc2() {
    let context = MetalNativeContext::new().unwrap();
    let storage = MetalNativeVectorStorage::new(
        Arc::new(context),
        128,
        GpuDistanceMetric::Cosine
    ).unwrap();
    
    // Test insert
    let vector = vec![1.0f32; 128];
    storage.insert_vector("test", &vector, None).unwrap();
    
    // Test search
    let results = storage.search(&vector, 10).unwrap();
    assert!(!results.is_empty());
}
```

### Performance Tests

Benchmark to ensure no regression:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_buffer_allocation(c: &mut Criterion) {
    let device = MTLCreateSystemDefaultDevice().unwrap();
    
    c.bench_function("buffer_alloc_objc2", |b| {
        b.iter(|| {
            device.newBufferWithLength_options(
                black_box(1024 * 1024),
                MTLResourceOptions::StorageModePrivate
            )
        });
    });
}

criterion_group!(benches, bench_buffer_allocation);
criterion_main!(benches);
```

## Performance Considerations

### Memory Management

objc2 uses `Retained<T>` for reference counting:

- More explicit than metal-rs's `StrongPtr`
- Better lifetime tracking
- No performance overhead vs metal-rs

### Buffer Operations

- Same underlying Metal APIs
- No performance difference expected
- May benefit from objc2 optimizations

### Command Encoding

- Identical dispatch patterns
- Same GPU execution
- Potential for better CPU-side performance

## Troubleshooting

### Common Issues

#### Issue: Method Not Found

```
error: no method named `new_buffer` found
```

**Solution**: Use camelCase Objective-C naming:
```rust
// Wrong
device.new_buffer(size, options)

// Correct
device.newBufferWithLength_options(size, options)
```

#### Issue: Type Mismatch

```
error: expected `&MTLDevice`, found `Device`
```

**Solution**: Update type names to MTL-prefixed versions:
```rust
// Wrong
let device: Device = ...;

// Correct
let device: Retained<MTLDevice> = ...;
```

#### Issue: String Conversion

```
error: expected `&NSString`, found `&str`
```

**Solution**: Convert strings explicitly:
```rust
use objc2_foundation::NSString;
let ns_str = NSString::from_str("my string");
```

#### Issue: Missing Import

```
error: cannot find type `MTLDevice` in this scope
```

**Solution**: Import from objc2-metal:
```rust
use objc2_metal::{MTLDevice, MTLCreateSystemDefaultDevice};
```

### Debugging Tips

1. **Check API Documentation**: Use `cargo doc --open` to see objc2-metal docs
2. **Enable Verbose Logging**: Set `RUST_LOG=debug` to see Metal calls
3. **Validate Metal Setup**: Use Metal debugger in Xcode
4. **Compare with Examples**: Check objc2-metal repository examples
5. **Test Incrementally**: Migrate and test one module at a time

## Reference Documentation

### Official Documentation

- [objc2 Documentation](https://docs.rs/objc2/)
- [objc2-metal Documentation](https://docs.rs/objc2-metal/)
- [objc2-foundation Documentation](https://docs.rs/objc2-foundation/)
- [Apple Metal Documentation](https://developer.apple.com/metal/)

### Community Resources

- [objc2 GitHub Repository](https://github.com/madsmtm/objc2)
- [Context7 objc2 Examples](https://context7.com/madsmtm/objc2)
- [Rust GPU Programming Guide](https://rust-gpu.github.io/)

### Internal Documentation

- `docs/ARCHITECTURE.md` - System architecture
- `docs/API_REFERENCE.md` - API documentation
- `openspec/changes/migrate-to-objc2-metal/` - Migration spec

## Conclusion

The migration from metal-rs to objc2-metal provides:

- ✅ **Better Safety**: Type-safe, modern Rust patterns
- ✅ **Active Maintenance**: Regular updates and bug fixes
- ✅ **Future-Proof**: Aligned with Rust-objc ecosystem
- ✅ **Performance Parity**: No regression expected
- ✅ **Improved Developer Experience**: Better documentation and tooling

For questions or issues during migration, refer to:
- This migration guide
- objc2-metal documentation  
- Metal backend specification in `openspec/`
- Community resources and examples

---

**Last Updated**: 2025-01-07  
**Migration Version**: v0.1.7 → v0.1.8 (or v0.2.0)  
**Status**: Ready for implementation

