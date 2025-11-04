# Migration Guide: metal-rs to objc2-metal

## Overview

This guide documents the migration from the discontinued `metal-rs` library to the actively maintained `objc2-metal` ecosystem. This migration was necessary for security, maintenance, and future-proofing.

## Why Migrate?

### Problems with metal-rs
- **Discontinued**: No longer maintained or updated
- **Security**: No security patches or vulnerability fixes
- **Type Safety**: Older Objective-C bindings with less type safety
- **API Gaps**: Missing newer Metal API features

### Benefits of objc2-metal
- **Actively Maintained**: Regular updates and security patches
- **Type Safe**: Modern Rust bindings with `ProtocolObject<dyn Trait>` pattern
- **Complete API**: Full coverage of Metal framework
- **Foundation Support**: Integrated with objc2-foundation for NSString, etc.

## Dependency Changes

### Before (metal-rs)
```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = { version = "0.27", optional = true }
objc = { version = "0.2", optional = true }

[features]
metal-native = ["metal", "objc"]
```

### After (objc2-metal)
```toml
[target.'cfg(target_os = "macos")'.dependencies]
objc2-metal = { version = "0.3", optional = true }
objc2-foundation = { version = "0.3", optional = true }
objc2 = { version = "0.6", optional = true }

[features]
metal-native = ["objc2-metal", "objc2-foundation", "objc2"]
```

## Key API Changes

### 1. Import Changes

**Before:**
```rust
use metal::{Device, CommandQueue, Library, Buffer, MTLSize};
```

**After:**
```rust
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLCommandQueue, MTLLibrary, MTLBuffer, MTLSize,
    MTLCreateSystemDefaultDevice,
};
```

### 2. Device Type Changes

**Before:**
```rust
struct Context {
    device: metal::Device,
}
```

**After:**
```rust
struct Context {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}
```

**Why**: Metal types are Objective-C protocols, not concrete types. `objc2` uses `ProtocolObject<dyn Trait>` wrapped in `Retained<>` for automatic reference counting.

### 3. Device Creation

**Before:**
```rust
let device = metal::Device::system_default()
    .ok_or_else(|| Error::NoDevice)?;
```

**After:**
```rust
let device = unsafe { MTLCreateSystemDefaultDevice() }
    .ok_or_else(|| Error::NoDevice)?;
```

**Note**: Device creation requires `unsafe` as it's a C FFI call.

### 4. Method Naming Convention

**Before (snake_case):**
```rust
device.new_buffer(size, options)
device.new_command_queue()
command_buffer.new_blit_command_encoder()
```

**After (camelCase):**
```rust
device.newBufferWithLength_options(size, options)
device.newCommandQueue()
command_buffer.blitCommandEncoder()
```

**Why**: `objc2` uses Objective-C naming conventions directly.

### 5. Buffer Creation

**Before:**
```rust
let buffer = device.new_buffer(
    1024,
    MTLResourceOptions::StorageModePrivate
)?;
```

**After:**
```rust
let buffer = device
    .newBufferWithLength_options(
        1024,
        MTLResourceOptions::StorageModePrivate
    )
    .ok_or_else(|| Error::BufferCreation)?;
```

### 6. Buffer Creation with Data

**Before:**
```rust
let buffer = device.new_buffer_with_data(
    data.as_ptr() as *const _,
    data.len() as u64,
    MTLResourceOptions::StorageModeShared
)?;
```

**After:**
```rust
use std::ptr::NonNull;

let buffer = unsafe {
    device.newBufferWithBytes_length_options(
        NonNull::new_unchecked(data.as_ptr() as *mut c_void),
        data.len(),
        MTLResourceOptions::StorageModeShared
    )
}.ok_or_else(|| Error::BufferCreation)?;
```

**Important**: 
- Size type changed from `u64` to `usize`
- Pointer must be wrapped in `NonNull<c_void>`
- Requires `unsafe` block

### 7. Command Buffer Operations

**Before:**
```rust
let command_buffer = queue.new_command_buffer();
let blit_encoder = command_buffer.new_blit_command_encoder();

blit_encoder.copy_from_buffer(
    &src_buffer,
    0,
    &dst_buffer,
    0,
    size
);

blit_encoder.end_encoding();
command_buffer.commit();
command_buffer.wait_until_completed();
```

**After:**
```rust
let command_buffer = queue.commandBuffer()
    .ok_or_else(|| Error::CommandBuffer)?;

let blit_encoder = command_buffer.blitCommandEncoder()
    .ok_or_else(|| Error::BlitEncoder)?;

unsafe {
    blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
        &src_buffer,
        0,
        &dst_buffer,
        0,
        size
    );
}

blit_encoder.endEncoding();
command_buffer.commit();
command_buffer.waitUntilCompleted();
```

### 8. Trait Imports for Methods

**Problem**: Method not found errors like:
```
error: no method named `endEncoding` found for struct `Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>`
```

**Solution**: Import the trait:
```rust
use objc2_metal::MTLCommandEncoder;  // For endEncoding()
use objc2_metal::MTLDevice;          // For device methods like name()
```

**Why**: `objc2` uses traits to provide methods on protocol objects. You must import the trait to access its methods.

### 9. Resource Options

**Before:**
```rust
MTLResourceOptions::StorageModePrivate
MTLResourceOptions::CPUCacheModeDefaultCache
```

**After:**
```rust
MTLResourceOptions::StorageModePrivate  // Still works
// Or use MTLStorageMode enum:
MTLStorageMode::Private
MTLStorageMode::Shared
```

**Note**: Both `MTLResourceOptions` and `MTLStorageMode` are available in objc2-metal.

### 10. String Conversion

**Before:**
```rust
device.name()  // Returns String directly
```

**After:**
```rust
use objc2_foundation::NSString;

let name = device.name();  // Returns &NSString
let name_string = name.to_string();  // Convert to String
```

**For shader source:**
```rust
use objc2_foundation::NSString;

let source = "...shader code...";
let ns_source = NSString::from_str(source);

let library = unsafe {
    device.newLibraryWithSource_options_error(&ns_source, Some(&options))
}.ok_or_else(|| Error::ShaderCompilation)?;
```

## Complete Migration Example

### Before (metal-rs)
```rust
use metal::{Device, CommandQueue, Buffer, MTLResourceOptions};

struct VectorStorage {
    device: metal::Device,
    queue: CommandQueue,
    buffer: Buffer,
}

impl VectorStorage {
    fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or(Error::NoDevice)?;
        
        let queue = device.new_command_queue();
        
        let buffer = device.new_buffer(
            1024,
            MTLResourceOptions::StorageModePrivate
        );
        
        Ok(Self { device, queue, buffer })
    }
    
    fn copy_data(&self, data: &[f32]) -> Result<()> {
        let staging = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_blit_command_encoder();
        
        encoder.copy_from_buffer(&staging, 0, &self.buffer, 0, data.len() * 4);
        encoder.end_encoding();
        
        cmd.commit();
        cmd.wait_until_completed();
        
        Ok(())
    }
}
```

### After (objc2-metal)
```rust
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLCommandQueue, MTLBuffer, MTLResourceOptions,
    MTLCreateSystemDefaultDevice, MTLBlitCommandEncoder, MTLCommandEncoder,
};
use std::ptr::NonNull;

struct VectorStorage {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl VectorStorage {
    fn new() -> Result<Self> {
        let device = unsafe { MTLCreateSystemDefaultDevice() }
            .ok_or(Error::NoDevice)?;
        
        let queue = device.newCommandQueue()
            .ok_or(Error::QueueCreation)?;
        
        let buffer = device.newBufferWithLength_options(
            1024,
            MTLResourceOptions::StorageModePrivate
        ).ok_or(Error::BufferCreation)?;
        
        Ok(Self { device, queue, buffer })
    }
    
    fn copy_data(&self, data: &[f32]) -> Result<()> {
        let size = data.len() * std::mem::size_of::<f32>();
        
        let staging = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(data.as_ptr() as *mut std::ffi::c_void),
                size,
                MTLResourceOptions::StorageModeShared
            )
        }.ok_or(Error::StagingBuffer)?;
        
        let cmd = self.queue.commandBuffer()
            .ok_or(Error::CommandBuffer)?;
        
        let encoder = cmd.blitCommandEncoder()
            .ok_or(Error::BlitEncoder)?;
        
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &staging,
                0,
                &self.buffer,
                0,
                size as u64
            );
        }
        
        encoder.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        
        Ok(())
    }
}
```

## Common Migration Pitfalls

### 1. Missing Trait Imports
**Error**: "no method named X found"
**Solution**: Import the corresponding trait (e.g., `MTLDevice`, `MTLCommandEncoder`)

### 2. Wrong Type Sizes
**Error**: "expected usize, found u64"
**Solution**: Cast sizes appropriately: `size as usize` or `size as u64`

### 3. Missing Unsafe Blocks
**Error**: "call to unsafe function requires unsafe block"
**Solution**: Wrap objc2 calls in `unsafe {}` blocks

### 4. Pointer Conversion
**Error**: "expected NonNull<c_void>, found *const _"
**Solution**: Use `NonNull::new_unchecked(ptr as *mut c_void)`

### 5. Return Type Changes
**Error**: Methods now return `Option<Retained<...>>` instead of direct types
**Solution**: Use `.ok_or_else()` or `?` to handle Options

## Testing Strategy

1. **Unit Tests**: Verify each module independently
2. **Integration Tests**: Test full workflows
3. **Performance Tests**: Ensure no regressions
4. **Device Compatibility**: Test on different Apple Silicon chips

## Rollback Plan

If migration fails:
```bash
git checkout pre-objc2-migration
```

## Resources

- [objc2 Documentation](https://docs.rs/objc2)
- [objc2-metal Documentation](https://docs.rs/objc2-metal)
- [objc2-foundation Documentation](https://docs.rs/objc2-foundation)
- [Metal API Reference](https://developer.apple.com/documentation/metal)

## Support

For issues or questions about this migration:
- Check project issues on GitHub
- Refer to OpenSpec change: `migrate-to-objc2-metal`
- Review CHANGELOG.md for migration notes
