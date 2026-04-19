//! # Intel (Vulkan) Vector Storage
//!
//! Stores vector payloads in a `HOST_VISIBLE | HOST_COHERENT` Vulkan
//! buffer. Brute-force search dispatches the `sgemv_dot` compute pipeline
//! built during context creation; metric post-processing and top-K run on
//! the host. Mirrors the shape of the CUDA, ROCm, and Metal backends.
//!
//! Memory strategy: for simplicity and because Intel GPUs (both
//! integrated Xe and Arc) handle coherent memory well, every buffer is
//! allocated as `HOST_VISIBLE | HOST_COHERENT` rather than staging
//! through device-local memory. A future optimisation can introduce a
//! staging + blit pattern matching the CUDA and ROCm backends if
//! benchmarks show it matters.
//!
//! ⚠️ AUTHORED BLIND — see phase3c_add-intel-backend.

#![cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))]

use super::context::{ComputePipeline, IntelContext, SgemvPushConstants};
use crate::error::{HiveGpuError, Result};
use crate::traits::GpuVectorStorage;
use crate::types::{GpuDistanceMetric, GpuSearchResult, GpuVector};
use ash::vk;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info};

const MIN_INITIAL_VECTORS: usize = 1024;
const MIN_INITIAL_BYTES: usize = 1024 * 1024;

/// Vulkan buffer + backing device memory, kept together so drop order
/// stays correct.
pub(crate) struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size_bytes: usize,
    /// Cached host-mapped pointer for `HOST_VISIBLE` buffers. Null when
    /// the buffer is device-local only.
    pub host_ptr: *mut u8,
}

// SAFETY: the raw pointer is only used through the device (Vulkan
// serialises it) and we never share it between threads without external
// synchronisation.
unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanBuffer {
    pub(crate) fn destroy(self, device: &ash::Device) {
        // SAFETY: both objects were created by us and no other thread is
        // accessing them (the caller holds exclusive ownership).
        unsafe {
            if !self.host_ptr.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Allocate a host-visible, host-coherent Vulkan buffer of at least
/// `size_bytes` bytes usable as a compute-shader storage buffer. Mapped
/// pointer is kept alive for the buffer's lifetime so `write_f32_slice`
/// and `read_f32_slice` can copy without remapping.
pub(crate) fn allocate_host_visible_buffer(
    context: &IntelContext,
    size_bytes: usize,
) -> Result<VulkanBuffer> {
    let size_bytes = size_bytes.max(1);
    let device = context.device();

    let buffer_info = vk::BufferCreateInfo::default()
        .size(size_bytes as u64)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: `buffer_info` scoped; `device` outlives the returned buffer
    // as long as the VulkanBuffer handle stays alive.
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| HiveGpuError::VulkanError(format!("create_buffer: {e:?}")))?;

    // SAFETY: buffer just created.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = pick_memory_type(
        context,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        HiveGpuError::VulkanError("no host-visible memory type available".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type_index);
    // SAFETY: alloc_info scoped.
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|e| {
        // SAFETY: buffer was created above; cleanup on failure.
        unsafe {
            device.destroy_buffer(buffer, None);
        }
        HiveGpuError::VulkanError(format!("allocate_memory: {e:?}"))
    })?;

    // SAFETY: buffer + memory just created; memory type selected to be
    // compatible with buffer.
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| HiveGpuError::VulkanError(format!("bind_buffer_memory: {e:?}")))?;
    }

    // Map for host access.
    // SAFETY: HOST_VISIBLE type selected; mapping the entire range is
    // supported for coherent memory.
    let host_ptr = unsafe {
        device
            .map_memory(memory, 0, reqs.size, vk::MemoryMapFlags::empty())
            .map_err(|e| HiveGpuError::VulkanError(format!("map_memory: {e:?}")))?
            as *mut u8
    };

    Ok(VulkanBuffer {
        buffer,
        memory,
        size_bytes: reqs.size as usize,
        host_ptr,
    })
}

fn pick_memory_type(
    context: &IntelContext,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let props = context.memory_properties();
    (0..props.memory_type_count).find(|i| {
        let ty = props.memory_types[*i as usize];
        (type_bits & (1 << i)) != 0 && ty.property_flags.contains(required)
    })
}

/// Write an f32 slice into the mapped region of a host-visible buffer
/// (overwrites from offset 0).
pub(crate) fn write_f32_slice(buf: &VulkanBuffer, data: &[f32]) -> Result<()> {
    if buf.host_ptr.is_null() {
        return Err(HiveGpuError::VulkanError(
            "write_f32_slice requires a host-visible buffer".to_string(),
        ));
    }
    let bytes = data.len() * std::mem::size_of::<f32>();
    if bytes > buf.size_bytes {
        return Err(HiveGpuError::VulkanError(format!(
            "write overflows buffer: {bytes} > {size}",
            size = buf.size_bytes
        )));
    }
    // SAFETY: `host_ptr` is mapped for at least `buf.size_bytes`; we bound
    // the write by `bytes`; HOST_COHERENT memory means no explicit flush
    // is required before the next queue submit reads the data.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.host_ptr, bytes);
    }
    Ok(())
}

/// Write a slice starting at a given byte offset inside the mapped region.
pub(crate) fn write_f32_slice_at(
    buf: &VulkanBuffer,
    byte_offset: usize,
    data: &[f32],
) -> Result<()> {
    if buf.host_ptr.is_null() {
        return Err(HiveGpuError::VulkanError(
            "write_f32_slice_at requires a host-visible buffer".to_string(),
        ));
    }
    let bytes = data.len() * std::mem::size_of::<f32>();
    if byte_offset + bytes > buf.size_bytes {
        return Err(HiveGpuError::VulkanError(format!(
            "offset write overflows buffer: {byte_offset}+{bytes} > {size}",
            size = buf.size_bytes
        )));
    }
    // SAFETY: bounds checked above.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.host_ptr.add(byte_offset),
            bytes,
        );
    }
    Ok(())
}

/// Read a contiguous prefix of a host-visible buffer into a Rust `Vec<f32>`.
pub(crate) fn read_f32_vec(buf: &VulkanBuffer, count: usize) -> Result<Vec<f32>> {
    if buf.host_ptr.is_null() {
        return Err(HiveGpuError::VulkanError(
            "read_f32_vec requires a host-visible buffer".to_string(),
        ));
    }
    let bytes = count * std::mem::size_of::<f32>();
    if bytes > buf.size_bytes {
        return Err(HiveGpuError::VulkanError(format!(
            "read overflows buffer: {bytes} > {size}",
            size = buf.size_bytes
        )));
    }
    let mut out = vec![0f32; count];
    // SAFETY: bounds checked.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.host_ptr as *const u8,
            out.as_mut_ptr() as *mut u8,
            bytes,
        );
    }
    Ok(out)
}

/// Device-to-device `vkCmdCopyBuffer` dispatched on a one-shot command
/// buffer. Used by storage growth and by anything that needs to move
/// bytes between two host-visible buffers without touching the host.
pub(crate) fn dtod_copy(
    context: &IntelContext,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    let device = context.device();
    let cb = begin_one_shot_command(context)?;
    let region = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: bytes as u64,
    }];
    // SAFETY: command buffer recorded inside the scope we just opened.
    unsafe {
        device.cmd_copy_buffer(cb, src.buffer, dst.buffer, &region);
    }
    end_and_submit_one_shot(context, cb)
}

/// Dispatch a compute pipeline over three storage buffers with a push
/// constants payload and a grid size. Allocates a one-shot descriptor
/// set + command buffer + fence; every operation waits to completion
/// before returning. Simple and slow; optimisation is a follow-up.
pub(crate) fn dispatch_three_buffer_compute<P: bytemuck_like::PodLike>(
    context: &IntelContext,
    pipeline: ComputePipeline,
    buffers: [&VulkanBuffer; 3],
    push_constants: P,
    grid: (u32, u32, u32),
) -> Result<()> {
    let device = context.device();

    // 1. Allocate a descriptor set from the context's pool.
    let layouts = [pipeline.set_layout];
    let ds_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(context.descriptor_pool())
        .set_layouts(&layouts);
    // SAFETY: layouts slice scoped; descriptor pool has FREE flag set.
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&ds_alloc) }
        .map_err(|e| HiveGpuError::VulkanError(format!("allocate_descriptor_sets: {e:?}")))?;
    let descriptor_set = descriptor_sets[0];

    // 2. Bind the three storage buffers.
    let buffer_infos: [vk::DescriptorBufferInfo; 3] = [
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[0].buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[1].buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE),
        vk::DescriptorBufferInfo::default()
            .buffer(buffers[2].buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE),
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[0])),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[1])),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[2])),
    ];
    // SAFETY: all slices scoped.
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    // 3. Record + submit + wait.
    let cb = begin_one_shot_command(context)?;
    // SAFETY: recording a one-shot command buffer we just allocated.
    unsafe {
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[descriptor_set],
            &[],
        );
        let pc_bytes = std::slice::from_raw_parts(
            &push_constants as *const P as *const u8,
            std::mem::size_of::<P>(),
        );
        device.cmd_push_constants(
            cb,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        device.cmd_dispatch(cb, grid.0, grid.1, grid.2);
    }
    end_and_submit_one_shot(context, cb)?;

    // 4. Free the descriptor set (pool was created with FREE_DESCRIPTOR_SET).
    // SAFETY: descriptor set allocated from this exact pool.
    unsafe {
        device
            .free_descriptor_sets(context.descriptor_pool(), &[descriptor_set])
            .map_err(|e| HiveGpuError::VulkanError(format!("free_descriptor_sets: {e:?}")))?;
    }

    Ok(())
}

fn begin_one_shot_command(context: &IntelContext) -> Result<vk::CommandBuffer> {
    let device = context.device();
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(context.command_pool())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: alloc scoped.
    let buffers = unsafe { device.allocate_command_buffers(&alloc) }
        .map_err(|e| HiveGpuError::VulkanError(format!("allocate_command_buffers: {e:?}")))?;
    let cb = buffers[0];
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: cb just allocated.
    unsafe { device.begin_command_buffer(cb, &begin) }
        .map_err(|e| HiveGpuError::VulkanError(format!("begin_command_buffer: {e:?}")))?;
    Ok(cb)
}

fn end_and_submit_one_shot(context: &IntelContext, cb: vk::CommandBuffer) -> Result<()> {
    let device = context.device();
    // SAFETY: cb is mid-recording.
    unsafe {
        device
            .end_command_buffer(cb)
            .map_err(|e| HiveGpuError::VulkanError(format!("end_command_buffer: {e:?}")))?;
    }

    // Create a fence to wait on.
    let fence_info = vk::FenceCreateInfo::default();
    // SAFETY: fence_info scoped.
    let fence = unsafe { device.create_fence(&fence_info, None) }
        .map_err(|e| HiveGpuError::VulkanError(format!("create_fence: {e:?}")))?;

    let cmd_buffers = [cb];
    let submit = [vk::SubmitInfo::default().command_buffers(&cmd_buffers)];
    // SAFETY: submit + cmd_buffers scoped; fence just created.
    unsafe {
        device
            .queue_submit(context.queue(), &submit, fence)
            .map_err(|e| HiveGpuError::VulkanError(format!("queue_submit: {e:?}")))?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| HiveGpuError::VulkanError(format!("wait_for_fences: {e:?}")))?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(context.command_pool(), &cmd_buffers);
    }
    Ok(())
}

/// Same as [`dispatch_three_buffer_compute`] but with an explicit byte
/// offset + range per binding, so callers can expose a sub-range of a
/// larger buffer. Used by the IVF index for per-cluster refined search.
pub(crate) fn dispatch_three_buffer_compute_ranged<P: bytemuck_like::PodLike>(
    context: &IntelContext,
    pipeline: ComputePipeline,
    bindings: [(&VulkanBuffer, u64, u64); 3],
    push_constants: P,
    grid: (u32, u32, u32),
) -> Result<()> {
    let device = context.device();

    let layouts = [pipeline.set_layout];
    let ds_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(context.descriptor_pool())
        .set_layouts(&layouts);
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&ds_alloc) }
        .map_err(|e| HiveGpuError::VulkanError(format!("allocate_descriptor_sets: {e:?}")))?;
    let descriptor_set = descriptor_sets[0];

    let buffer_infos: [vk::DescriptorBufferInfo; 3] = [
        vk::DescriptorBufferInfo::default()
            .buffer(bindings[0].0.buffer)
            .offset(bindings[0].1)
            .range(bindings[0].2),
        vk::DescriptorBufferInfo::default()
            .buffer(bindings[1].0.buffer)
            .offset(bindings[1].1)
            .range(bindings[1].2),
        vk::DescriptorBufferInfo::default()
            .buffer(bindings[2].0.buffer)
            .offset(bindings[2].1)
            .range(bindings[2].2),
    ];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[0])),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[1])),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_infos[2])),
    ];
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }

    let cb = begin_one_shot_command(context)?;
    unsafe {
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[descriptor_set],
            &[],
        );
        let pc_bytes = std::slice::from_raw_parts(
            &push_constants as *const P as *const u8,
            std::mem::size_of::<P>(),
        );
        device.cmd_push_constants(
            cb,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );
        device.cmd_dispatch(cb, grid.0, grid.1, grid.2);
    }
    end_and_submit_one_shot(context, cb)?;

    unsafe {
        device
            .free_descriptor_sets(context.descriptor_pool(), &[descriptor_set])
            .map_err(|e| HiveGpuError::VulkanError(format!("free_descriptor_sets: {e:?}")))?;
    }

    Ok(())
}

/// Tiny helper trait so the generic `dispatch_three_buffer_compute` can
/// accept arbitrary plain-old-data structs without pulling in the full
/// `bytemuck` dependency.
pub(crate) mod bytemuck_like {
    /// Marker for `#[repr(C)] Copy` payloads that can be safely sent as
    /// push constants via a raw pointer + size copy.
    pub trait PodLike: Copy {}
    impl<T: Copy> PodLike for T {}
}

// ---------------------------------------------------------------------
// Vector storage
// ---------------------------------------------------------------------

pub struct IntelVectorStorage {
    context: Arc<IntelContext>,
    storage: Option<VulkanBuffer>,
    buffer_capacity: usize,
    vector_count: usize,
    dimension: usize,
    metric: GpuDistanceMetric,
    vector_id_map: HashMap<String, usize>,
    index_to_id: Vec<String>,
    removed_indices: HashSet<usize>,
    payloads: HashMap<String, HashMap<String, String>>,
    norms_sq: Vec<f32>,
}

impl std::fmt::Debug for IntelVectorStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntelVectorStorage")
            .field("vector_count", &self.vector_count)
            .field("buffer_capacity", &self.buffer_capacity)
            .field("dimension", &self.dimension)
            .field("metric", &self.metric)
            .field("removed", &self.removed_indices.len())
            .finish()
    }
}

impl IntelVectorStorage {
    pub fn new(
        context: Arc<IntelContext>,
        dimension: usize,
        metric: GpuDistanceMetric,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(HiveGpuError::InvalidConfiguration(
                "dimension must be > 0".to_string(),
            ));
        }

        let min_vectors_by_bytes =
            (MIN_INITIAL_BYTES / (dimension * std::mem::size_of::<f32>())).max(1);
        let capacity = MIN_INITIAL_VECTORS.max(min_vectors_by_bytes);
        let slots = capacity
            .checked_mul(dimension)
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("capacity overflow".to_string()))?;
        let bytes = slots * std::mem::size_of::<f32>();

        let storage = allocate_host_visible_buffer(&context, bytes)?;

        debug!(
            "intel storage created: dim={} capacity={} bytes={}",
            dimension, capacity, bytes
        );

        Ok(Self {
            context,
            storage: Some(storage),
            buffer_capacity: capacity,
            vector_count: 0,
            dimension,
            metric,
            vector_id_map: HashMap::new(),
            index_to_id: Vec::new(),
            removed_indices: HashSet::new(),
            payloads: HashMap::new(),
            norms_sq: Vec::new(),
        })
    }

    fn validate_vector(&self, vector: &GpuVector) -> Result<()> {
        if vector.data.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.data.len(),
            });
        }
        if vector.id.is_empty() {
            return Err(HiveGpuError::InvalidConfiguration(
                "vector id must be non-empty".to_string(),
            ));
        }
        if vector.id.len() > 256 {
            return Err(HiveGpuError::InvalidConfiguration(
                "vector id must be <= 256 chars".to_string(),
            ));
        }
        if self.vector_id_map.contains_key(&vector.id) {
            return Err(HiveGpuError::InvalidConfiguration(format!(
                "duplicate vector id: {}",
                vector.id
            )));
        }
        for (i, &v) in vector.data.iter().enumerate() {
            if !v.is_finite() {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite component at index {i} in vector {}",
                    vector.id
                )));
            }
        }
        Ok(())
    }

    fn ensure_capacity(&mut self, additional: usize) -> Result<()> {
        let required = self
            .vector_count
            .checked_add(additional)
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("capacity overflow".to_string()))?;
        if required <= self.buffer_capacity {
            return Ok(());
        }
        let mut new_capacity = self.buffer_capacity;
        while new_capacity < required {
            let factor = if new_capacity < 1_000 {
                2.0f32
            } else if new_capacity < 10_000 {
                1.5f32
            } else {
                1.2f32
            };
            new_capacity = ((new_capacity as f32) * factor).ceil() as usize;
            new_capacity = new_capacity.max(required);
        }
        let slots = new_capacity
            .checked_mul(self.dimension)
            .ok_or_else(|| HiveGpuError::InvalidConfiguration("slots overflow".to_string()))?;
        let bytes = slots * std::mem::size_of::<f32>();

        let new_buf = allocate_host_visible_buffer(&self.context, bytes)?;
        if self.vector_count > 0 {
            let live_bytes = self.vector_count * self.dimension * std::mem::size_of::<f32>();
            let old = self.storage.as_ref().expect("storage live");
            dtod_copy(&self.context, old, &new_buf, live_bytes)?;
        }
        if let Some(old) = self.storage.take() {
            old.destroy(self.context.device());
        }
        info!(
            "intel storage expand: {} -> {} vectors ({:.2} MiB)",
            self.buffer_capacity,
            new_capacity,
            bytes as f64 / (1024.0 * 1024.0)
        );
        self.storage = Some(new_buf);
        self.buffer_capacity = new_capacity;
        Ok(())
    }

    /// Borrow the storage buffer (useful when the IVF index or downstream
    /// tooling needs to dispatch against a sub-range).
    #[allow(dead_code)] // exposed for future cross-module sharing
    pub(crate) fn storage_buffer(&self) -> &VulkanBuffer {
        self.storage.as_ref().expect("storage initialised")
    }

    #[allow(dead_code)] // exposed for future cross-module sharing
    pub(crate) fn norms_sq(&self) -> &[f32] {
        &self.norms_sq
    }

    /// Run the `sgemv_dot` pipeline against the full stored vector
    /// buffer. Returns one dot product per stored vector.
    pub(crate) fn gpu_scores(&self, query: &[f32]) -> Result<Vec<f32>> {
        if self.vector_count == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        for (i, &v) in query.iter().enumerate() {
            if !v.is_finite() {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "non-finite query component at index {i}"
                )));
            }
        }

        let query_bytes = query.len() * std::mem::size_of::<f32>();
        let query_buf = allocate_host_visible_buffer(&self.context, query_bytes)?;
        write_f32_slice(&query_buf, query)?;
        let scores_bytes = self.vector_count * std::mem::size_of::<f32>();
        let scores_buf = allocate_host_visible_buffer(&self.context, scores_bytes)?;

        let pipeline = self.context.sgemv_dot();
        let pc = SgemvPushConstants {
            dimension: self.dimension as u32,
            n_vectors: self.vector_count as u32,
        };
        let grid = (((self.vector_count as u32) + 255) / 256, 1, 1);
        let storage_buf = self.storage.as_ref().expect("storage live");
        dispatch_three_buffer_compute(
            &self.context,
            pipeline,
            [storage_buf, &query_buf, &scores_buf],
            pc,
            grid,
        )?;

        let scores = read_f32_vec(&scores_buf, self.vector_count)?;

        query_buf.destroy(self.context.device());
        scores_buf.destroy(self.context.device());
        Ok(scores)
    }

    fn apply_metric(&self, raw_scores: &mut [f32], query: &[f32]) {
        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        match self.metric {
            GpuDistanceMetric::DotProduct => {}
            GpuDistanceMetric::Cosine => {
                let q_norm = query_norm_sq.sqrt();
                for (i, s) in raw_scores.iter_mut().enumerate() {
                    let v_norm = self.norms_sq[i].sqrt();
                    let denom = v_norm * q_norm;
                    *s = if denom > 0.0 { *s / denom } else { 0.0 };
                }
            }
            GpuDistanceMetric::Euclidean => {
                for (i, s) in raw_scores.iter_mut().enumerate() {
                    *s = (self.norms_sq[i] - 2.0 * *s + query_norm_sq).max(0.0);
                }
            }
        }
    }

    fn select_top_k(&self, mut scored: Vec<(usize, f32)>, limit: usize) -> Vec<GpuSearchResult> {
        scored.retain(|(idx, _)| !self.removed_indices.contains(idx));
        match self.metric {
            GpuDistanceMetric::Euclidean => {
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            }
            _ => scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
        }
        scored.truncate(limit);
        scored
            .into_iter()
            .map(|(index, score)| {
                let id = self.index_to_id[index].clone();
                let similarity = match self.metric {
                    GpuDistanceMetric::Euclidean => 1.0 / (1.0 + score.sqrt()),
                    _ => score,
                };
                GpuSearchResult {
                    id,
                    score: similarity,
                    index,
                }
            })
            .collect()
    }
}

impl Drop for IntelVectorStorage {
    fn drop(&mut self) {
        if let Some(buf) = self.storage.take() {
            buf.destroy(self.context.device());
        }
    }
}

impl GpuVectorStorage for IntelVectorStorage {
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        let mut seen = HashSet::with_capacity(vectors.len());
        for v in vectors {
            self.validate_vector(v)?;
            if !seen.insert(v.id.as_str()) {
                return Err(HiveGpuError::InvalidConfiguration(format!(
                    "duplicate vector id within batch: {}",
                    v.id
                )));
            }
        }
        self.ensure_capacity(vectors.len())?;

        let mut flat = Vec::with_capacity(vectors.len() * self.dimension);
        for v in vectors {
            flat.extend_from_slice(&v.data);
        }
        let offset_bytes = self.vector_count * self.dimension * std::mem::size_of::<f32>();
        let storage = self.storage.as_ref().expect("storage live");
        write_f32_slice_at(storage, offset_bytes, &flat)?;

        let mut indices = Vec::with_capacity(vectors.len());
        for v in vectors {
            let index = self.vector_count;
            self.vector_id_map.insert(v.id.clone(), index);
            self.index_to_id.push(v.id.clone());
            self.payloads.insert(v.id.clone(), v.metadata.clone());
            self.norms_sq
                .push(v.data.iter().map(|&x| x * x).sum::<f32>());
            self.vector_count += 1;
            indices.push(index);
        }
        Ok(indices)
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
        if limit == 0 || self.vector_count == 0 {
            return Ok(Vec::new());
        }
        let mut scores = self.gpu_scores(query)?;
        self.apply_metric(&mut scores, query);
        let scored: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        Ok(self.select_top_k(scored, limit))
    }

    fn remove_vectors(&mut self, ids: &[String]) -> Result<()> {
        for id in ids {
            if let Some(&index) = self.vector_id_map.get(id) {
                self.removed_indices.insert(index);
                self.payloads.remove(id);
            } else {
                return Err(HiveGpuError::VectorNotFound(id.clone()));
            }
        }
        Ok(())
    }

    fn vector_count(&self) -> usize {
        self.vector_count.saturating_sub(self.removed_indices.len())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn get_vector(&self, id: &str) -> Result<Option<GpuVector>> {
        let Some(&index) = self.vector_id_map.get(id) else {
            return Ok(None);
        };
        if self.removed_indices.contains(&index) {
            return Ok(None);
        }
        let offset_bytes = index * self.dimension * std::mem::size_of::<f32>();
        let storage = self.storage.as_ref().expect("storage live");
        // Read straight from the mapped host pointer — same pattern as
        // `read_f32_vec` but starting at an offset.
        if storage.host_ptr.is_null() {
            return Err(HiveGpuError::VulkanError(
                "storage buffer not host-visible".to_string(),
            ));
        }
        let mut host = vec![0f32; self.dimension];
        // SAFETY: offset + dimension bounded by vector_count * dimension.
        unsafe {
            std::ptr::copy_nonoverlapping(
                storage.host_ptr.add(offset_bytes) as *const u8,
                host.as_mut_ptr() as *mut u8,
                self.dimension * std::mem::size_of::<f32>(),
            );
        }
        let metadata = self.payloads.get(id).cloned().unwrap_or_default();
        Ok(Some(GpuVector {
            id: id.to_string(),
            data: host,
            metadata,
        }))
    }

    fn clear(&mut self) -> Result<()> {
        self.vector_count = 0;
        self.buffer_capacity = self.buffer_capacity.max(MIN_INITIAL_VECTORS);
        self.vector_id_map.clear();
        self.index_to_id.clear();
        self.removed_indices.clear();
        self.payloads.clear();
        self.norms_sq.clear();
        Ok(())
    }
}
