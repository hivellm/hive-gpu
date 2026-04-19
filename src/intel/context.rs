//! # Intel (Vulkan Compute) Context
//!
//! Real Vulkan instance + compute device + queue + pre-compiled compute
//! pipelines for the `sgemv_dot` and `sgemm_dot` kernels that the vector
//! storage and IVF index dispatch. Mirrors the shape of
//! [`crate::cuda::CudaContext`] and [`crate::rocm::RocmContext`] so
//! downstream code stays portable.
//!
//! Target gating: this module only builds with `feature = "intel"` and
//! `target_os` in `{linux, windows}`. macOS is intentionally excluded —
//! users on Apple Silicon should use the Metal backend.
//!
//! ⚠️ AUTHORED BLIND — no Intel Arc / Battlemage hardware was available at
//! write time. See `phase3c_add-intel-backend`. The Vulkan loader is
//! dlopen'd by `ash` at runtime, so cross-platform `cargo check` passes
//! without any Vulkan SDK installed, but every Intel code path needs a
//! real Vulkan-capable maintainer to run the integration tests.

#![cfg(all(feature = "intel", any(target_os = "linux", target_os = "windows")))]

use crate::error::{HiveGpuError, Result};
use crate::traits::{GpuBackend, GpuContext};
use crate::types::{GpuCapabilities, GpuDeviceInfo, GpuMemoryStats};
use ash::vk;
use std::ffi::{CStr, CString};
use std::sync::Arc;
use tracing::debug;

/// Intel vendor ID. Any Vulkan physical device reports this in
/// [`vk::PhysicalDeviceProperties::vendor_id`] when it is an Intel GPU.
pub const INTEL_VENDOR_ID: u32 = 0x8086;

/// Set this env var to a non-empty value to relax the vendor filter and
/// accept any Vulkan-capable GPU. Useful for Docker images that lack a
/// vendor SDK but have a working Vulkan driver.
pub const UNIVERSAL_ENV: &str = "HIVE_GPU_VULKAN_UNIVERSAL";

/// Pre-compiled SPIR-V blobs produced by `build.rs` from
/// `src/intel/shaders/*.comp` at build time.
const SGEMV_DOT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sgemv_dot.spv"));
const SGEMM_DOT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sgemm_dot.spv"));

/// Owning handle for the Vulkan compute context. Construct once per
/// process and share via `Arc`; every operation borrows the shared state.
pub struct IntelContext {
    // Field order matters: Vulkan objects must drop in reverse creation
    // order. We rely on Rust's field-drop order to enforce that.
    sgemm_dot_pipeline: vk::Pipeline,
    sgemv_dot_pipeline: vk::Pipeline,
    sgemm_dot_layout: vk::PipelineLayout,
    sgemv_dot_layout: vk::PipelineLayout,
    sgemm_dot_module: vk::ShaderModule,
    sgemv_dot_module: vk::ShaderModule,
    sgemm_dot_set_layout: vk::DescriptorSetLayout,
    sgemv_dot_set_layout: vk::DescriptorSetLayout,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,

    queue: vk::Queue,
    queue_family_index: u32,
    device: ash::Device,

    #[allow(dead_code)] // physical device handle retained for Drop ordering
    physical_device: vk::PhysicalDevice,
    instance: ash::Instance,
    #[allow(dead_code)] // Entry keeps the Vulkan loader alive for the process
    entry: ash::Entry,

    device_name: String,
    vendor_id: u32,
    device_id: u32,
    api_version: u32,
    driver_version: u32,
    #[allow(dead_code)] // kept for telemetry / future device-class logic
    device_type: vk::PhysicalDeviceType,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
}

// SAFETY: Vulkan handles are plain pointers that are safe to share across
// threads as long as no two threads submit to the same queue concurrently.
// Downstream code serialises through the queue, matching the CUDA / ROCm
// / Metal backends.
unsafe impl Send for IntelContext {}
unsafe impl Sync for IntelContext {}

impl std::fmt::Debug for IntelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntelContext")
            .field("device_name", &self.device_name)
            .field("vendor_id", &format_args!("0x{:04x}", self.vendor_id))
            .field("device_id", &format_args!("0x{:04x}", self.device_id))
            .field("api_version", &format_args!("0x{:x}", self.api_version))
            .finish()
    }
}

impl IntelContext {
    /// Create a context on the first suitable Vulkan physical device.
    /// Returns the device whose vendor id is Intel's (`0x8086`), unless
    /// [`UNIVERSAL_ENV`] is set in the environment — in which case the
    /// first Vulkan-capable device is accepted regardless of vendor.
    pub fn new() -> Result<Arc<Self>> {
        let universal = std::env::var(UNIVERSAL_ENV)
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        Self::new_with_preference(universal)
    }

    /// `universal = true` accepts any Vulkan device; `false` restricts to
    /// Intel (vendor id `0x8086`).
    pub fn new_with_preference(universal: bool) -> Result<Arc<Self>> {
        // SAFETY: `Entry::load()` is documented as safe to call multiple
        // times; each call performs its own dlopen of the Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            HiveGpuError::IntelError(format!("failed to load Vulkan loader: {e:?}"))
        })?;

        let app_name = CString::new("hive-gpu").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 2, 0))
            .engine_name(&app_name)
            .engine_version(vk::make_api_version(0, 0, 2, 0))
            .api_version(vk::API_VERSION_1_2);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        // SAFETY: `create_info` borrows `app_info` + `app_name` which live
        // until end of scope; no extensions / layers requested.
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| HiveGpuError::VulkanError(format!("create_instance: {e:?}")))?;

        let (physical_device, queue_family_index, props) =
            select_physical_device(&instance, universal)?;

        // Create a logical device with a single compute queue.
        let queue_priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];
        let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        // SAFETY: `device_create_info` borrows `queue_infos` that lives to
        // end of scope.
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(|e| HiveGpuError::VulkanError(format!("create_device: {e:?}")))?;
        // SAFETY: we just created the queue family.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Command pool + descriptor pool. Pools are sized conservatively —
        // every per-search allocation below resets the pool before reusing
        // its slots.
        let cp_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `device` is live; `cp_info` scoped.
        let command_pool = unsafe { device.create_command_pool(&cp_info, None) }
            .map_err(|e| HiveGpuError::VulkanError(format!("create_command_pool: {e:?}")))?;

        let dp_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4096, // plenty for many concurrent dispatches
        }];
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1024)
            .pool_sizes(&dp_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        // SAFETY: pool sizes scoped, device live.
        let descriptor_pool = unsafe { device.create_descriptor_pool(&dp_info, None) }
            .map_err(|e| HiveGpuError::VulkanError(format!("create_descriptor_pool: {e:?}")))?;

        // Build the two compute pipelines.
        let (sgemv_dot_module, sgemv_dot_set_layout, sgemv_dot_layout, sgemv_dot_pipeline) =
            build_pipeline(
                &device,
                SGEMV_DOT_SPIRV,
                3, // matrix, query, scores
                std::mem::size_of::<SgemvPushConstants>() as u32,
            )?;
        let (sgemm_dot_module, sgemm_dot_set_layout, sgemm_dot_layout, sgemm_dot_pipeline) =
            build_pipeline(
                &device,
                SGEMM_DOT_SPIRV,
                3, // samples, centroids, out
                std::mem::size_of::<SgemmPushConstants>() as u32,
            )?;

        // Collect metadata once at construction — avoids re-querying on
        // every device_info() call.
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        // SAFETY: `instance` outlives the returned properties view.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        debug!(
            "intel context ready: device={:?} vendor=0x{:04x} id=0x{:04x}",
            device_name, props.vendor_id, props.device_id
        );

        Ok(Arc::new(Self {
            sgemm_dot_pipeline,
            sgemv_dot_pipeline,
            sgemm_dot_layout,
            sgemv_dot_layout,
            sgemm_dot_module,
            sgemv_dot_module,
            sgemm_dot_set_layout,
            sgemv_dot_set_layout,
            command_pool,
            descriptor_pool,
            queue,
            queue_family_index,
            device,
            physical_device,
            instance,
            entry,
            device_name,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            api_version: props.api_version,
            driver_version: props.driver_version,
            device_type: props.device_type,
            memory_properties,
            limits: props.limits,
        }))
    }

    /// Non-failing availability probe used by the backend detector.
    pub fn is_available() -> bool {
        let Ok(entry) = (unsafe { ash::Entry::load() }) else {
            return false;
        };
        let app_name = CString::new("hive-gpu-probe").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_2);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: temporary instance used only for enumeration; it is
        // destroyed on the way out of this function.
        let instance = match unsafe { entry.create_instance(&create_info, None) } {
            Ok(i) => i,
            Err(_) => return false,
        };
        let universal = std::env::var(UNIVERSAL_ENV)
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let found = enumerate_matching_devices(&instance, universal).is_some();
        // SAFETY: created above; no other borrow outstanding.
        unsafe { instance.destroy_instance(None) };
        found
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    // ---- accessors used by vector_storage / ivf -------------------------

    pub(crate) fn device(&self) -> &ash::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> vk::Queue {
        self.queue
    }

    #[allow(dead_code)] // exposed for future per-family queue work
    pub(crate) fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub(crate) fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub(crate) fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }

    pub(crate) fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    pub(crate) fn sgemv_dot(&self) -> ComputePipeline {
        ComputePipeline {
            pipeline: self.sgemv_dot_pipeline,
            layout: self.sgemv_dot_layout,
            set_layout: self.sgemv_dot_set_layout,
        }
    }

    pub(crate) fn sgemm_dot(&self) -> ComputePipeline {
        ComputePipeline {
            pipeline: self.sgemm_dot_pipeline,
            layout: self.sgemm_dot_layout,
            set_layout: self.sgemm_dot_set_layout,
        }
    }
}

/// Opaque handle pairing a pipeline with the layouts needed to dispatch
/// it. Plain `Copy` trio — the underlying Vulkan objects are owned by the
/// `IntelContext` and must outlive any pipeline view the caller holds.
#[derive(Clone, Copy)]
pub(crate) struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}

/// Push-constant payload for the `sgemv_dot` shader.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SgemvPushConstants {
    pub dimension: u32,
    pub n_vectors: u32,
}

/// Push-constant payload for the `sgemm_dot` shader.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SgemmPushConstants {
    pub dimension: u32,
    pub n_list: u32,
    pub n_samples: u32,
}

impl Drop for IntelContext {
    fn drop(&mut self) {
        // SAFETY: The device is still live at this point (we have
        // exclusive access via `&mut self`). Destruction order matches the
        // reverse of creation.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_pipeline(self.sgemm_dot_pipeline, None);
            self.device.destroy_pipeline(self.sgemv_dot_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.sgemm_dot_layout, None);
            self.device
                .destroy_pipeline_layout(self.sgemv_dot_layout, None);
            self.device
                .destroy_shader_module(self.sgemm_dot_module, None);
            self.device
                .destroy_shader_module(self.sgemv_dot_module, None);
            self.device
                .destroy_descriptor_set_layout(self.sgemm_dot_set_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.sgemv_dot_set_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---- helpers --------------------------------------------------------

/// Walk every physical device and return the first one whose vendor id is
/// Intel's (or any, if `universal`). Returns the device, its compute
/// queue family, and its properties.
fn select_physical_device(
    instance: &ash::Instance,
    universal: bool,
) -> Result<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> {
    enumerate_matching_devices(instance, universal).ok_or_else(|| {
        if universal {
            HiveGpuError::NoDeviceAvailable
        } else {
            HiveGpuError::IntelError(
                "no Intel GPU found; set HIVE_GPU_VULKAN_UNIVERSAL=1 to accept any vendor"
                    .to_string(),
            )
        }
    })
}

fn enumerate_matching_devices(
    instance: &ash::Instance,
    universal: bool,
) -> Option<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> {
    // SAFETY: `instance` is live.
    let devices = unsafe { instance.enumerate_physical_devices() }.ok()?;
    let mut fallback: Option<(vk::PhysicalDevice, u32, vk::PhysicalDeviceProperties)> = None;

    for pd in devices {
        // SAFETY: `instance` is live; `pd` was just enumerated.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if !universal && props.vendor_id != INTEL_VENDOR_ID {
            continue;
        }
        // SAFETY: same live instance.
        let qfp = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let qfi = qfp.iter().enumerate().find_map(|(idx, qfp)| {
            if qfp.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                Some(idx as u32)
            } else {
                None
            }
        });
        let Some(qfi) = qfi else {
            continue;
        };

        // Prefer discrete GPUs; fall back to integrated if that is all we
        // see.
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            return Some((pd, qfi, props));
        }
        if fallback.is_none() {
            fallback = Some((pd, qfi, props));
        }
    }
    fallback
}

/// Compile a SPIR-V blob, build a descriptor-set layout with `n_bindings`
/// storage-buffer entries, and create the matching compute pipeline.
/// Returns every Vulkan object the caller must drop in reverse order.
fn build_pipeline(
    device: &ash::Device,
    spirv: &[u8],
    n_bindings: u32,
    push_constant_size: u32,
) -> Result<(
    vk::ShaderModule,
    vk::DescriptorSetLayout,
    vk::PipelineLayout,
    vk::Pipeline,
)> {
    // SPIR-V is declared as `u32[]` per Vulkan spec — check alignment.
    if spirv.len() % 4 != 0 {
        return Err(HiveGpuError::SpirvCompileError(
            "shader SPIR-V size not a multiple of 4 bytes".to_string(),
        ));
    }
    // SAFETY: slice contains a multiple of 4 bytes; aligning u8 → u32
    // is the idiomatic pattern for SPIR-V modules.
    let code: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
    // SAFETY: module_info borrows `code` that lives to end of scope.
    let module = unsafe { device.create_shader_module(&module_info, None) }
        .map_err(|e| HiveGpuError::SpirvCompileError(format!("create_shader_module: {e:?}")))?;

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: bindings live to end of scope.
    let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
        .map_err(|e| HiveGpuError::VulkanError(format!("create_descriptor_set_layout: {e:?}")))?;

    let push_constant_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constant_size)];
    let set_layouts = [set_layout];
    let pl_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_range);
    // SAFETY: all slices live to end of scope.
    let layout = unsafe { device.create_pipeline_layout(&pl_info, None) }
        .map_err(|e| HiveGpuError::VulkanError(format!("create_pipeline_layout: {e:?}")))?;

    let entry_name = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_name);
    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout)];

    // SAFETY: `pipeline_info` is a valid array; the shader module and
    // layout outlive this call.
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None) }
            .map_err(|(_, e)| {
                HiveGpuError::VulkanError(format!("create_compute_pipelines: {e:?}"))
            })?;

    Ok((module, set_layout, layout, pipelines[0]))
}

impl GpuBackend for IntelContext {
    fn device_info(&self) -> GpuDeviceInfo {
        // Sum up heap sizes for a rough total VRAM estimate.
        let total_vram_bytes: u64 = self.memory_properties.memory_heaps
            [..self.memory_properties.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();

        GpuDeviceInfo {
            name: self.device_name.clone(),
            backend: "Intel".to_string(),
            total_vram_bytes,
            // Vulkan has no portable live-memory query short of the
            // VK_EXT_memory_budget extension; we defer to total == used-
            // free math in a future iteration.
            available_vram_bytes: total_vram_bytes,
            used_vram_bytes: 0,
            driver_version: format!(
                "Vulkan {}.{}.{} (driver 0x{:x})",
                vk::api_version_major(self.api_version),
                vk::api_version_minor(self.api_version),
                vk::api_version_patch(self.api_version),
                self.driver_version
            ),
            compute_capability: Some(format!(
                "vk{}.{}-0x{:04x}",
                vk::api_version_major(self.api_version),
                vk::api_version_minor(self.api_version),
                self.device_id
            )),
            max_threads_per_block: self.limits.max_compute_work_group_invocations,
            max_shared_memory_per_block: self.limits.max_compute_shared_memory_size as u64,
            device_id: self.device_id as i32,
            pci_bus_id: None,
        }
    }

    fn supports_operations(&self) -> GpuCapabilities {
        GpuCapabilities {
            supports_hnsw: false,
            supports_batch: true,
            max_dimension: 4096,
            max_batch_size: 100_000,
        }
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        let info = GpuBackend::device_info(self);
        let used = info.used_vram_bytes as usize;
        let available = info.available_vram_bytes as usize;
        let total = info.total_vram_bytes as usize;
        GpuMemoryStats {
            total_allocated: used,
            available,
            utilization: if total == 0 {
                0.0
            } else {
                used as f32 / total as f32
            },
            buffer_count: 0,
        }
    }
}

impl GpuContext for IntelContext {
    fn create_storage(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        use super::vector_storage::IntelVectorStorage;
        // Same Arc-aliasing pattern as the ROCm backend: the IntelContext
        // is constructed as Arc<Self>; downstream storages hold a cloned
        // Arc so the Vulkan objects outlive them.
        // SAFETY: `self` is always held in an `Arc<IntelContext>` thanks
        // to `new_with_preference` being the only public constructor.
        let ctx: Arc<Self> = unsafe {
            let raw = self as *const Self;
            Arc::increment_strong_count(raw);
            Arc::from_raw(raw)
        };
        let storage = IntelVectorStorage::new(ctx, dimension, metric)?;
        Ok(Box::new(storage))
    }

    fn create_storage_with_config(
        &self,
        dimension: usize,
        metric: crate::types::GpuDistanceMetric,
        _config: crate::types::HnswConfig,
    ) -> Result<Box<dyn crate::traits::GpuVectorStorage>> {
        self.create_storage(dimension, metric)
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        GpuBackend::memory_stats(self)
    }

    fn device_info(&self) -> Result<GpuDeviceInfo> {
        Ok(GpuBackend::device_info(self))
    }
}

// Re-export of ComputePipeline::new so this module stays the sole creator.
