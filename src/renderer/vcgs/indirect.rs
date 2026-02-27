//! Indirect Draw Pass
//!
//! Manages GPU-driven indirect rendering with occlusion culling.
//! Uses the Hi-Z pyramid to cull objects before generating indirect draw commands.

use ash::vk;
use std::sync::Arc;

use super::culling::{CullObjectData, OcclusionCulling};
use crate::Result;
use crate::vulkan::descriptor_bindless::BindlessManager;
use crate::vulkan::{Allocator, VulkanDevice};

/// Maximum objects per frame for indirect drawing
pub const MAX_INDIRECT_OBJECTS: usize = 1_048_576; // 1M clusters
pub const MAX_DRAWS: usize = 65536; // Matches occlusion_culling::MAX_CULLABLE_OBJECTS

/// GPU resources for indirect draw pass
pub struct IndirectDrawPass {
    device: Arc<ash::Device>,

    // Object data buffer (input)
    object_buffer: vk::Buffer,
    object_allocation: Option<vk_mem::Allocation>,
    object_buffer_size: u64,
    object_buffer_index: u32,

    // Draw commands template buffer (input) - DELETED (Using BDA Mesh Data Directly)

    // Indirect draw commands buffer (output)
    indirect_buffer: vk::Buffer,
    indirect_allocation: Option<vk_mem::Allocation>,

    // Visible count buffer (atomic counter)
    count_buffer: vk::Buffer,
    count_allocation: Option<vk_mem::Allocation>,

    // Cull pipeline for culling
    cull_pipeline: vk::Pipeline,
    cull_layout: vk::PipelineLayout,

    // Cached BDA pointers (computed at creation, zero-cost to access at runtime)
    object_buffer_addr: u64,
    indirect_buffer_addr: u64,
    count_buffer_addr: u64,

    initialized: bool,
    destroyed: bool,

    allocator: Option<Arc<Allocator>>,
}

impl IndirectDrawPass {
    /// Create a new indirect draw pass (uninitialized)
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            object_buffer: vk::Buffer::null(),
            object_allocation: None,
            object_buffer_size: 0,
            object_buffer_index: 0,
            indirect_buffer: vk::Buffer::null(),
            indirect_allocation: None,
            count_buffer: vk::Buffer::null(),
            count_allocation: None,
            cull_pipeline: vk::Pipeline::null(),
            cull_layout: vk::PipelineLayout::null(),
            object_buffer_addr: 0,
            indirect_buffer_addr: 0,
            count_buffer_addr: 0,
            initialized: false,
            destroyed: false,
            allocator: None,
        }
    }

    /// Initialize indirect draw pass resources.
    ///
    /// # Safety
    /// Caller must ensure that the provided allocator, device, and bindless manager are valid and remain active for the duration of the pass.
    pub unsafe fn init(
        &mut self,
        allocator: &Arc<Allocator>,
        _vulkan_device: &VulkanDevice,
        bindless_manager: &mut BindlessManager,
        max_objects: usize,
    ) -> Result<()> {
        self.allocator = Some(Arc::clone(allocator));
        let vma_allocator = &allocator.vma;
        if self.initialized {
            return Ok(());
        }

        unsafe { self.create_buffers(vma_allocator, max_objects) }?;

        // Register object buffer with BindlessManager
        let index = bindless_manager.add_storage_buffer(self.object_buffer, 0, vk::WHOLE_SIZE)?;
        self.object_buffer_index = index;

        unsafe { self.create_pipeline() }?;

        self.initialized = true;
        Ok(())
    }

    /// Create GPU buffers
    unsafe fn create_buffers(
        &mut self,
        allocator: &vk_mem::Allocator,
        max_objects: usize,
    ) -> Result<()> {
        use vk_mem::Alloc;

        let object_size = (std::mem::size_of::<CullObjectData>() * max_objects) as u64;
        let command_size =
            (std::mem::size_of::<vk::DrawIndexedIndirectCommand>() * max_objects) as u64;
        let count_size = 16u64; // Atomic counter + padding

        let buffer_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let device_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        // Object buffer (CPU writable)
        let object_info = vk::BufferCreateInfo::default().size(object_size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let (object_buffer, mut object_alloc) =
            unsafe { allocator.create_buffer(&object_info, &buffer_alloc_info) }
                .map_err(|e| crate::AshError::VulkanError(format!("Object buffer: {e:?}")))?;

        // SAFETY: BDA requires initialized memory. Zero it out to prevent wild pointers.
        // SAFETY: Directly mapping memory and writing zero bytes to ensure
        // BDA pointers are initialized to null/zero state.
        unsafe {
            let ptr = allocator.map_memory(&mut object_alloc)?;
            std::ptr::write_bytes(ptr, 0, object_size as usize);

            // CRITICAL: Flush to ensure GPU sees the zeros!
            allocator.flush_allocation(&object_alloc, 0, object_size)?;

            allocator.unmap_memory(&mut object_alloc);
        }

        // Check address alignment
        let info = vk::BufferDeviceAddressInfo::default().buffer(object_buffer);
        let addr = unsafe { self.device.get_buffer_device_address(&info) };
        log::info!(
            "IndirectDrawPass: Object Buffer Address = {addr:#x} (Aligned: {aligned})",
            aligned = addr % 16 == 0
        );

        self.object_buffer = object_buffer;
        self.object_allocation = Some(object_alloc);
        self.object_buffer_size = object_size;

        // Indirect buffer (GPU only, indirect draw source)
        let indirect_info = vk::BufferCreateInfo::default().size(command_size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let (indirect_buffer, indirect_alloc) =
            unsafe { allocator.create_buffer(&indirect_info, &device_alloc_info) }
                .map_err(|e| crate::AshError::VulkanError(format!("Indirect buffer: {e:?}")))?;
        self.indirect_buffer = indirect_buffer;
        self.indirect_allocation = Some(indirect_alloc);

        // Count buffer (GPU readback)
        let count_info = vk::BufferCreateInfo::default().size(count_size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let (count_buffer, count_alloc) =
            unsafe { allocator.create_buffer(&count_info, &buffer_alloc_info) }
                .map_err(|e| crate::AshError::VulkanError(format!("Count buffer: {e:?}")))?;
        self.count_buffer = count_buffer;
        self.count_allocation = Some(count_alloc);

        // Cache all BDA addresses at creation time — zero-cost access at dispatch.
        self.object_buffer_addr = unsafe {
            let info = vk::BufferDeviceAddressInfo::default().buffer(self.object_buffer);
            self.device.get_buffer_device_address(&info)
        };
        self.indirect_buffer_addr = unsafe {
            let info = vk::BufferDeviceAddressInfo::default().buffer(self.indirect_buffer);
            self.device.get_buffer_device_address(&info)
        };
        self.count_buffer_addr = unsafe {
            let info = vk::BufferDeviceAddressInfo::default().buffer(self.count_buffer);
            self.device.get_buffer_device_address(&info)
        };

        log::debug!(
            "IndirectDrawPass: Created buffers (obj={object_size}, cmd={command_size}, cnt={count_size})"
        );
        log::info!(
            "IndirectDrawPass BDA: obj={:#018X} indirect={:#018X} cnt={:#018X}",
            self.object_buffer_addr,
            self.indirect_buffer_addr,
            self.count_buffer_addr
        );
        Ok(())
    }

    /// Create the compute pipeline. The layout has ZERO descriptor sets.
    /// All resources are addressed via BDA push constants — no descriptor bindings.
    unsafe fn create_pipeline(&mut self) -> Result<()> {
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/cull_instances.comp.spv"));
        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(shader_code));
        let shader_module = unsafe { self.device.create_shader_module(&shader_module_info, None) }?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<crate::renderer::types::GpuPushConstants>() as u32);

        // Phase 4: Zero descriptor sets. Pure BDA pipeline.
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[])
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        self.cull_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.cull_layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| e)?;
        self.cull_pipeline = pipelines[0];
        unsafe { self.device.destroy_shader_module(shader_module, None) };
        log::info!("IndirectDrawPass: Pipeline created successfully");
        Ok(())
    }

    /// Hot-reload the culling compute pipeline with new SPIR-V bytecode.
    ///
    /// # Safety
    /// Caller must ensure the pipeline is not in flight on the GPU.
    pub unsafe fn reload_pipeline(&mut self, spirv_code: &[u32]) -> Result<()> {
        log::info!("IndirectDrawPass: Reloading pipeline...");
        if self.cull_pipeline != vk::Pipeline::null() {
            unsafe { self.device.destroy_pipeline(self.cull_pipeline, None) };
            self.cull_pipeline = vk::Pipeline::null();
        }
        if self.cull_layout != vk::PipelineLayout::null() {
            unsafe { self.device.destroy_pipeline_layout(self.cull_layout, None) };
            self.cull_layout = vk::PipelineLayout::null();
        }
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(spirv_code);
        let shader_module = unsafe { self.device.create_shader_module(&shader_module_info, None) }?;
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<crate::renderer::types::GpuPushConstants>() as u32);
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[])
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        self.cull_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }?;
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.cull_layout);
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| e)?;
        self.cull_pipeline = pipelines[0];
        unsafe { self.device.destroy_shader_module(shader_module, None) };
        log::info!("IndirectDrawPass: Pipeline reloaded successfully");
        Ok(())
    }

    /// Upload object data for culling
    ///
    /// # Safety
    /// Allocator must be valid. The provided offset and objects slice must not exceed the allocated buffer capacity.
    pub unsafe fn upload_objects(
        &self,
        allocator: &vk_mem::Allocator,
        objects: &[CullObjectData],
        offset: usize,
    ) -> Result<()> {
        if !self.initialized || objects.is_empty() {
            return Ok(());
        }

        if let Some(ref alloc) = self.object_allocation {
            let info = allocator.get_allocation_info(alloc);
            if !info.mapped_data.is_null() {
                // CRITICAL SAFETY: Bounds check with overflow protection before write
                let object_size = std::mem::size_of::<CullObjectData>();
                let required_size = objects
                    .len()
                    .checked_mul(object_size)
                    .and_then(|total_obj_size| offset.checked_add(total_obj_size))
                    .ok_or_else(|| {
                        crate::AshError::VulkanError(
                            "IndirectDraw: Upload size arithmetic overflow".to_string(),
                        )
                    })?;

                if required_size > info.size as usize {
                    return Err(crate::AshError::VulkanError(format!(
                        "IndirectDraw: Object buffer overrun. Size: {}, Allocation: {}",
                        required_size, info.size
                    )));
                }

                let dest = unsafe { (info.mapped_data as *mut CullObjectData).add(offset) };
                unsafe {
                    std::ptr::copy_nonoverlapping(objects.as_ptr(), dest, objects.len());
                }

                // CRITICAL FIX: Flush memory to ensure GPU visibility on non-coherent heaps
                allocator.flush_allocation(
                    alloc,
                    offset as u64,
                    std::mem::size_of_val(objects) as u64,
                )?;
            }
        }

        Ok(())
    }

    /// Execute culling pass
    ///
    /// # Safety
    /// Command buffer must be in recording state. All bound resources (object buffer, hiz image, etc.) must be valid and correctly synchronized for compute access.
    pub unsafe fn execute_culling(
        &self,
        cmd: vk::CommandBuffer,
        culling: &OcclusionCulling,
        ctx: &crate::renderer::types::CullingContext,
    ) -> Result<()> {
        if !self.initialized || !culling.is_enabled() || ctx.object_count == 0 {
            return Ok(());
        }

        // SAFETY: BDA crash prevention
        // If the object buffer address is 0 (uninitialized) or the buffer has not been uploaded,
        // dispatching the shader will cause a TDR/Device Lost error.
        let object_addr = self.object_buffer_address();
        if object_addr == 0 {
            return Ok(());
        }

        // Reset count buffer
        unsafe {
            self.device.cmd_fill_buffer(cmd, self.count_buffer, 0, 4, 0);
        }

        // Barrier for fill (Sync2)
        let barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
            .buffer(self.count_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        let buffer_barriers = [barrier];
        let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);
        unsafe {
            self.device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

        // Bind pipeline
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
        }

        // Phase 2+3 BDA Routing: all output buffers routed as raw 64-bit pointers.
        // No descriptor sets needed — the shader receives raw 64-bit pointers.
        // Push constants — view_proj removed (Phase 3: shader reads it from FrameData UBO via BDA)
        let mut push = culling.push_constants();
        push.object_count = ctx.object_count;
        push.base_index = ctx.object_offset;
        push.indirect_start = ctx.indirect_offset;

        // Phase 2+3 BDA Routing: map to unified GpuPushConstants
        push.instance_ptr = self.object_buffer_addr;
        push.material_ptr = self.indirect_buffer_addr;
        push.index_ptr = self.count_buffer_addr;
        push.light_ptr = ctx.hiz_buffer_addr;
        push.frame_ptr = ctx.camera_buffer_addr;

        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.cull_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
        }

        // Dispatch: 64 threads per workgroup
        let group_count = ctx.object_count.div_ceil(64);
        unsafe {
            self.device.cmd_dispatch(cmd, group_count, 1, 1);

            // 4. CRITICAL BARRIER: Compute-to-Graphics for Indirect Buffers (Sync2)
            let indirect_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                .buffer(self.indirect_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            let count_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                .buffer(self.count_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            let buffer_barriers = [indirect_barrier, count_barrier];
            let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);
            self.device.cmd_pipeline_barrier2(cmd, &dep_info);
        }

        Ok(())
    }

    /// Get indirect buffer for drawing
    pub fn object_buffer(&self) -> vk::Buffer {
        self.object_buffer
    }

    pub fn object_buffer_index(&self) -> u32 {
        self.object_buffer_index
    }

    /// Get object buffer device address (cached at creation)
    pub fn object_buffer_address(&self) -> u64 {
        self.object_buffer_addr
    }

    /// Get indirect buffer device address (cached at creation)
    pub fn indirect_buffer_address(&self) -> u64 {
        self.indirect_buffer_addr
    }

    /// Get count buffer device address (cached at creation)
    pub fn count_buffer_address(&self) -> u64 {
        self.count_buffer_addr
    }

    /// Get count buffer for indirect count
    pub fn count_buffer(&self) -> vk::Buffer {
        self.count_buffer
    }

    pub fn indirect_buffer(&self) -> vk::Buffer {
        self.indirect_buffer
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn object_buffer_size(&self) -> vk::DeviceSize {
        self.object_buffer_size
    }

    /// Read the visible count back to the CPU
    ///
    /// # Safety
    /// Allocator must be valid. The caller must ensure that previous GPU commands writing to the count buffer have completed (e.g., via sync fences).
    pub unsafe fn read_visible_count(&self, allocator: &vk_mem::Allocator) -> u32 {
        if !self.initialized {
            return 0;
        }

        if let Some(ref alloc) = self.count_allocation {
            let info = allocator.get_allocation_info(alloc);
            if !info.mapped_data.is_null() {
                let ptr = info.mapped_data as *const u32;
                return unsafe { *ptr };
            }
        }

        0
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Caller must ensure that no GPU commands are currently referencing any resources within this pass.
    pub unsafe fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        if !self.initialized {
            return;
        }

        let allocator = if let Some(ref a) = self.allocator {
            &a.vma
        } else {
            log::error!("IndirectDrawPass: Destroy called without allocator!");
            return;
        };

        // Destroy buffers
        if let Some(mut alloc) = self.object_allocation.take() {
            unsafe {
                allocator.destroy_buffer(self.object_buffer, &mut alloc);
            }
        }
        if let Some(mut alloc) = self.indirect_allocation.take() {
            unsafe {
                allocator.destroy_buffer(self.indirect_buffer, &mut alloc);
            }
        }
        if let Some(mut alloc) = self.count_allocation.take() {
            unsafe {
                allocator.destroy_buffer(self.count_buffer, &mut alloc);
            }
        }

        if self.cull_pipeline != vk::Pipeline::null() {
            unsafe { self.device.destroy_pipeline(self.cull_pipeline, None) };
        }
        if self.cull_layout != vk::PipelineLayout::null() {
            unsafe { self.device.destroy_pipeline_layout(self.cull_layout, None) };
        }
        self.initialized = false;
        log::info!("IndirectDrawPass: Resources destroyed");
    }
}

impl Drop for IndirectDrawPass {
    fn drop(&mut self) {
        // SAFETY: Destruction of raw Vulkan resources. Caller must ensure GPU is idle.
        unsafe {
            self.destroy();
        }
    }
}

impl crate::renderer::cleanup_traits::VulkanResourceCleanup for IndirectDrawPass {
    fn cleanup_with_device(&mut self, _device: &ash::Device) -> std::result::Result<(), String> {
        // SAFETY: Delegate cleanup to the destroy method.
        unsafe {
            self.destroy();
        }
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "IndirectDrawPass"
    }
}

impl crate::renderer::resource_registry::VulkanResource for IndirectDrawPass {}
