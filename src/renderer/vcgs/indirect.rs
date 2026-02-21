//! Indirect Draw Pass
//!
//! Manages GPU-driven indirect rendering with occlusion culling.
//! Uses the Hi-Z pyramid to cull objects before generating indirect draw commands.

use ash::vk;
use std::sync::Arc;

use super::culling::{CullObjectData, CullingPushConstants, OcclusionCulling};
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

    // Visibility flags buffer (output)
    visibility_buffer: vk::Buffer,
    visibility_allocation: Option<vk_mem::Allocation>,

    // Visible count buffer (atomic counter)
    count_buffer: vk::Buffer,
    count_allocation: Option<vk_mem::Allocation>,

    // Compute pipeline for culling
    cull_pipeline: vk::Pipeline,
    cull_layout: vk::PipelineLayout,

    // Descriptors
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,

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
            // template_buffer: vk::Buffer::null(),
            // template_allocation: None,
            indirect_buffer: vk::Buffer::null(),
            indirect_allocation: None,
            visibility_buffer: vk::Buffer::null(),
            visibility_allocation: None,
            count_buffer: vk::Buffer::null(),
            count_allocation: None,
            cull_pipeline: vk::Pipeline::null(),
            cull_layout: vk::PipelineLayout::null(),
            pool: vk::DescriptorPool::null(),
            layout: vk::DescriptorSetLayout::null(),
            set: vk::DescriptorSet::null(),
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

        unsafe { self.create_descriptors() }?;
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
        let visibility_size = (max_objects * 4) as u64; // u32 per object
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

        // Template buffer - DELETED

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

        // Visibility buffer (GPU only)
        let visibility_info = vk::BufferCreateInfo::default().size(visibility_size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let (visibility_buffer, visibility_alloc) =
            unsafe { allocator.create_buffer(&visibility_info, &device_alloc_info) }
                .map_err(|e| crate::AshError::VulkanError(format!("Visibility buffer: {e:?}")))?;
        self.visibility_buffer = visibility_buffer;
        self.visibility_allocation = Some(visibility_alloc);

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

        log::debug!(
            "IndirectDrawPass: Created buffers (obj={object_size}, cmd={command_size}, vis={visibility_size}, cnt={count_size})"
        );
        Ok(())
    }

    /// Create descriptor layout and pool
    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // Bindings match cull_instances.comp
        let bindings = [
            // 1: Hi-Z pyramid
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 3: Visibility flags
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 4: Indirect draw commands
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 5: Visible count
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        self.pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(std::slice::from_ref(&self.layout));

        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?;
        self.set = sets[0];

        Ok(())
    }

    /// Create compute pipeline
    unsafe fn create_pipeline(&mut self) -> Result<()> {
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/cull_instances.comp.spv"));

        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(shader_code));
        let shader_module = unsafe { self.device.create_shader_module(&shader_module_info, None) }?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CullingPushConstants>() as u32);

        // Define layouts: Only Set 0 (Indirect Resources) is needed now. Set 1 (Bindless) is gone.
        let layouts = [self.layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
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
        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        log::info!("IndirectDrawPass: Pipeline created successfully");
        Ok(())
    }

    /// Reload compute pipeline with new shader code
    ///
    /// # Safety
    /// Caller must ensure that the pipeline is not currently in use by the GPU. The provided SPIR-V code must be valid compute shader code.
    pub unsafe fn reload_pipeline(&mut self, spirv_code: &[u32]) -> Result<()> {
        log::info!("IndirectDrawPass: Reloading pipeline...");

        // Destroy old pipeline
        if self.cull_pipeline != vk::Pipeline::null() {
            unsafe {
                self.device.destroy_pipeline(self.cull_pipeline, None);
            }
            self.cull_pipeline = vk::Pipeline::null();
        }
        if self.cull_layout != vk::PipelineLayout::null() {
            unsafe {
                self.device.destroy_pipeline_layout(self.cull_layout, None);
            }
            self.cull_layout = vk::PipelineLayout::null();
        }

        // Create new shader module
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(spirv_code);
        let shader_module = unsafe { self.device.create_shader_module(&shader_module_info, None) }?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CullingPushConstants>() as u32);

        let layouts = [self.layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
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
        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        log::info!("IndirectDrawPass: Pipeline reloaded successfully");
        Ok(())
    }

    /// Update descriptors with Hi-Z image
    ///
    /// # Safety
    /// Hi-Z view and sampler must be valid and accurately represent the current frame's occlusion data.
    pub unsafe fn update_hiz_descriptor(&self, hiz_view: vk::ImageView, hiz_sampler: vk::Sampler) {
        if !self.initialized {
            return;
        }

        // Update buffer descriptors

        let visibility_info = vk::DescriptorBufferInfo::default()
            .buffer(self.visibility_buffer)
            .range(vk::WHOLE_SIZE);

        let indirect_info = vk::DescriptorBufferInfo::default()
            .buffer(self.indirect_buffer)
            .range(vk::WHOLE_SIZE);

        let count_info = vk::DescriptorBufferInfo::default()
            .buffer(self.count_buffer)
            .range(vk::WHOLE_SIZE);

        let hiz_image_info = vk::DescriptorImageInfo::default()
            .sampler(hiz_sampler)
            .image_view(hiz_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&hiz_image_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&visibility_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&indirect_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&count_info)),
        ];

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
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

        // log::debug!("IndirectDraw: Uploading {} objects", objects.len());

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

    // upload_templates DELETED

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
            // This is expected on the first frame if upload hasn't happened yet
            // log::warn!("IndirectDrawPass: Object buffer address is 0, skipping culling dispatch");
            return Ok(());
        }

        // Reset count buffer
        unsafe {
            self.device.cmd_fill_buffer(cmd, self.count_buffer, 0, 4, 0);
        }

        // Barrier for fill
        let barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            buffer: self.count_buffer,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }

        // Bind pipeline
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
        }

        // Bind descriptors (Set 0)
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cull_layout,
                0,
                &[self.set],
                &[],
            );
        }

        // Bindless (Set 1) REMOVED - using BDA now

        // Push constants
        let mut push = culling.push_constants(ctx.view_proj, ctx.width, ctx.height);
        push.object_count = ctx.object_count;
        push.base_index = ctx.object_offset;
        push.indirect_start = ctx.indirect_offset;

        // ADDRESS INJECTION: Pass the BDA pointer directly to the shader
        push.object_buffer_addr = self.object_buffer_address();
        push.cluster_buffer_addr = ctx.cluster_buffer_addr;

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
        }

        // Barrier for indirect read
        // Barriers for Indirect Draw & Count Read
        let indirect_barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::INDIRECT_COMMAND_READ,
            buffer: self.indirect_buffer,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };

        let count_barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::INDIRECT_COMMAND_READ,
            buffer: self.count_buffer,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[],
                &[indirect_barrier, count_barrier],
                &[],
            );
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

    /// Get object buffer device address for BDA pulling
    pub fn object_buffer_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.object_buffer);
        // SAFETY: Retrieves the device address of the object buffer. Buffer must be valid.
        unsafe { self.device.get_buffer_device_address(&info) }
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
        // if let Some(mut alloc) = self.template_allocation.take() {
        //     allocator.destroy_buffer(self.template_buffer, &mut alloc);
        // }
        if let Some(mut alloc) = self.indirect_allocation.take() {
            unsafe {
                allocator.destroy_buffer(self.indirect_buffer, &mut alloc);
            }
        }
        if let Some(mut alloc) = self.visibility_allocation.take() {
            unsafe {
                allocator.destroy_buffer(self.visibility_buffer, &mut alloc);
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
        if self.pool != vk::DescriptorPool::null() {
            unsafe {
                self.device.destroy_descriptor_pool(self.pool, None);
            }
        }
        if self.layout != vk::DescriptorSetLayout::null() {
            unsafe {
                self.device.destroy_descriptor_set_layout(self.layout, None);
            }
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
