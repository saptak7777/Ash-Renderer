//! Indirect Draw Pass
//!
//! Manages GPU-driven indirect rendering with occlusion culling.
//! Uses the Hi-Z pyramid to cull objects before generating indirect draw commands.

use ash::vk;
use std::sync::Arc;

use crate::renderer::hiz_pass::HiZPass;
use crate::renderer::occlusion_culling::{CullObjectData, CullingPushConstants, OcclusionCulling};
use crate::vulkan::VulkanDevice;
use crate::Result;

/// Maximum objects per frame for indirect drawing
pub const MAX_INDIRECT_OBJECTS: usize = 65536;

/// GPU resources for indirect draw pass
pub struct IndirectDrawPass {
    device: Arc<ash::Device>,

    // Object data buffer (input)
    object_buffer: vk::Buffer,
    object_allocation: Option<vk_mem::Allocation>,
    object_buffer_size: u64,

    // Draw commands template buffer (input)
    template_buffer: vk::Buffer,
    template_allocation: Option<vk_mem::Allocation>,

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

    // Descriptor resources
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,

    initialized: bool,
}

impl IndirectDrawPass {
    /// Create a new indirect draw pass (uninitialized)
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            object_buffer: vk::Buffer::null(),
            object_allocation: None,
            object_buffer_size: 0,
            template_buffer: vk::Buffer::null(),
            template_allocation: None,
            indirect_buffer: vk::Buffer::null(),
            indirect_allocation: None,
            visibility_buffer: vk::Buffer::null(),
            visibility_allocation: None,
            count_buffer: vk::Buffer::null(),
            count_allocation: None,
            cull_pipeline: vk::Pipeline::null(),
            cull_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_set: vk::DescriptorSet::null(),
            initialized: false,
        }
    }

    /// Initialize GPU resources
    ///
    /// # Safety
    /// Allocator must be valid.
    pub unsafe fn initialize(
        &mut self,
        allocator: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        max_objects: usize,
    ) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        log::info!("IndirectDrawPass: Initializing for {max_objects} objects");

        // Create buffers
        self.create_buffers(allocator, max_objects)?;

        // Create descriptor layout and pool
        self.create_descriptors()?;

        // Load and create compute pipeline
        self.create_pipeline()?;

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
        let object_info = vk::BufferCreateInfo::default()
            .size(object_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let (object_buffer, object_alloc) = allocator
            .create_buffer(&object_info, &buffer_alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Object buffer: {e:?}")))?;
        self.object_buffer = object_buffer;
        self.object_allocation = Some(object_alloc);
        self.object_buffer_size = object_size;

        // Template buffer (CPU writable)
        let template_info = vk::BufferCreateInfo::default()
            .size(command_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let (template_buffer, template_alloc) = allocator
            .create_buffer(&template_info, &buffer_alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Template buffer: {e:?}")))?;
        self.template_buffer = template_buffer;
        self.template_allocation = Some(template_alloc);

        // Indirect buffer (GPU only, indirect draw source)
        let indirect_info = vk::BufferCreateInfo::default()
            .size(command_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER);
        let (indirect_buffer, indirect_alloc) = allocator
            .create_buffer(&indirect_info, &device_alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Indirect buffer: {e:?}")))?;
        self.indirect_buffer = indirect_buffer;
        self.indirect_allocation = Some(indirect_alloc);

        // Visibility buffer (GPU only)
        let visibility_info = vk::BufferCreateInfo::default()
            .size(visibility_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
        let (visibility_buffer, visibility_alloc) = allocator
            .create_buffer(&visibility_info, &device_alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Visibility buffer: {e:?}")))?;
        self.visibility_buffer = visibility_buffer;
        self.visibility_allocation = Some(visibility_alloc);

        // Count buffer (GPU readback)
        let count_info = vk::BufferCreateInfo::default()
            .size(count_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        let (count_buffer, count_alloc) = allocator
            .create_buffer(&count_info, &buffer_alloc_info)
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
        // Bindings match occlusion_cull.comp
        let bindings = [
            // 0: Object data
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 1: Hi-Z pyramid
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // 2: Draw commands template
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
        self.descriptor_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 5,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        self.descriptor_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_layout));

        let sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_set = sets[0];

        Ok(())
    }

    /// Create compute pipeline
    unsafe fn create_pipeline(&mut self) -> Result<()> {
        let shader_path = std::path::Path::new("shaders/occlusion_cull.spv");
        let shader_code = std::fs::read(shader_path)?;

        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&shader_code));
        let shader_module = self
            .device
            .create_shader_module(&shader_module_info, None)?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CullingPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.descriptor_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.cull_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.cull_layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?;

        self.cull_pipeline = pipelines[0];
        self.device.destroy_shader_module(shader_module, None);

        log::info!("IndirectDrawPass: Pipeline created successfully");
        Ok(())
    }

    /// Update descriptors with Hi-Z image
    ///
    /// # Safety
    /// Resources must be valid.
    pub unsafe fn update_hiz_descriptor(&self, hiz_pass: &HiZPass) {
        if !self.initialized {
            return;
        }

        let hiz_view = match hiz_pass.hiz_view() {
            Some(v) => v,
            None => return,
        };

        // Update buffer descriptors
        let object_info = vk::DescriptorBufferInfo::default()
            .buffer(self.object_buffer)
            .range(vk::WHOLE_SIZE);

        let template_info = vk::DescriptorBufferInfo::default()
            .buffer(self.template_buffer)
            .range(vk::WHOLE_SIZE);

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
            .sampler(hiz_pass.hiz_sampler())
            .image_view(hiz_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&object_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&hiz_image_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&template_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&visibility_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&indirect_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&count_info)),
        ];

        self.device.update_descriptor_sets(&writes, &[]);
    }

    /// Upload object data for culling
    ///
    /// # Safety
    /// Allocator must be valid.
    pub unsafe fn upload_objects(
        &self,
        allocator: &vk_mem::Allocator,
        culling: &OcclusionCulling,
    ) -> Result<()> {
        if !self.initialized || culling.object_count() == 0 {
            return Ok(());
        }

        let obj_data = culling.object_data();
        if let Some(ref alloc) = self.object_allocation {
            let info = allocator.get_allocation_info(alloc);
            if !info.mapped_data.is_null() {
                std::ptr::copy_nonoverlapping(
                    obj_data.as_ptr() as *const u8,
                    info.mapped_data as *mut u8,
                    std::mem::size_of_val(obj_data),
                );
            }
        }

        Ok(())
    }

    /// Execute culling pass
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn execute_culling(
        &self,
        cmd: vk::CommandBuffer,
        culling: &OcclusionCulling,
        view_proj: glam::Mat4,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if !self.initialized || !culling.is_enabled() || culling.object_count() == 0 {
            return Ok(());
        }

        // Reset count buffer
        self.device.cmd_fill_buffer(cmd, self.count_buffer, 0, 4, 0);

        // Barrier for fill
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .buffer(self.count_buffer)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );

        // Bind pipeline
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);

        // Bind descriptors
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.cull_layout,
            0,
            &[self.descriptor_set],
            &[],
        );

        // Push constants
        let push = culling.push_constants(view_proj, width, height);
        self.device.cmd_push_constants(
            cmd,
            self.cull_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push),
        );

        // Dispatch: 64 threads per workgroup
        let group_count = (culling.object_count() as u32).div_ceil(64);
        self.device.cmd_dispatch(cmd, group_count, 1, 1);

        // Barrier for indirect read
        let indirect_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
            .buffer(self.indirect_buffer)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::DRAW_INDIRECT,
            vk::DependencyFlags::empty(),
            &[],
            &[indirect_barrier],
            &[],
        );

        Ok(())
    }

    /// Get indirect buffer for drawing
    pub fn indirect_buffer(&self) -> vk::Buffer {
        self.indirect_buffer
    }

    /// Get count buffer for indirect count
    pub fn count_buffer(&self) -> vk::Buffer {
        self.count_buffer
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        // Destroy buffers
        if let Some(mut alloc) = self.object_allocation.take() {
            allocator.destroy_buffer(self.object_buffer, &mut alloc);
        }
        if let Some(mut alloc) = self.template_allocation.take() {
            allocator.destroy_buffer(self.template_buffer, &mut alloc);
        }
        if let Some(mut alloc) = self.indirect_allocation.take() {
            allocator.destroy_buffer(self.indirect_buffer, &mut alloc);
        }
        if let Some(mut alloc) = self.visibility_allocation.take() {
            allocator.destroy_buffer(self.visibility_buffer, &mut alloc);
        }
        if let Some(mut alloc) = self.count_allocation.take() {
            allocator.destroy_buffer(self.count_buffer, &mut alloc);
        }

        if self.cull_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.cull_pipeline, None);
        }
        if self.cull_layout != vk::PipelineLayout::null() {
            self.device.destroy_pipeline_layout(self.cull_layout, None);
        }
        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
        if self.descriptor_layout != vk::DescriptorSetLayout::null() {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }

        self.initialized = false;
        log::info!("IndirectDrawPass: Resources destroyed");
    }
}
