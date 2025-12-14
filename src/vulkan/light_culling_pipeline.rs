//! Light Culling Vulkan Pipeline
//!
//! Manages the compute pipeline, descriptor sets, and buffers for GPU light culling.

use ash::vk;

use std::sync::Arc;

use crate::renderer::features::light_culling::{
    CullingCameraData, GpuLight, LightCullingPushConstants, MAX_LIGHTS, MAX_LIGHTS_PER_TILE,
};
use crate::vulkan::{Allocator, ComputePipeline, VulkanDevice};
use crate::{AshError, Result};

/// Light culling compute pipeline and resources
pub struct LightCullingPipeline {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    // Buffers (managed by VMA)
    light_buffer: vk::Buffer,
    light_buffer_alloc: vk_mem::Allocation,
    tile_buffer: vk::Buffer,
    tile_buffer_alloc: vk_mem::Allocation,
    camera_buffer: vk::Buffer,
    camera_buffer_alloc: vk_mem::Allocation,
    // State
    #[allow(dead_code)]
    light_buffer_size: usize,
    tile_buffer_size: usize,
}

impl LightCullingPipeline {
    /// Create a new light culling pipeline
    ///
    /// # Safety
    /// Device must be valid and shader must be loaded.
    pub unsafe fn new(
        vulkan_device: &VulkanDevice,
        allocator: Arc<Allocator>,
        shader_module: vk::ShaderModule,
        depth_sampler: vk::Sampler,
        depth_image_view: vk::ImageView,
        screen_width: u32,
        screen_height: u32,
    ) -> Result<Self> {
        let device = Arc::clone(&vulkan_device.device);

        // Calculate buffer sizes
        let light_buffer_size = std::mem::size_of::<GpuLight>() * MAX_LIGHTS;
        let tiles_x = screen_width.div_ceil(16);
        let tiles_y = screen_height.div_ceil(16);
        let tile_buffer_size =
            (tiles_x * tiles_y) as usize * (MAX_LIGHTS_PER_TILE + 1) * std::mem::size_of::<u32>();
        let camera_buffer_size = std::mem::size_of::<CullingCameraData>();

        // Create descriptor set layout
        let bindings = [
            // Binding 0: Light buffer (SSBO)
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            // Binding 1: Depth buffer (sampler)
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            // Binding 2: Tile light indices (SSBO, writeonly)
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            // Binding 3: Camera data (UBO)
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create descriptor set layout: {e}"))
            })?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let descriptor_pool = device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| AshError::VulkanError(format!("Failed to create descriptor pool: {e}")))?;

        // Allocate descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let descriptor_sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
            AshError::VulkanError(format!("Failed to allocate descriptor sets: {e}"))
        })?;

        let descriptor_set = descriptor_sets[0];

        // Create buffers using VMA (consistent with rest of engine)
        // Light buffer (host-visible for CPU uploads)
        let (light_buffer, light_buffer_alloc) = allocator.create_buffer(
            light_buffer_size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferHost,
        )?;

        // Tile buffer (device-local for GPU-only access)
        let (tile_buffer, tile_buffer_alloc) = allocator.create_buffer(
            tile_buffer_size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
        )?;

        // Camera buffer (host-visible for CPU uploads)
        let (camera_buffer, camera_buffer_alloc) = allocator.create_buffer(
            camera_buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::AutoPreferHost,
        )?;

        // Update descriptor set
        let light_buffer_info = vk::DescriptorBufferInfo {
            buffer: light_buffer,
            offset: 0,
            range: light_buffer_size as u64,
        };

        let depth_image_info = vk::DescriptorImageInfo {
            sampler: depth_sampler,
            image_view: depth_image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let tile_buffer_info = vk::DescriptorBufferInfo {
            buffer: tile_buffer,
            offset: 0,
            range: tile_buffer_size as u64,
        };

        let camera_buffer_info = vk::DescriptorBufferInfo {
            buffer: camera_buffer,
            offset: 0,
            range: camera_buffer_size as u64,
        };

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&light_buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_image_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&tile_buffer_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&camera_buffer_info)),
        ];

        device.update_descriptor_sets(&writes, &[]);

        // Create push constant range
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<LightCullingPushConstants>() as u32,
        };

        // Create compute pipeline
        let pipeline = ComputePipeline::builder(Arc::clone(&device))
            .with_shader(shader_module)
            .add_set_layout(descriptor_set_layout)
            .add_push_constant(push_constant_range)
            .build()?;

        log::info!("Light culling pipeline created ({tiles_x}x{tiles_y} tiles)");

        Ok(Self {
            device,
            allocator,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            light_buffer,
            light_buffer_alloc,
            tile_buffer,
            tile_buffer_alloc,
            camera_buffer,
            camera_buffer_alloc,
            light_buffer_size,
            tile_buffer_size,
        })
    }

    /// Upload lights to GPU buffer
    /// # Safety
    /// Memory mapping requires external synchronization and valid buffer.
    pub unsafe fn upload_lights(&mut self, lights: &[GpuLight]) -> Result<()> {
        let data_ptr = self
            .allocator
            .vma
            .map_memory(&mut self.light_buffer_alloc)
            .map_err(|e| AshError::VulkanError(format!("Failed to map memory: {e:?}")))?;

        let slice =
            std::slice::from_raw_parts_mut(data_ptr as *mut GpuLight, lights.len().min(MAX_LIGHTS));
        slice.copy_from_slice(&lights[..slice.len()]);

        self.allocator
            .vma
            .unmap_memory(&mut self.light_buffer_alloc);
        Ok(())
    }

    /// Upload camera data
    /// # Safety
    /// Memory mapping requires external synchronization and valid buffer.
    pub unsafe fn upload_camera(&mut self, camera: &CullingCameraData) -> Result<()> {
        let data_ptr = self
            .allocator
            .vma
            .map_memory(&mut self.camera_buffer_alloc)
            .map_err(|e| AshError::VulkanError(format!("Failed to map memory: {e:?}")))?;

        std::ptr::copy_nonoverlapping(camera as *const _, data_ptr as *mut CullingCameraData, 1);

        self.allocator
            .vma
            .unmap_memory(&mut self.camera_buffer_alloc);
        Ok(())
    }

    /// Record dispatch commands
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        tiles_x: u32,
        tiles_y: u32,
        push_constants: &LightCullingPushConstants,
    ) {
        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.handle(),
        );

        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout(),
            0,
            &[self.descriptor_set],
            &[],
        );

        let pc_bytes = bytemuck::bytes_of(push_constants);
        self.device.cmd_push_constants(
            command_buffer,
            self.pipeline.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            pc_bytes,
        );

        self.device
            .cmd_dispatch(command_buffer, tiles_x, tiles_y, 1);

        // Memory barrier to ensure tile buffer writes are visible to fragment shader
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .buffer(self.tile_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }

    /// Get tile buffer for fragment shader binding
    pub fn tile_buffer(&self) -> vk::Buffer {
        self.tile_buffer
    }

    /// Get tile buffer size
    pub fn tile_buffer_size(&self) -> usize {
        self.tile_buffer_size
    }
}

impl Drop for LightCullingPipeline {
    fn drop(&mut self) {
        unsafe {
            // Destroy buffers using VMA
            self.allocator
                .destroy_buffer(self.light_buffer, &mut self.light_buffer_alloc);
            self.allocator
                .destroy_buffer(self.tile_buffer, &mut self.tile_buffer_alloc);
            self.allocator
                .destroy_buffer(self.camera_buffer, &mut self.camera_buffer_alloc);

            // Cleanup descriptor resources
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
        log::info!("Light culling pipeline destroyed");
    }
}
