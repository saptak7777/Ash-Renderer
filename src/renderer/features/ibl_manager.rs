use crate::renderer::resources::ImageHandle;
use crate::vulkan::{descriptor_layout::DescriptorSetLayoutBuilder, Allocator, VulkanDevice};
use crate::{AshError, Result};
use ash::vk;
use std::sync::Arc;

pub struct IblManager {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,

    // Compute Pipelines
    equirect_pipeline: vk::Pipeline,
    irradiance_pipeline: vk::Pipeline,
    prefilter_pipeline: vk::Pipeline,

    pipeline_layout: vk::PipelineLayout,
    descriptor_layout: crate::vulkan::descriptor_layout::DescriptorSetLayout,
    descriptor_allocator: crate::vulkan::descriptor_allocator::DescriptorAllocator,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct IblComputePushConstants {
    output_size: u32,
    roughness: f32,
    _padding: [u32; 2],
}

impl IblManager {
    pub fn new(device: Arc<ash::Device>, allocator: Arc<Allocator>) -> Result<Self> {
        // 1. Create Descriptor Set Layout
        let descriptor_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .build(Arc::clone(&device))?;

        // 2. Create Pipeline Layout
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<IblComputePushConstants>() as u32,
        };

        let layouts = [descriptor_layout.handle()];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // 3. Helper to create compute pipelines
        let create_compute = |shader_bytes: &[u8]| -> Result<vk::Pipeline> {
            let shader_module = unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(shader_bytes)),
                    None,
                )?
            };

            let entry_point = c"main";
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry_point);

            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);

            let pipelines = unsafe {
                device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| {
                        AshError::VulkanError(format!("Failed to create compute pipeline: {e}"))
                    })?
            };

            unsafe { device.destroy_shader_module(shader_module, None) };
            Ok(pipelines[0])
        };

        let equirect_pipeline = create_compute(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/equirect_to_cubemap.comp.spv"
        )))?;
        let irradiance_pipeline = create_compute(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/irradiance_convolution.comp.spv"
        )))?;
        let prefilter_pipeline = create_compute(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/prefilter_envmap.comp.spv"
        )))?;

        let descriptor_allocator = crate::vulkan::descriptor_allocator::DescriptorAllocator::new(
            Arc::clone(&device),
            64, // Enough sets for baking passes
            None,
        )?;

        Ok(Self {
            device,
            allocator,
            equirect_pipeline,
            irradiance_pipeline,
            prefilter_pipeline,
            pipeline_layout,
            descriptor_layout,
            descriptor_allocator,
        })
    }

    /// Converts an equirectangular environment map to a cubemap using compute shaders.
    pub fn create_cubemap_from_equirect(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        equirect_view: vk::ImageView,
        equirect_sampler: vk::Sampler,
        resolution: u32,
    ) -> Result<ImageHandle> {
        let env_cubemap = ImageHandle::create_cubemap(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            Some("EnvironmentCubemap".to_string()),
        )?;

        self.dispatch_compute(
            vulkan_device,
            command_pool,
            self.equirect_pipeline,
            equirect_view,
            equirect_sampler,
            &env_cubemap,
            0,
            0.0,
        )?;

        Ok(env_cubemap)
    }

    /// Generates an irradiance map from an environment cubemap using compute shaders.
    pub fn generate_irradiance(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        env_view: vk::ImageView,
        env_sampler: vk::Sampler,
    ) -> Result<ImageHandle> {
        let resolution = 32;
        let irradiance_map = ImageHandle::create_cubemap(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            Some("IrradianceMap".to_string()),
        )?;

        self.dispatch_compute(
            vulkan_device,
            command_pool,
            self.irradiance_pipeline,
            env_view,
            env_sampler,
            &irradiance_map,
            0,
            0.0,
        )?;

        Ok(irradiance_map)
    }

    /// Generates a pre-filtered environment map for specular IBL using compute shaders.
    pub fn generate_prefiltered(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        env_view: vk::ImageView,
        env_sampler: vk::Sampler,
    ) -> Result<ImageHandle> {
        let resolution = 256;
        let max_mips = 5;

        let prefiltered_map = ImageHandle::create_cubemap(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution,
            max_mips,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            Some("PrefilteredMap".to_string()),
        )?;

        for mip in 0..max_mips {
            let roughness = mip as f32 / (max_mips - 1) as f32;
            self.dispatch_compute(
                vulkan_device,
                command_pool,
                self.prefilter_pipeline,
                env_view,
                env_sampler,
                &prefiltered_map,
                mip,
                roughness,
            )?;
        }

        Ok(prefiltered_map)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_compute(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        pipeline: vk::Pipeline,
        source_view: vk::ImageView,
        source_sampler: vk::Sampler,
        target: &ImageHandle,
        mip: u32,
        roughness: f32,
    ) -> Result<()> {
        let resolution = target.extent().width >> mip;

        // 1. Descriptor Set (REUSED)
        let desc_set = self.descriptor_allocator.allocate_static_set(
            &self.descriptor_layout.handle(),
            self.descriptor_layout.bindings(),
        )?;

        desc_set.update_image(
            0,
            source_view,
            source_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        // Image view for the specific mip level (for storage)
        let view_info = vk::ImageViewCreateInfo::default()
            .image(target.handle())
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(target.format())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });
        let target_view = unsafe { self.device.create_image_view(&view_info, None)? };

        desc_set.update_image(
            1,
            target_view,
            vk::Sampler::null(),
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        )?;

        // 2. Execute
        vulkan_device.execute_single_use(command_pool, |cmd| {
            // Transition target to GENERAL for writing
            let barrier_start = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(target.handle())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });

            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_start],
                );

                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);

                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[desc_set.handle()],
                    &[],
                );

                let push = IblComputePushConstants {
                    output_size: resolution,
                    roughness,
                    _padding: [0; 2],
                };
                self.device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push),
                );

                // Dispatch for all 6 faces
                let groups = resolution.div_ceil(8);
                self.device.cmd_dispatch(cmd, groups, groups, 6);

                // Transition back to SHADER_READ_ONLY for sampling
                let barrier_end = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(target.handle())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: mip,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 6,
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_end],
                );
            }
        })?;

        unsafe {
            self.device.destroy_image_view(target_view, None);
        }

        Ok(())
    }

    pub fn destroy(&mut self) {
        unsafe {
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.equirect_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.equirect_pipeline, None);
                self.equirect_pipeline = vk::Pipeline::null();
            }
            if self.irradiance_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.irradiance_pipeline, None);
                self.irradiance_pipeline = vk::Pipeline::null();
            }
            if self.prefilter_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.prefilter_pipeline, None);
                self.prefilter_pipeline = vk::Pipeline::null();
            }
        }
    }
}

impl Drop for IblManager {
    fn drop(&mut self) {
        self.destroy();
    }
}
