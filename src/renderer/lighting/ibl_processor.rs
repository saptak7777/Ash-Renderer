use crate::renderer::resources::Texture;
use crate::vulkan::{ComputePipeline, ComputePipelineBuilder, VulkanDevice};
use crate::Result;
use ash::vk;
use std::sync::Arc;

/// Container for generated IBL textures.
pub struct IblBundle {
    pub irradiance_map: Texture,
    pub prefilter_map: Texture,
    pub brdf_lut: Texture,
}

pub struct IblProcessor {
    device: Arc<ash::Device>,
    equirect_to_cubemap_pipeline: ComputePipeline,
    irradiance_pipeline: ComputePipeline,
    prefilter_pipeline: ComputePipeline,
    brdf_pipeline: ComputePipeline,

    // Layouts
    compute_dsl: vk::DescriptorSetLayout,
    brdf_dsl: vk::DescriptorSetLayout,
    compute_pool: vk::DescriptorPool,
}

impl IblProcessor {
    pub fn new(device: Arc<ash::Device>) -> Result<Self> {
        // Load pre-compiled SPIR-V shaders
        let equirect_spv =
            include_bytes!(concat!(env!("OUT_DIR"), "/equirect_to_cubemap.comp.spv"));
        let irradiance_spv =
            include_bytes!(concat!(env!("OUT_DIR"), "/irradiance_convolution.comp.spv"));
        let prefilter_spv = include_bytes!(concat!(env!("OUT_DIR"), "/prefilter_env_map.comp.spv"));
        let brdf_spv = include_bytes!(concat!(env!("OUT_DIR"), "/gen_brdf_lut.comp.spv"));

        let equirect_mod = unsafe {
            let code =
                ash::util::read_spv(&mut std::io::Cursor::new(equirect_spv)).map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to parse SPIR-V: {e}"))
                })?;
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };

        let irradiance_mod = unsafe {
            let code =
                ash::util::read_spv(&mut std::io::Cursor::new(irradiance_spv)).map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to parse SPIR-V: {e}"))
                })?;
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };

        let prefilter_mod = unsafe {
            let code =
                ash::util::read_spv(&mut std::io::Cursor::new(prefilter_spv)).map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to parse SPIR-V: {e}"))
                })?;
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };

        let brdf_mod = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(brdf_spv)).map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to parse SPIR-V: {e}"))
            })?;
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };

        // Create layouts
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let storage_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let brdf_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let compute_dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&[sampler_binding, storage_binding]),
                None,
            )?
        };

        let brdf_dsl = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[brdf_binding]),
                None,
            )?
        };

        // Create Pipelines
        let equirect_to_cubemap_pipeline = unsafe {
            ComputePipelineBuilder::new(Arc::clone(&device))
                .with_shader(equirect_mod)
                .add_set_layout(compute_dsl)
                .build()?
        };

        let irradiance_pipeline = unsafe {
            ComputePipelineBuilder::new(Arc::clone(&device))
                .with_shader(irradiance_mod)
                .add_set_layout(compute_dsl)
                .build()?
        };

        let prefilter_pipeline = unsafe {
            ComputePipelineBuilder::new(Arc::clone(&device))
                .with_shader(prefilter_mod)
                .add_set_layout(compute_dsl)
                .add_push_constant(vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: 4, // roughness
                })
                .build()?
        };

        let brdf_pipeline = unsafe {
            ComputePipelineBuilder::new(Arc::clone(&device))
                .with_shader(brdf_mod)
                .add_set_layout(brdf_dsl)
                .build()?
        };

        // Descriptor Pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 64, // General overhead
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 64,
            },
        ];

        let compute_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(64)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        // Cleanup temporary shader modules
        unsafe {
            device.destroy_shader_module(equirect_mod, None);
            device.destroy_shader_module(irradiance_mod, None);
            device.destroy_shader_module(prefilter_mod, None);
            device.destroy_shader_module(brdf_mod, None);
        }

        Ok(Self {
            device,
            equirect_to_cubemap_pipeline,
            irradiance_pipeline,
            prefilter_pipeline,
            brdf_pipeline,
            compute_dsl,
            brdf_dsl,
            compute_pool,
        })
    }

    /// Generates IBL maps from an equirectangular HDR texture.
    /// This is a synchronous operation intended for load-time.
    pub unsafe fn generate(
        &self,
        vulkan_device: &VulkanDevice,
        allocator: Arc<crate::vulkan::Allocator>,
        command_pool: vk::CommandPool,
        _queue: vk::Queue,
        equirect_hdr: &Texture,
    ) -> Result<IblBundle> {
        let env_res = 1024;
        let irr_res = 32;
        let pref_res = 128; // Usually enough for prefilter
        let brdf_res = 512;

        let hdr_format = vk::Format::R16G16B16A16_SFLOAT;

        // 1. Allocate resources
        let env_cubemap = Texture::create_empty_cubemap(
            Arc::clone(&allocator),
            Arc::clone(&self.device),
            env_res,
            1, // Mips handled later or not needed for base env
            hdr_format,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            Some("IBL_Environment_Cubemap"),
        )?;

        let irradiance_map = Texture::create_empty_cubemap(
            Arc::clone(&allocator),
            Arc::clone(&self.device),
            irr_res,
            1,
            hdr_format,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            Some("IBL_Irradiance_Map"),
        )?;

        let prefilter_mips = (pref_res as f32).log2().floor() as u32 + 1;
        let prefilter_map = Texture::create_empty_cubemap(
            Arc::clone(&allocator),
            Arc::clone(&self.device),
            pref_res,
            prefilter_mips,
            hdr_format,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            Some("IBL_Prefilter_Map"),
        )?;

        let brdf_lut = Texture::create_empty_2d(
            Arc::clone(&allocator),
            Arc::clone(&self.device),
            brdf_res,
            brdf_res,
            1,
            vk::Format::R16G16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            Some("IBL_BRDF_LUT"),
        )?;

        // 2. Dispatch Compute
        let mut dispatch_result: Result<()> = Ok(());
        vulkan_device.execute_single_use(command_pool, |cmd| {
            // Initial transition
            self.transition_to_general(cmd, env_cubemap.image.unwrap(), 1, 6);
            self.transition_to_general(cmd, irradiance_map.image.unwrap(), 1, 6);
            self.transition_to_general(cmd, prefilter_map.image.unwrap(), prefilter_mips, 6);
            self.transition_to_general(cmd, brdf_lut.image.unwrap(), 1, 1);

            // A. Equirect -> Cubemap
            if let Err(e) = self.dispatch_equirect(cmd, equirect_hdr, &env_cubemap) {
                dispatch_result = Err(e);
                return;
            }

            // Transition Env Map to SHADER_READ for sampling
            self.transition_to_read(cmd, env_cubemap.image.unwrap(), 1, 6);

            // B. Irradiance Convolution
            if let Err(e) = self.dispatch_irradiance(cmd, &env_cubemap, &irradiance_map) {
                dispatch_result = Err(e);
                return;
            }

            // C. Specular Prefilter
            if let Err(e) =
                self.dispatch_prefilter(cmd, &env_cubemap, &prefilter_map, prefilter_mips)
            {
                dispatch_result = Err(e);
                return;
            }

            // D. BRDF LUT
            if let Err(e) = self.dispatch_brdf(cmd, &brdf_lut) {
                dispatch_result = Err(e);
                return;
            }

            // Final transitions
            self.transition_to_read(cmd, irradiance_map.image.unwrap(), 1, 6);
            self.transition_to_read(cmd, prefilter_map.image.unwrap(), prefilter_mips, 6);
            self.transition_to_read(cmd, brdf_lut.image.unwrap(), 1, 1);
        })?;

        dispatch_result?;

        Ok(IblBundle {
            irradiance_map,
            prefilter_map,
            brdf_lut,
        })
    }

    unsafe fn dispatch_equirect(
        &self,
        cmd: vk::CommandBuffer,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        let set = self.allocate_and_update(
            self.compute_dsl,
            input.view.unwrap(),
            input.sampler.unwrap(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            output.view.unwrap(),
            vk::ImageLayout::GENERAL,
        )?;

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.equirect_to_cubemap_pipeline.handle(),
        );
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.equirect_to_cubemap_pipeline.layout(),
            0,
            &[set],
            &[],
        );

        let res = output.image.map(|_img| 1024).unwrap_or(1024);
        self.device.cmd_dispatch(cmd, res / 32, res / 32, 6);
        Ok(())
    }

    unsafe fn dispatch_irradiance(
        &self,
        cmd: vk::CommandBuffer,
        input: &Texture,
        output: &Texture,
    ) -> Result<()> {
        let set = self.allocate_and_update(
            self.compute_dsl,
            input.view.unwrap(),
            input.sampler.unwrap(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            output.view.unwrap(),
            vk::ImageLayout::GENERAL,
        )?;

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.irradiance_pipeline.handle(),
        );
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.irradiance_pipeline.layout(),
            0,
            &[set],
            &[],
        );

        self.device.cmd_dispatch(cmd, 32 / 32, 32 / 32, 6);
        Ok(())
    }

    unsafe fn dispatch_prefilter(
        &self,
        cmd: vk::CommandBuffer,
        input: &Texture,
        output: &Texture,
        mips: u32,
    ) -> Result<()> {
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.prefilter_pipeline.handle(),
        );

        for mip in 0..mips {
            // Create temporary view for the specific mip level for storage write
            let view_info = vk::ImageViewCreateInfo::default()
                .image(output.image.unwrap())
                .view_type(vk::ImageViewType::CUBE)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });
            let mip_view = self.device.create_image_view(&view_info, None)?;

            let set = self.allocate_and_update(
                self.compute_dsl,
                input.view.unwrap(),
                input.sampler.unwrap(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mip_view,
                vk::ImageLayout::GENERAL,
            )?;

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.prefilter_pipeline.layout(),
                0,
                &[set],
                &[],
            );

            let roughness = mip as f32 / (mips - 1) as f32;
            self.device.cmd_push_constants(
                cmd,
                self.prefilter_pipeline.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&roughness),
            );

            let mip_res = (128 >> mip).max(1);
            self.device
                .cmd_dispatch(cmd, (mip_res + 31) / 32, (mip_res + 31) / 32, 6);

            self.device.destroy_image_view(mip_view, None);
        }

        Ok(())
    }

    unsafe fn dispatch_brdf(&self, cmd: vk::CommandBuffer, output: &Texture) -> Result<()> {
        let set = self.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.compute_pool)
                .set_layouts(&[self.brdf_dsl]),
        )?[0];

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(output.view.unwrap())
            .image_layout(vk::ImageLayout::GENERAL);

        self.device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&[image_info])],
            &[],
        );

        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.brdf_pipeline.handle(),
        );
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.brdf_pipeline.layout(),
            0,
            &[set],
            &[],
        );

        self.device.cmd_dispatch(cmd, 512 / 16, 512 / 16, 1);
        Ok(())
    }

    unsafe fn allocate_and_update(
        &self,
        layout: vk::DescriptorSetLayout,
        sampler_view: vk::ImageView,
        sampler: vk::Sampler,
        sampler_layout: vk::ImageLayout,
        storage_view: vk::ImageView,
        storage_layout: vk::ImageLayout,
    ) -> Result<vk::DescriptorSet> {
        let set = self.device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.compute_pool)
                .set_layouts(&[layout]),
        )?[0];

        let sampler_info = vk::DescriptorImageInfo::default()
            .image_view(sampler_view)
            .sampler(sampler)
            .image_layout(sampler_layout);

        let storage_info = vk::DescriptorImageInfo::default()
            .image_view(storage_view)
            .image_layout(storage_layout);

        self.device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&[sampler_info]),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[storage_info]),
            ],
            &[],
        );

        Ok(set)
    }

    unsafe fn transition_to_general(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        mips: u32,
        layers: u32,
    ) {
        self.transition_layout(
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
            mips,
            layers,
        );
    }

    unsafe fn transition_to_read(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        mips: u32,
        layers: u32,
    ) {
        self.transition_layout(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
            mips,
            layers,
        );
    }

    unsafe fn transition_layout(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        mips: u32,
        layers: u32,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mips,
                base_array_layer: 0,
                layer_count: layers,
            });

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

impl Drop for IblProcessor {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.compute_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.compute_dsl, None);
            self.device
                .destroy_descriptor_set_layout(self.brdf_dsl, None);
        }
    }
}
