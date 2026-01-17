//! A-Trous Wavelet Denoiser
//!
//! Implements Unity-style multi-iteration bilateral filtering for SSGI.
//! Uses 5 iterations with exponentially increasing step sizes (1,2,4,8,16)
//! to efficiently denoise while preserving edges.

use ash::vk;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// Push constants for A-Trous denoising
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DenoisePushConstants {
    /// Step size for current iteration (1, 2, 4, 8, 16)
    pub step_size: u32,
    /// Edge-stopping weights
    pub depth_weight: f32,
    pub normal_weight: f32,
    pub luma_weight: f32,
    /// Screen dimensions
    pub screen_width: u32,
    pub screen_height: u32,
    /// Padding for alignment
    pub _padding: [u32; 2],
}

// Compile-time size verification (GPU expects exactly 32 bytes)
const _: () = assert!(std::mem::size_of::<DenoisePushConstants>() == 32);
const _: () = assert!(std::mem::align_of::<DenoisePushConstants>() == 16);

/// A-Trous wavelet denoiser for SSGI
pub struct ATrousDenoiser {
    device: Arc<ash::Device>,

    // 5 pipelines (one per iteration)
    pipelines: [vk::Pipeline; 5],
    pipeline_layout: vk::PipelineLayout,

    // Ping-pong buffers for iterative filtering
    temp_imgs: [vk::Image; 2],
    temp_allocs: [Option<vk_mem::Allocation>; 2],
    temp_views: [vk::ImageView; 2],

    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    desc_sets: Vec<vk::DescriptorSet>,

    sampler: vk::Sampler,

    // Dimensions
    width: u32,
    height: u32,

    // Denoising parameters
    depth_weight: f32,
    normal_weight: f32,
    luma_weight: f32,

    initialized: bool,
}

impl ATrousDenoiser {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            pipelines: [vk::Pipeline::null(); 5],
            pipeline_layout: vk::PipelineLayout::null(),
            temp_imgs: [vk::Image::null(); 2],
            temp_allocs: [None, None],
            temp_views: [vk::ImageView::null(); 2],
            descriptor_pool: vk::DescriptorPool::null(),
            desc_layout: vk::DescriptorSetLayout::null(),
            desc_sets: Vec::new(),
            sampler: vk::Sampler::null(),
            width: 0,
            height: 0,
            depth_weight: 0.1,
            normal_weight: 32.0,
            luma_weight: 4.0,
            initialized: false,
        }
    }

    /// Initialize denoiser
    ///
    /// # Safety
    /// Device and allocator must be valid
    pub unsafe fn init(
        &mut self,
        alloc: &vk_mem::Allocator,
        vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        self.width = width;
        self.height = height;

        self.create_temp_buffers(alloc)?;
        self.create_sampler()?;
        self.create_descriptors()?;
        self.create_pipelines(vulkan_device)?;

        self.initialized = true;
        log::info!("A-Trous denoiser initialized ({width}x{height})");
        Ok(())
    }

    unsafe fn create_temp_buffers(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let format = vk::Format::R16G16B16A16_SFLOAT;

        for i in 0..2 {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            let (image, allocation) = alloc.create_image(&image_info, &alloc_info)?;

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = self.device.create_image_view(&view_info, None)?;

            self.temp_imgs[i] = image;
            self.temp_allocs[i] = Some(allocation);
            self.temp_views[i] = view;
        }

        log::debug!("A-Trous: Created ping-pong buffers");
        Ok(())
    }

    unsafe fn create_sampler(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        self.sampler = self.device.create_sampler(&sampler_info, None)?;
        Ok(())
    }

    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // Descriptor layout: 4 bindings (input GI, depth, normals, output)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        self.desc_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        // Pool: 10 sets (5 iterations × 2 for ping-pong)
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 30, // 3 samplers × 10 sets
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 10, // 1 storage × 10 sets
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(10);

        self.descriptor_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        // Allocate 10 descriptor sets
        let layouts = vec![self.desc_layout; 10];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        self.desc_sets = self.device.allocate_descriptor_sets(&alloc_info)?;

        log::debug!("A-Trous: Created descriptors (10 sets)");
        Ok(())
    }

    unsafe fn create_pipelines(&mut self, _vulkan_device: &VulkanDevice) -> Result<()> {
        // Load shader
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/atrous_denoise.comp.spv"));

        let spv_code = ash::util::read_spv(&mut std::io::Cursor::new(shader_code))
            .map_err(|e| crate::AshError::VulkanError(e.to_string()))?;
        let shader_module = self
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv_code), None)?;

        // Push constant range
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<DenoisePushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.pipeline_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        // Create 5 identical pipelines (could be specialized later)
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        for i in 0..5 {
            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.pipeline_layout);

            let pipelines = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, e)| e)?;

            self.pipelines[i] = pipelines[0];
        }

        self.device.destroy_shader_module(shader_module, None);

        log::info!("A-Trous: Created 5 denoising pipelines");
        Ok(())
    }

    /// Denoise noisy GI using 5-iteration A-Trous filter
    ///
    /// # Safety
    /// Command buffer must be in recording state
    pub unsafe fn denoise(
        &mut self,
        cmd: vk::CommandBuffer,
        noisy: vk::ImageView,
        depth: vk::ImageView,
        normals: vk::ImageView,
    ) -> Result<vk::ImageView> {
        let mut current_input = noisy;

        // 5 iterations with exponentially increasing step sizes
        for i in 0..5 {
            let step_size = 1 << i; // 1, 2, 4, 8, 16
            let output_idx = i % 2;
            let output_view = self.temp_views[output_idx];

            // Update descriptors for this iteration
            self.update_descriptors(i, current_input, depth, normals, output_view)?;

            // Bind pipeline
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[i]);

            // Bind descriptor set
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.desc_sets[i]],
                &[],
            );

            // Push constants
            let consts = DenoisePushConstants {
                step_size,
                depth_weight: self.depth_weight,
                normal_weight: self.normal_weight,
                luma_weight: self.luma_weight,
                screen_width: self.width,
                screen_height: self.height,
                _padding: [0, 0],
            };

            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&consts),
            );

            // Dispatch compute
            let group_count_x = self.width.div_ceil(8);
            let group_count_y = self.height.div_ceil(8);
            self.device
                .cmd_dispatch(cmd, group_count_x, group_count_y, 1);

            // Memory barrier for next iteration
            if i < 4 {
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.temp_imgs[output_idx])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
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

            // Output of this iteration becomes input for next
            current_input = output_view;
        }

        // Return final denoised result
        Ok(current_input)
    }

    unsafe fn update_descriptors(
        &self,
        iteration: usize,
        input: vk::ImageView,
        depth: vk::ImageView,
        normals: vk::ImageView,
        output: vk::ImageView,
    ) -> Result<()> {
        let input_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(input)
            .sampler(self.sampler);

        let depth_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(depth)
            .sampler(self.sampler);

        let normals_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(normals)
            .sampler(self.sampler);

        let output_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(output);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[iteration])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&input_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[iteration])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[iteration])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&normals_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[iteration])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
        ];

        self.device.update_descriptor_sets(&writes, &[]);
        Ok(())
    }

    /// Set edge-stopping weights
    pub fn set_weights(&mut self, depth: f32, normal: f32, luma: f32) {
        self.depth_weight = depth;
        self.normal_weight = normal;
        self.luma_weight = luma;
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        // Destroy pipelines
        for pipeline in &mut self.pipelines {
            if *pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(*pipeline, None);
                *pipeline = vk::Pipeline::null();
            }
        }

        if self.pipeline_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.pipeline_layout = vk::PipelineLayout::null();
        }

        // Destroy temp buffers
        for i in 0..2 {
            if self.temp_views[i] != vk::ImageView::null() {
                self.device.destroy_image_view(self.temp_views[i], None);
                self.temp_views[i] = vk::ImageView::null();
            }

            if self.temp_imgs[i] != vk::Image::null() {
                if let Some(mut alloc) = self.temp_allocs[i].take() {
                    allocator.destroy_image(self.temp_imgs[i], &mut alloc);
                }
                self.temp_imgs[i] = vk::Image::null();
            }
        }

        // Destroy descriptors
        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }

        if self.desc_layout != vk::DescriptorSetLayout::null() {
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
            self.desc_layout = vk::DescriptorSetLayout::null();
        }

        if self.sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }

        self.initialized = false;
        log::info!("A-Trous denoiser destroyed");
    }
}

impl Drop for ATrousDenoiser {
    fn drop(&mut self) {
        if self.initialized {
            log::warn!("ATrousDenoiser dropped without calling destroy()");
        }
    }
}
