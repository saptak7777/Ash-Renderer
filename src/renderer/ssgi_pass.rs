//! Screen-Space Global Illumination (SSGI)
//!
//! Implements simplified Lumen-style indirect lighting using screen-space techniques.
//! Key features:
//! - Screen-space ray marching against depth buffer
//! - Temporal accumulation with history buffer
//! - Configurable quality levels (ray count, step count)
//! - Denoising via spatial and temporal filtering

use ash::vk;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// SSGI quality presets
#[derive(Clone, Copy, Debug, Default)]
pub enum SsgiQuality {
    /// 4 rays, 8 steps - Fastest
    Low,
    /// 8 rays, 16 steps - Balanced
    #[default]
    Medium,
    /// 16 rays, 32 steps - Quality
    High,
    /// 32 rays, 64 steps - Ultra
    Ultra,
}

impl SsgiQuality {
    /// Get ray count
    pub fn ray_count(&self) -> u32 {
        match self {
            SsgiQuality::Low => 4,
            SsgiQuality::Medium => 8,
            SsgiQuality::High => 16,
            SsgiQuality::Ultra => 32,
        }
    }

    /// Get step count per ray
    pub fn step_count(&self) -> u32 {
        match self {
            SsgiQuality::Low => 8,
            SsgiQuality::Medium => 16,
            SsgiQuality::High => 32,
            SsgiQuality::Ultra => 64,
        }
    }
}

/// Push constants for SSGI compute shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsgiPushConstants {
    /// Inverse view-projection for ray reconstruction
    pub inv_view_proj: [[f32; 4]; 4],
    /// Screen dimensions
    pub screen_size: [f32; 2],
    /// Ray marching parameters
    pub ray_count: u32,
    pub step_count: u32,
    /// Max ray distance in world space
    pub max_distance: f32,
    /// GI intensity
    pub intensity: f32,
    /// Frame index for temporal jitter
    pub frame_index: u32,
    /// History blend factor
    pub history_weight: f32,
}

/// Screen-Space GI pass
pub struct SsgiPass {
    device: Arc<ash::Device>,

    // GI output (R11G11B10_UFLOAT for compact HDR)
    gi_img: vk::Image,
    gi_alloc: Option<vk_mem::Allocation>,
    gi_view: vk::ImageView,

    // Ping-pong history for temporal accumulation
    history_imgs: [vk::Image; 2],
    history_allocs: [Option<vk_mem::Allocation>; 2],
    history_vs: [vk::ImageView; 2],

    // Pipelines
    gi_pipeline: vk::Pipeline,
    gi_layout: vk::PipelineLayout,

    denoise_pipeline: vk::Pipeline,
    denoise_layout: vk::PipelineLayout,

    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    desc_sets: [vk::DescriptorSet; 2],

    sampler: vk::Sampler,

    // State
    width: u32,
    height: u32,
    quality: SsgiQuality,
    intensity: f32,
    max_distance: f32,
    frame_index: u32,

    initialized: bool,
}

impl SsgiPass {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            gi_img: vk::Image::null(),
            gi_alloc: None,
            gi_view: vk::ImageView::null(),
            history_imgs: [vk::Image::null(); 2],
            history_allocs: [None, None],
            history_vs: [vk::ImageView::null(); 2],
            gi_pipeline: vk::Pipeline::null(),
            gi_layout: vk::PipelineLayout::null(),
            denoise_pipeline: vk::Pipeline::null(),
            denoise_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            desc_layout: vk::DescriptorSetLayout::null(),
            desc_sets: [vk::DescriptorSet::null(); 2],
            sampler: vk::Sampler::null(),
            width: 0,
            height: 0,
            quality: SsgiQuality::default(),
            intensity: 1.0,
            max_distance: 50.0,
            frame_index: 0,
            initialized: false,
        }
    }

    /// # Safety
    /// Device and allocator must stay valid for the lifetime of this pass.
    pub unsafe fn init(
        &mut self,
        alloc: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
        quality: SsgiQuality,
    ) {
        if self.initialized {
            return;
        }

        // Half-res GI for bandwidth savings
        self.width = width / 2;
        self.height = height / 2;
        self.quality = quality;

        // Note: failures here are considered fatal since we can't recover
        // without the primary GI buffers.
        self.create_gi_image(alloc)
            .expect("SSGI: GI image allocation failed");
        self.create_history_images(alloc)
            .expect("SSGI: History buffer allocation failed");
        self.create_sampler()
            .expect("SSGI: Sampler creation failed");
        self.create_descriptors()
            .expect("SSGI: Descriptor setup failed");
        self.create_pipelines()
            .expect("SSGI: Pipeline compilation failed");

        self.initialized = true;
    }

    /// # Safety
    /// This function creates Vulkan resources.
    unsafe fn create_gi_image(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        // QUALITY: R11G11B10_UFLOAT is perfect for diffuse GI as it fits in 32bpp
        // while preserving high dynamic range without the alpha channel.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B10G11R11_UFLOAT_PACK32)
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
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = alloc
            .create_image(&image_info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("GI image: {e:?}")))?;

        self.gi_img = image;
        self.gi_alloc = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.gi_img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::B10G11R11_UFLOAT_PACK32)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.gi_view = self.device.create_image_view(&view_info, None)?;

        Ok(())
    }

    /// # Safety
    /// This function creates Vulkan resources.
    unsafe fn create_history_images(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        for i in 0..2 {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B10G11R11_UFLOAT_PACK32)
                .extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            let (image, allocation) = alloc
                .create_image(&image_info, &alloc_info)
                .map_err(|e| crate::AshError::VulkanError(format!("GI history {i}: {e:?}")))?;

            self.history_imgs[i] = image;
            self.history_allocs[i] = Some(allocation);

            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.history_imgs[i])
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B10G11R11_UFLOAT_PACK32)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );

            self.history_vs[i] = self.device.create_image_view(&view_info, None)?;
        }

        Ok(())
    }

    unsafe fn create_sampler(&mut self) -> Result<()> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

        self.sampler = self.device.create_sampler(&info, None)?;
        Ok(())
    }

    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // [0..3]: Depth, Normal, Albedo, History
        // [4]: Output
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
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.desc_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 8,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);

        self.descriptor_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        let layouts = [self.desc_layout, self.desc_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        self.desc_sets = [sets[0], sets[1]];

        Ok(())
    }

    /// Create SSGI compute pipelines
    unsafe fn create_pipelines(&mut self) -> Result<()> {
        // GI main pass
        let gi_shader_path = std::path::Path::new("shaders/ssgi.spv");
        if gi_shader_path.exists() {
            let shader_code = std::fs::read(gi_shader_path)?;

            let shader_module_info =
                vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&shader_code));
            let shader_module = self
                .device
                .create_shader_module(&shader_module_info, None)?;

            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<SsgiPushConstants>() as u32);

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&self.desc_layout))
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));

            self.gi_layout = self.device.create_pipeline_layout(&layout_info, None)?;

            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(self.gi_layout);

            let pipelines = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?;

            self.gi_pipeline = pipelines[0];
            self.device.destroy_shader_module(shader_module, None);

            log::info!("SSGI: GI pipeline created");
        } else {
            log::warn!("SSGI: ssgi.spv not found, pipeline creation skipped");
        }

        // Denoise pass
        let denoise_shader_path = std::path::Path::new("shaders/ssgi_denoise.spv");
        if denoise_shader_path.exists() {
            let shader_code = std::fs::read(denoise_shader_path)?;

            let shader_module_info =
                vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&shader_code));
            let shader_module = self
                .device
                .create_shader_module(&shader_module_info, None)?;

            // Reuse same layout for simplicity
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&self.desc_layout));

            self.denoise_layout = self.device.create_pipeline_layout(&layout_info, None)?;

            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");

            let pipeline_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage_info)
                .layout(self.denoise_layout);

            let pipelines = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?;

            self.denoise_pipeline = pipelines[0];
            self.device.destroy_shader_module(shader_module, None);

            log::info!("SSGI: Denoise pipeline created");
        }

        Ok(())
    }

    /// Compute SSGI for the current frame
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn compute_gi(
        &mut self,
        cmd: vk::CommandBuffer,
        depth_view: vk::ImageView,
        normal_view: vk::ImageView,
        albedo_view: vk::ImageView,
        inv_view_proj: glam::Mat4,
    ) -> Result<()> {
        if !self.initialized || self.gi_pipeline == vk::Pipeline::null() {
            return Ok(());
        }

        let curr_idx = (self.frame_index % 2) as usize;
        let prev_idx = ((self.frame_index + 1) % 2) as usize;
        let desc_set = self.desc_sets[curr_idx];

        // 1. Update descriptors
        let sampler_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let depth_info = sampler_info.image_view(depth_view);
        let normal_info = sampler_info.image_view(normal_view);
        let albedo_info = sampler_info.image_view(albedo_view);
        let history_info = sampler_info.image_view(self.history_vs[prev_idx]);

        let output_info = vk::DescriptorImageInfo::default()
            .image_view(self.gi_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&normal_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&albedo_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&history_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
        ];

        self.device.update_descriptor_sets(&writes, &[]);

        // 2. GI Dispatch
        let barrier = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(self.gi_img)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );

        // 3. Dispatch Compute
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.gi_pipeline);

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.gi_layout,
            0,
            std::slice::from_ref(&desc_set),
            &[],
        );

        let push_constants = self.push_constants(inv_view_proj);
        self.device.cmd_push_constants(
            cmd,
            self.gi_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        let group_x = self.width.div_ceil(8);
        let group_y = self.height.div_ceil(8);
        self.device.cmd_dispatch(cmd, group_x, group_y, 1);

        // 4. Copy result to history
        let src_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(self.gi_img)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        let dst_barrier = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(self.history_imgs[curr_idx])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[src_barrier, dst_barrier],
        );

        let copy_region = vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            });

        self.device.cmd_copy_image(
            cmd,
            self.gi_img,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.history_imgs[curr_idx],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&copy_region),
        );

        // 5. Final transition: history to shader read (for compositing and next frame)
        let final_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.history_imgs[curr_idx])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[final_barrier],
        );

        Ok(())
    }

    /// Returns latest history buffer
    pub fn gi_view(&self) -> vk::ImageView {
        if !self.initialized {
            return vk::ImageView::null();
        }
        self.history_vs[(self.frame_index % 2) as usize]
    }

    pub fn gi_image(&self) -> vk::Image {
        self.gi_img
    }

    /// Get current quality preset
    pub fn quality(&self) -> SsgiQuality {
        self.quality
    }

    /// Set GI intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.max(0.0);
    }

    /// Get GI intensity
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Set max ray distance
    pub fn set_max_distance(&mut self, distance: f32) {
        self.max_distance = distance.max(1.0);
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get push constants for current frame
    pub fn push_constants(&self, inv_view_proj: glam::Mat4) -> SsgiPushConstants {
        SsgiPushConstants {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            screen_size: [self.width as f32, self.height as f32],
            ray_count: self.quality.ray_count(),
            step_count: self.quality.step_count(),
            max_distance: self.max_distance,
            intensity: self.intensity,
            frame_index: self.frame_index,
            history_weight: 0.9,
        }
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        if self.gi_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.gi_view, None);
        }
        if let Some(mut a) = self.gi_alloc.take() {
            allocator.destroy_image(self.gi_img, &mut a);
        }

        for i in 0..2 {
            if self.history_vs[i] != vk::ImageView::null() {
                self.device.destroy_image_view(self.history_vs[i], None);
            }
            if let Some(mut a) = self.history_allocs[i].take() {
                allocator.destroy_image(self.history_imgs[i], &mut a);
            }
        }

        if self.sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.sampler, None);
        }

        if self.gi_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.gi_pipeline, None);
            self.device.destroy_pipeline_layout(self.gi_layout, None);
        }

        if self.denoise_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.denoise_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.denoise_layout, None);
        }

        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
        }

        self.initialized = false;
    }
}
