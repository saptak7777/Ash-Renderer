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

    // GI output image (R11G11B10_UFLOAT for compact HDR)
    gi_image: vk::Image,
    gi_allocation: Option<vk_mem::Allocation>,
    gi_view: vk::ImageView,

    // History buffer for temporal accumulation
    history_images: [vk::Image; 2],
    history_allocations: [Option<vk_mem::Allocation>; 2],
    history_views: [vk::ImageView; 2],

    // Compute pipeline for GI
    gi_pipeline: vk::Pipeline,
    gi_layout: vk::PipelineLayout,

    // Denoise pipeline (spatial filtering)
    denoise_pipeline: vk::Pipeline,
    denoise_layout: vk::PipelineLayout,

    // Descriptor resources
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: [vk::DescriptorSet; 2],

    // Sampler
    linear_sampler: vk::Sampler,

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
    /// Create a new SSGI pass (uninitialized)
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            gi_image: vk::Image::null(),
            gi_allocation: None,
            gi_view: vk::ImageView::null(),
            history_images: [vk::Image::null(); 2],
            history_allocations: [None, None],
            history_views: [vk::ImageView::null(); 2],
            gi_pipeline: vk::Pipeline::null(),
            gi_layout: vk::PipelineLayout::null(),
            denoise_pipeline: vk::Pipeline::null(),
            denoise_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: [vk::DescriptorSet::null(); 2],
            linear_sampler: vk::Sampler::null(),
            width: 0,
            height: 0,
            quality: SsgiQuality::default(),
            intensity: 1.0,
            max_distance: 50.0,
            frame_index: 0,
            initialized: false,
        }
    }

    /// Initialize SSGI resources
    ///
    /// # Safety
    /// Allocator must be valid.
    pub unsafe fn initialize(
        &mut self,
        allocator: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
        quality: SsgiQuality,
    ) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Half-resolution for performance
        self.width = width / 2;
        self.height = height / 2;
        self.quality = quality;

        log::info!(
            "SSGI: Initializing {}x{} ({} rays, {} steps)",
            self.width,
            self.height,
            quality.ray_count(),
            quality.step_count()
        );

        // Create GI output image
        self.create_gi_image(allocator)?;

        // Create history buffers
        self.create_history_images(allocator)?;

        // Create sampler
        self.create_sampler()?;

        // Create descriptors
        self.create_descriptors()?;

        // Create pipelines
        self.create_pipelines()?;

        self.initialized = true;
        Ok(())
    }

    /// Create GI output image
    unsafe fn create_gi_image(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

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

        let (image, allocation) = allocator
            .create_image(&image_info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("GI image: {e:?}")))?;

        self.gi_image = image;
        self.gi_allocation = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.gi_image)
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

    /// Create history buffer images
    unsafe fn create_history_images(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
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

            let (image, allocation) = allocator
                .create_image(&image_info, &alloc_info)
                .map_err(|e| crate::AshError::VulkanError(format!("GI history {i}: {e:?}")))?;

            self.history_images[i] = image;
            self.history_allocations[i] = Some(allocation);

            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.history_images[i])
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B10G11R11_UFLOAT_PACK32)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );

            self.history_views[i] = self.device.create_image_view(&view_info, None)?;
        }

        Ok(())
    }

    /// Create linear sampler
    unsafe fn create_sampler(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

        self.linear_sampler = self.device.create_sampler(&sampler_info, None)?;
        Ok(())
    }

    /// Create descriptor layout and pool
    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // Bindings:
        // 0: Depth buffer (sampled)
        // 1: Normal buffer (sampled)
        // 2: Albedo buffer (sampled)
        // 3: History buffer (sampled)
        // 4: GI output (storage)
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
        self.descriptor_layout = self
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

        let layouts = [self.descriptor_layout, self.descriptor_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        self.descriptor_sets = [sets[0], sets[1]];

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
                .set_layouts(std::slice::from_ref(&self.descriptor_layout))
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
                .set_layouts(std::slice::from_ref(&self.descriptor_layout));

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

    /// Get GI image view for compositing
    pub fn gi_view(&self) -> vk::ImageView {
        self.gi_view
    }

    /// Get GI image for layout transitions
    pub fn gi_image(&self) -> vk::Image {
        self.gi_image
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

        // Destroy GI image
        if self.gi_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.gi_view, None);
        }
        if let Some(mut alloc) = self.gi_allocation.take() {
            allocator.destroy_image(self.gi_image, &mut alloc);
        }

        // Destroy history buffers
        for i in 0..2 {
            if self.history_views[i] != vk::ImageView::null() {
                self.device.destroy_image_view(self.history_views[i], None);
            }
            if let Some(mut alloc) = self.history_allocations[i].take() {
                allocator.destroy_image(self.history_images[i], &mut alloc);
            }
        }

        // Destroy sampler
        if self.linear_sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.linear_sampler, None);
        }

        // Destroy pipelines
        if self.gi_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.gi_pipeline, None);
        }
        if self.gi_layout != vk::PipelineLayout::null() {
            self.device.destroy_pipeline_layout(self.gi_layout, None);
        }
        if self.denoise_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.denoise_pipeline, None);
        }
        if self.denoise_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.denoise_layout, None);
        }

        // Destroy descriptors
        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
        if self.descriptor_layout != vk::DescriptorSetLayout::null() {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }

        self.initialized = false;
        log::info!("SSGI: Resources destroyed");
    }
}
