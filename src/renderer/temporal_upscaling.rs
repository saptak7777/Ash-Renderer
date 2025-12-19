//! Temporal Super-Resolution (TSR)
//!
//! Implements temporal upscaling for improved image quality at reduced rendering cost.
//! Key features:
//! - Jittered projection matrices (Halton sequence)
//! - Motion vector tracking for temporal reprojection
//! - History buffer accumulation with AABB clamping
//! - Configurable upscale factors (1.33x, 1.5x, 2.0x)

use ash::vk;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// Upscale quality presets
#[derive(Clone, Copy, Debug, Default)]
pub enum VsrQuality {
    /// 2.0x upscale (50% render resolution) - Fastest
    Performance,
    /// 1.5x upscale (67% render resolution) - Balanced
    #[default]
    Balanced,
    /// 1.33x upscale (75% render resolution) - Quality
    Quality,
    /// 1.0x (native) - No upscaling, TAA only
    Native,
}

impl VsrQuality {
    /// Get the upscale factor
    pub fn factor(&self) -> f32 {
        match self {
            VsrQuality::Performance => 2.0,
            VsrQuality::Balanced => 1.5,
            VsrQuality::Quality => 1.33,
            VsrQuality::Native => 1.0,
        }
    }

    /// Get render resolution from display resolution
    pub fn render_size(&self, display_width: u32, display_height: u32) -> (u32, u32) {
        let factor = self.factor();
        (
            (display_width as f32 / factor) as u32,
            (display_height as f32 / factor) as u32,
        )
    }
}

/// Halton sequence generator for jittered sampling
pub struct HaltonSequence {
    index: u32,
    base2: Vec<f32>,
    base3: Vec<f32>,
}

impl HaltonSequence {
    /// Create a new Halton sequence generator
    pub fn new(max_samples: usize) -> Self {
        let mut base2 = Vec::with_capacity(max_samples);
        let mut base3 = Vec::with_capacity(max_samples);

        for i in 1..=max_samples {
            base2.push(Self::halton(i as u32, 2));
            base3.push(Self::halton(i as u32, 3));
        }

        Self {
            index: 0,
            base2,
            base3,
        }
    }

    /// Generate Halton sequence value
    fn halton(mut index: u32, base: u32) -> f32 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f32;

        while index > 0 {
            result += f * (index % base) as f32;
            index /= base;
            f /= base as f32;
        }

        result
    }

    /// Get next jitter offset in [-0.5, 0.5] range
    pub fn next_sample(&mut self) -> (f32, f32) {
        let idx = self.index as usize % self.base2.len();
        self.index = self.index.wrapping_add(1);
        (self.base2[idx] - 0.5, self.base3[idx] - 0.5)
    }

    /// Reset the sequence
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

/// Push constants for TSR upscale shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TsrPushConstants {
    /// Current jitter offset (applied to projection)
    pub jitter: [f32; 2],
    /// Screen dimensions (render resolution)
    pub render_size: [f32; 2],
    /// Screen dimensions (display resolution)
    pub display_size: [f32; 2],
    /// History blend factor (0.9 = 90% history)
    pub history_weight: f32,
    /// Frame index for temporal variation
    pub frame_index: u32,
}

/// Temporal Super-Resolution pass
pub struct VsrPass {
    device: Arc<ash::Device>,

    // Motion vectors image (R16G16_SFLOAT)
    motion_image: vk::Image,
    motion_allocation: Option<vk_mem::Allocation>,
    motion_view: vk::ImageView,

    // History buffers (ping-pong)
    history_images: [vk::Image; 2],
    history_allocations: [Option<vk_mem::Allocation>; 2],
    history_views: [vk::ImageView; 2],

    // Upscale compute pipeline
    upscale_pipeline: vk::Pipeline,
    upscale_layout: vk::PipelineLayout,

    // Descriptor resources
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_sets: [vk::DescriptorSet; 2],

    // Sampler for linear filtering
    linear_sampler: vk::Sampler,

    // State
    render_width: u32,
    render_height: u32,
    display_width: u32,
    display_height: u32,
    quality: VsrQuality,
    halton: HaltonSequence,
    frame_index: u32,

    initialized: bool,
}

impl VsrPass {
    /// Create a new TSR pass (uninitialized)
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            motion_image: vk::Image::null(),
            motion_allocation: None,
            motion_view: vk::ImageView::null(),
            history_images: [vk::Image::null(); 2],
            history_allocations: [None, None],
            history_views: [vk::ImageView::null(); 2],
            upscale_pipeline: vk::Pipeline::null(),
            upscale_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: [vk::DescriptorSet::null(); 2],
            linear_sampler: vk::Sampler::null(),
            render_width: 0,
            render_height: 0,
            display_width: 0,
            display_height: 0,
            quality: VsrQuality::default(),
            halton: HaltonSequence::new(16),
            frame_index: 0,
            initialized: false,
        }
    }

    /// Initialize TSR resources
    ///
    /// # Safety
    /// Allocator must be valid.
    pub unsafe fn initialize(
        &mut self,
        allocator: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        display_width: u32,
        display_height: u32,
        quality: VsrQuality,
    ) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        self.display_width = display_width;
        self.display_height = display_height;
        self.quality = quality;

        let (rw, rh) = quality.render_size(display_width, display_height);
        self.render_width = rw;
        self.render_height = rh;

        log::info!(
            "TSR: Initializing {}x{} -> {}x{} ({}x upscale)",
            rw,
            rh,
            display_width,
            display_height,
            quality.factor()
        );

        // Create motion vector image
        self.create_motion_image(allocator)?;

        // Create history buffers
        self.create_history_images(allocator)?;

        // Create sampler
        self.create_sampler()?;

        // Create descriptors
        self.create_descriptors()?;

        // Create upscale pipeline
        self.create_pipeline()?;

        self.initialized = true;
        Ok(())
    }

    /// Create motion vector image
    unsafe fn create_motion_image(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16_SFLOAT)
            .extent(vk::Extent3D {
                width: self.render_width,
                height: self.render_height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = allocator
            .create_image(&image_info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Motion image: {e:?}")))?;

        self.motion_image = image;
        self.motion_allocation = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.motion_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R16G16_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.motion_view = self.device.create_image_view(&view_info, None)?;

        Ok(())
    }

    /// Create history buffer images (ping-pong for temporal accumulation)
    unsafe fn create_history_images(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        for i in 0..2 {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .extent(vk::Extent3D {
                    width: self.display_width,
                    height: self.display_height,
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
                .map_err(|e| crate::AshError::VulkanError(format!("History image {i}: {e:?}")))?;

            self.history_images[i] = image;
            self.history_allocations[i] = Some(allocation);

            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.history_images[i])
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
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
        // 0: Current frame color (sampled)
        // 1: Motion vectors (sampled)
        // 2: Depth buffer (sampled)
        // 3: History buffer (sampled)
        // 4: Output buffer (storage)
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

    /// Create upscale compute pipeline
    unsafe fn create_pipeline(&mut self) -> Result<()> {
        let shader_path = std::path::Path::new("shaders/tsr_upscale.spv");

        // Check if shader exists, if not skip pipeline creation
        if !shader_path.exists() {
            log::warn!("TSR: tsr_upscale.spv not found, pipeline creation skipped");
            return Ok(());
        }

        let shader_code = std::fs::read(shader_path)?;

        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&shader_code));
        let shader_module = self
            .device
            .create_shader_module(&shader_module_info, None)?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<TsrPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.descriptor_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.upscale_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.upscale_layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?;

        self.upscale_pipeline = pipelines[0];
        self.device.destroy_shader_module(shader_module, None);

        log::info!("TSR: Upscale pipeline created");
        Ok(())
    }

    /// Get current jitter offset for projection matrix
    pub fn get_jitter(&mut self) -> (f32, f32) {
        self.halton.next_sample()
    }

    /// Apply jitter to projection matrix
    pub fn jitter_projection(&mut self, projection: glam::Mat4) -> glam::Mat4 {
        let (jx, jy) = self.get_jitter();

        // Scale jitter to pixel size
        let pixel_jitter_x = jx / self.render_width as f32;
        let pixel_jitter_y = jy / self.render_height as f32;

        // Apply sub-pixel jitter via translation in clip space
        let jitter_matrix = glam::Mat4::from_translation(glam::Vec3::new(
            pixel_jitter_x * 2.0,
            pixel_jitter_y * 2.0,
            0.0,
        ));

        jitter_matrix * projection
    }

    /// Get motion vector image view for render pass attachment
    pub fn motion_view(&self) -> vk::ImageView {
        self.motion_view
    }

    /// Get motion image for layout transitions
    pub fn motion_image(&self) -> vk::Image {
        self.motion_image
    }

    /// Get render resolution
    pub fn render_size(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }

    /// Get display resolution
    pub fn display_size(&self) -> (u32, u32) {
        (self.display_width, self.display_height)
    }

    /// Get current quality preset
    pub fn quality(&self) -> VsrQuality {
        self.quality
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Get current frame index (for ping-pong buffer selection)
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        // Destroy motion vector resources
        if self.motion_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.motion_view, None);
        }
        if let Some(mut alloc) = self.motion_allocation.take() {
            allocator.destroy_image(self.motion_image, &mut alloc);
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

        // Destroy pipeline
        if self.upscale_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.upscale_pipeline, None);
        }
        if self.upscale_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.upscale_layout, None);
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
        log::info!("TSR: Resources destroyed");
    }
}
