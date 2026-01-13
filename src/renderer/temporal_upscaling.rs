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

use crate::renderer::temporal_aa::{SharpeningMode, TaaQuality};
use crate::vulkan::VulkanDevice;
use crate::Result;

/// TAA performance and quality metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct TaaMetrics {
    /// Rejection rate (percentage of pixels rejected)
    pub rejection_rate: f32,
    /// Average history blend weight
    pub avg_blend_weight: f32,
    /// Ghosting score (0-1, lower is better)
    pub ghosting_score: f32,
    /// Shimmering score (0-1, lower is better)
    pub shimmering_score: f32,
}

#[derive(Debug)]
pub struct TaaQualityReport {
    pub quality_mode: TaaQuality,
    pub sharpening_mode: SharpeningMode,
    pub rejection_rate: f32,
    pub ghosting_score: f32,
    pub shimmering_score: f32,
}

impl std::fmt::Display for TaaQualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TAA Quality Report:\n\
             - Quality: {:?}\n\
             - Sharpening: {:?}\n\
             - Rejection Rate: {:.1}%\n\
             - Ghosting: {:.2} {}\n\
             - Shimmering: {:.2} {}",
            self.quality_mode,
            self.sharpening_mode,
            self.rejection_rate * 100.0,
            self.ghosting_score,
            if self.ghosting_score < 0.1 {
                "✅"
            } else {
                "⚠️"
            },
            self.shimmering_score,
            if self.shimmering_score < 0.15 {
                "✅"
            } else {
                "⚠️"
            }
        )
    }
}

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

/// Halton sequence for QMC jittering
pub struct HaltonSequence {
    u: u32, // current index
    b2: Vec<f32>,
    b3: Vec<f32>,
}

impl HaltonSequence {
    pub fn new(samples: usize) -> Self {
        let mut b2 = Vec::with_capacity(samples);
        let mut b3 = Vec::with_capacity(samples);

        for i in 1..=samples {
            b2.push(Self::phi(i as u32, 2));
            b3.push(Self::phi(i as u32, 3));
        }

        Self { u: 0, b2, b3 }
    }

    // Corput radical inverse in base b
    fn phi(mut i: u32, b: u32) -> f32 {
        let mut r = 0.0;
        let mut f = 1.0 / b as f32;
        while i > 0 {
            r += f * (i % b) as f32;
            i /= b;
            f /= b as f32;
        }
        r
    }

    /// Next jitter sample in [-0.5, 0.5]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> (f32, f32) {
        let idx = self.u as usize % self.b2.len();
        self.u = self.u.wrapping_add(1);
        (self.b2[idx] - 0.5, self.b3[idx] - 0.5)
    }

    pub fn reset(&mut self) {
        self.u = 0;
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
    /// Clamping gamma (1.0 Quality - 1.5 Responsive)
    pub clamping_gamma: f32,
    /// Velocity threshold
    pub velocity_threshold: f32,
    /// Anti-flicker toggle (1.0 on, 0.0 off)
    pub anti_flicker: f32,
    /// Padding for alignment
    pub padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SharpenPushConstants {
    pub strength: f32,
    pub padding1: f32,
    pub padding2: f32,
    pub padding3: f32,
}

/// Temporal Super-Resolution pass
pub struct VsrPass {
    device: Arc<ash::Device>,

    // Render-res buffers
    motion_img: vk::Image,
    motion_alloc: Option<vk_mem::Allocation>,
    motion_v: vk::ImageView,

    // Display-res history (ping-pong)
    history_imgs: [vk::Image; 2],
    history_allocs: [Option<vk_mem::Allocation>; 2],
    history_vs: [vk::ImageView; 2],

    upscale_pl: vk::Pipeline,
    upscale_layout: vk::PipelineLayout,

    descriptor_pool: vk::DescriptorPool,
    desc_layout: vk::DescriptorSetLayout,
    desc_sets: [vk::DescriptorSet; 2],

    sampler: vk::Sampler,

    // Dimensions
    render_w: u32,
    render_h: u32,
    display_w: u32,
    display_h: u32,

    quality: VsrQuality,
    halton: HaltonSequence,
    frame_idx: u32,

    // Sharpening resources
    sharpen_pl: vk::Pipeline,
    sharpen_layout: vk::PipelineLayout,
    sharpen_desc_layout: vk::DescriptorSetLayout,
    sharpen_pool: vk::DescriptorPool,
    sharpen_sets: [vk::DescriptorSet; 2],
    sharpened_img: vk::Image,
    sharpened_alloc: Option<vk_mem::Allocation>,
    sharpened_v: vk::ImageView,

    // Quality metrics
    metrics: TaaMetrics,

    // Metrics buffer for GPU feedback (AAA standard)
    metrics_buffer: vk::Buffer,
    metrics_alloc: Option<vk_mem::Allocation>,
    metrics_readback_buffer: vk::Buffer,
    metrics_readback_alloc: Option<vk_mem::Allocation>,

    initialized: bool,
}

impl VsrPass {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            motion_img: vk::Image::null(),
            motion_alloc: None,
            motion_v: vk::ImageView::null(),
            history_imgs: [vk::Image::null(); 2],
            history_allocs: [None, None],
            history_vs: [vk::ImageView::null(); 2],
            upscale_pl: vk::Pipeline::null(),
            upscale_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            desc_layout: vk::DescriptorSetLayout::null(),
            desc_sets: [vk::DescriptorSet::null(); 2],
            sampler: vk::Sampler::null(),
            render_w: 0,
            render_h: 0,
            display_w: 0,
            display_h: 0,
            quality: VsrQuality::default(),
            halton: HaltonSequence::new(16),
            frame_idx: 0,
            sharpen_pl: vk::Pipeline::null(),
            sharpen_layout: vk::PipelineLayout::null(),
            sharpen_desc_layout: vk::DescriptorSetLayout::null(),
            sharpen_pool: vk::DescriptorPool::null(),
            sharpen_sets: [vk::DescriptorSet::null(); 2],
            sharpened_img: vk::Image::null(),
            sharpened_alloc: None,
            sharpened_v: vk::ImageView::null(),
            metrics: TaaMetrics::default(),
            metrics_buffer: vk::Buffer::null(),
            metrics_alloc: None,
            metrics_readback_buffer: vk::Buffer::null(),
            metrics_readback_alloc: None,
            initialized: false,
        }
    }

    /// # Safety
    /// Allocator must be valid.
    pub unsafe fn init(
        &mut self,
        alloc: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        w: u32,
        h: u32,
        quality: VsrQuality,
    ) {
        if self.initialized {
            return;
        }

        self.display_w = w;
        self.display_h = h;
        self.quality = quality;

        let (rw, rh) = quality.render_size(w, h);
        self.render_w = rw;
        self.render_h = rh;

        // VSR resources are mandatory for TSR rendering paths
        self.create_motion_image(alloc)
            .expect("TSR: Motion buffer failed");
        self.create_history_images(alloc)
            .expect("TSR: History buffer failed");
        self.create_metrics_buffers(alloc)
            .expect("TSR: Metrics buffers failed");
        self.create_sampler().expect("TSR: Sampler failed");
        self.create_descriptors().expect("TSR: Descriptors failed");
        self.create_descriptors().expect("TSR: Descriptors failed");
        self.create_pipeline().expect("TSR: Pipeline failed");

        // Sharpening
        self.create_sharpening_resources(alloc)
            .expect("TSR: Sharpening resources failed");
        self.create_sharpening_pipeline()
            .expect("TSR: Sharpening pipeline failed");

        self.initialized = true;
    }

    unsafe fn create_motion_image(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16_SFLOAT)
            .extent(vk::Extent3D {
                width: self.render_w,
                height: self.render_h,
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

        let (img, allocation) = alloc
            .create_image(&info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Motion image: {e:?}")))?;

        self.motion_img = img;
        self.motion_alloc = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.motion_img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R16G16_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.motion_v = self.device.create_image_view(&view_info, None)?;
        Ok(())
    }

    unsafe fn create_history_images(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        for i in 0..2 {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .extent(vk::Extent3D {
                    width: self.display_w,
                    height: self.display_h,
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

            let (img, allocation) = alloc
                .create_image(&info, &alloc_info)
                .map_err(|e| crate::AshError::VulkanError(format!("History image {i}: {e:?}")))?;

            self.history_imgs[i] = img;
            self.history_allocs[i] = Some(allocation);

            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.history_imgs[i])
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
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

    unsafe fn create_metrics_buffers(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        // GPU-side metrics buffer (device local)
        let buffer_size = std::mem::size_of::<u32>() * 4; // 4 uint32s

        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (buffer, allocation) = alloc
            .create_buffer(&buffer_info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Metrics buffer: {e:?}")))?;

        self.metrics_buffer = buffer;
        self.metrics_alloc = Some(allocation);

        // CPU-side readback buffer (host visible)
        let readback_info = vk::BufferCreateInfo::default()
            .size(buffer_size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let readback_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferHost,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (readback_buffer, readback_allocation) = alloc
            .create_buffer(&readback_info, &readback_alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Readback buffer: {e:?}")))?;

        self.metrics_readback_buffer = readback_buffer;
        self.metrics_readback_alloc = Some(readback_allocation);

        Ok(())
    }

    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // [0..3]: Color, Motion, Depth, History
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
            // Binding 5: Metrics buffer (AAA standard)
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
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

    /// Create upscale compute pipeline
    unsafe fn create_pipeline(&mut self) -> Result<()> {
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/vsr_upscale.comp.spv"));

        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(shader_code));
        let shader_module = self
            .device
            .create_shader_module(&shader_module_info, None)?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<TsrPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.desc_layout))
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
        self.upscale_pl = pipelines[0];

        self.device.destroy_shader_module(shader_module, None);
        Ok(())
    }

    unsafe fn create_sharpening_resources(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: self.display_w,
                height: self.display_h,
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

        let (img, allocation) = alloc
            .create_image(&info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Sharpen image: {e:?}")))?;

        self.sharpened_img = img;
        self.sharpened_alloc = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.sharpened_img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.sharpened_v = self.device.create_image_view(&view_info, None)?;

        // Descriptor pool for sharpening
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2, // Input
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2, // Output
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);

        self.sharpen_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        // Layout: Binding 0 = Input, Binding 1 = Output
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.sharpen_desc_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        let layouts = [self.sharpen_desc_layout, self.sharpen_desc_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.sharpen_pool)
            .set_layouts(&layouts);

        self.sharpen_sets = self
            .device
            .allocate_descriptor_sets(&alloc_info)?
            .try_into()
            .unwrap();

        Ok(())
    }

    unsafe fn create_sharpening_pipeline(&mut self) -> Result<()> {
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/sharpen.comp.spv"));

        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(shader_code));
        let shader_module = self
            .device
            .create_shader_module(&shader_module_info, None)?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SharpenPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.sharpen_desc_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.sharpen_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.sharpen_layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?;

        self.sharpen_pl = pipelines[0];
        self.device.destroy_shader_module(shader_module, None);
        Ok(())
    }

    pub fn get_jitter(&mut self) -> (f32, f32) {
        self.halton.next()
    }

    pub fn jitter_projection(&mut self, projection: glam::Mat4) -> glam::Mat4 {
        let (jx, jy) = self.get_jitter();

        let dx = jx / self.render_w as f32;
        let dy = jy / self.render_h as f32;

        let mat = glam::Mat4::from_translation(glam::Vec3::new(dx * 2.0, dy * 2.0, 0.0));
        mat * projection
    }

    pub fn motion_view(&self) -> vk::ImageView {
        self.motion_v
    }

    pub fn motion_image(&self) -> vk::Image {
        self.motion_img
    }

    pub fn render_size(&self) -> (u32, u32) {
        (self.render_w, self.render_h)
    }

    pub fn display_size(&self) -> (u32, u32) {
        (self.display_w, self.display_h)
    }

    /// Get current quality preset
    pub fn quality(&self) -> VsrQuality {
        self.quality
    }

    /// Perform temporal upscaling
    ///
    /// # Safety
    /// command_buffer must be in a recording state. Image views must be valid.
    pub unsafe fn upscale(
        &mut self,
        command_buffer: vk::CommandBuffer,
        input_view: vk::ImageView,
        depth_view: vk::ImageView,
        motion_view: vk::ImageView,
        jitter: [f32; 2],
        config: &crate::renderer::temporal_aa::TaaConfig,
    ) -> Result<()> {
        if !self.initialized || self.upscale_pl == vk::Pipeline::null() {
            return Ok(());
        }

        let prev = (self.frame_idx % 2) as usize;
        let curr = ((self.frame_idx + 1) % 2) as usize;

        self.update_descriptor_set(curr, input_view, depth_view, motion_view, prev)?;

        // Image barrier for history and output
        let image_barriers = [
            vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[prev])
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                }),
            vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[curr])
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                }),
        ];

        self.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barriers,
        );

        // Bind pipeline and descriptor set
        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.upscale_pl,
        );

        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.upscale_layout,
            0,
            &[self.desc_sets[curr]],
            &[],
        );

        // Push constants
        let push_constants = TsrPushConstants {
            jitter,
            render_size: [self.render_w as f32, self.render_h as f32],
            display_size: [self.display_w as f32, self.display_h as f32],
            history_weight: if config.quality
                == crate::renderer::temporal_aa::TaaQuality::Responsive
            {
                0.7
            } else {
                0.95
            },
            frame_index: self.frame_idx,
            clamping_gamma: if config.quality
                == crate::renderer::temporal_aa::TaaQuality::Responsive
            {
                1.5
            } else {
                1.0
            },
            velocity_threshold: config.velocity_threshold,
            anti_flicker: if config.anti_flicker { 1.0 } else { 0.0 },
            padding: 0.0,
        };

        self.device.cmd_push_constants(
            command_buffer,
            self.upscale_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        // Dispatch compute (8x8 groups)
        let gx = self.display_w.div_ceil(8);
        let gy = self.display_h.div_ceil(8);
        self.device.cmd_dispatch(command_buffer, gx, gy, 1);

        // Sharpening Pass (Optional)
        if config.sharpening != crate::renderer::temporal_aa::SharpeningMode::None {
            // Barrier: Wait for VSR output (History[curr]) to be ready for reading
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[curr])
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                });

            // Transition output to GENERAL
            let out_barrier = vk::ImageMemoryBarrier::default()
                .image(self.sharpened_img)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                });

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier, out_barrier],
            );

            // Update Descriptor
            let sampler_info = vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(self.history_vs[curr])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let out_info = vk::DescriptorImageInfo::default()
                .image_view(self.sharpened_v)
                .image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sharpen_sets[curr])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sharpen_sets[curr])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&out_info)),
            ];
            self.device.update_descriptor_sets(&writes, &[]);

            // Dispatch
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.sharpen_pl,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.sharpen_layout,
                0,
                &[self.sharpen_sets[curr]],
                &[],
            );

            let pc = SharpenPushConstants {
                strength: config.sharpening.strength(),
                padding1: 0.0,
                padding2: 0.0,
                padding3: 0.0,
            };

            self.device.cmd_push_constants(
                command_buffer,
                self.sharpen_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );

            self.device.cmd_dispatch(command_buffer, gx, gy, 1);

            // Restore history image layout for next frame consistency
            let restore_barrier = vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[curr])
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                });

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[restore_barrier],
            );
        }

        Ok(())
    }

    unsafe fn update_descriptor_set(
        &self,
        set_idx: usize,
        input: vk::ImageView,
        depth: vk::ImageView,
        motion: vk::ImageView,
        hist_idx: usize,
    ) -> Result<()> {
        let sampler_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let input_info = [sampler_info.image_view(input)];
        let motion_info = [sampler_info.image_view(motion)];
        let depth_info = [sampler_info.image_view(depth)];
        let history_info = [sampler_info.image_view(self.history_vs[hist_idx])];

        let output_info = [vk::DescriptorImageInfo::default()
            .image_view(self.history_vs[(hist_idx + 1) % 2])
            .image_layout(vk::ImageLayout::GENERAL)];

        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.metrics_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[set_idx])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&input_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[set_idx])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&motion_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[set_idx])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[set_idx])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&history_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[set_idx])
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&output_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets[set_idx])
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info),
        ];

        self.device.update_descriptor_sets(&writes, &[]);
        Ok(())
    }

    /// Read back metrics from GPU and update local state
    ///
    /// This should be called at the start of the frame to read stats from the *previous* frame
    /// to avoid stalling the pipeline.
    pub unsafe fn readback_metrics(
        &mut self,
        cmd: vk::CommandBuffer,
        allocator: &vk_mem::Allocator,
    ) {
        if !self.initialized || self.metrics_buffer == vk::Buffer::null() {
            return;
        }

        // 1. Copy metrics from GPU buffer to CPU readback buffer
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: std::mem::size_of::<u32>() as u64 * 4,
        };

        // Barrier to ensure compute shader is done writing
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.metrics_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );

        self.device.cmd_copy_buffer(
            cmd,
            self.metrics_buffer,
            self.metrics_readback_buffer,
            &[region],
        );

        // Barrier to ensure transfer is done before host read
        let host_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.metrics_readback_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[host_barrier],
            &[],
        );

        // 2. Reset the GPU buffer for the next frame
        self.device
            .cmd_fill_buffer(cmd, self.metrics_buffer, 0, vk::WHOLE_SIZE, 0);

        // 3. Map memory and read results
        if let Some(alloc) = &self.metrics_readback_alloc {
            let info = allocator.get_allocation_info(alloc);
            let ptr = info.mapped_data;

            if !ptr.is_null() {
                let data = std::slice::from_raw_parts(ptr as *const u32, 4);

                let rejected = data[0];
                let total_weight_scaled = data[1];
                let total_pixels = data[2];

                if total_pixels > 0 {
                    self.metrics.rejection_rate = rejected as f32 / total_pixels as f32;
                    self.metrics.avg_blend_weight =
                        (total_weight_scaled as f32 / 1000.0) / total_pixels as f32;

                    // Synthetic scoring based on real metrics
                    self.metrics.ghosting_score = (1.0 - self.metrics.avg_blend_weight).max(0.0)
                        * 0.5
                        + self.metrics.rejection_rate * 0.5;
                    self.metrics.shimmering_score = self.metrics.rejection_rate * 2.0;
                }
            }
        }
    }

    /// Get current upscaled output view
    pub fn output_view(&self, sharpening_enabled: bool) -> vk::ImageView {
        if sharpening_enabled {
            self.sharpened_v
        } else {
            self.history_vs[(self.frame_idx % 2) as usize]
        }
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) {
        self.frame_idx = self.frame_idx.wrapping_add(1);
    }

    pub fn next_jitter(&mut self) -> (f32, f32) {
        if !self.initialized {
            return (0.0, 0.0);
        }
        self.halton.next()
    }

    /// Get current frame index (for ping-pong buffer selection)
    pub fn frame_index(&self) -> u32 {
        self.frame_idx
    }

    /// Get quality report (AAA standard)
    pub fn quality_report(
        &self,
        quality: TaaQuality,
        sharpening: SharpeningMode,
    ) -> TaaQualityReport {
        TaaQualityReport {
            quality_mode: quality,
            sharpening_mode: sharpening,
            rejection_rate: self.metrics.rejection_rate,
            ghosting_score: self.metrics.ghosting_score,
            shimmering_score: self.metrics.shimmering_score,
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

        if self.sharpened_v != vk::ImageView::null() {
            self.device.destroy_image_view(self.sharpened_v, None);
        }
        if let Some(mut a) = self.sharpened_alloc.take() {
            allocator.destroy_image(self.sharpened_img, &mut a);
        }
        if self.sharpen_pool != vk::DescriptorPool::null() {
            self.device.destroy_descriptor_pool(self.sharpen_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.sharpen_desc_layout, None);
        }
        if self.sharpen_pl != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.sharpen_pl, None);
            self.device
                .destroy_pipeline_layout(self.sharpen_layout, None);
        }

        if self.motion_v != vk::ImageView::null() {
            self.device.destroy_image_view(self.motion_v, None);
        }
        if let Some(mut a) = self.motion_alloc.take() {
            allocator.destroy_image(self.motion_img, &mut a);
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

        if self.upscale_pl != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.upscale_pl, None);
            self.device
                .destroy_pipeline_layout(self.upscale_layout, None);
        }

        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_layout, None);
        }

        if self.metrics_buffer != vk::Buffer::null() {
            self.device.destroy_buffer(self.metrics_buffer, None);
        }
        if let Some(mut a) = self.metrics_alloc.take() {
            allocator.destroy_buffer(self.metrics_buffer, &mut a);
        }

        if self.metrics_readback_buffer != vk::Buffer::null() {
            self.device
                .destroy_buffer(self.metrics_readback_buffer, None);
        }
        if let Some(mut a) = self.metrics_readback_alloc.take() {
            allocator.destroy_buffer(self.metrics_readback_buffer, &mut a);
        }

        self.initialized = false;
    }
}
