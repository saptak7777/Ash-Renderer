//! Virtual Super Resolution (VSR)
//!
//! Implements temporal upscaling for improved image quality at reduced rendering cost.
//! Key features:
//! - Jittered projection matrices (Halton sequence)
//! - Motion vector tracking for temporal reprojection
//! - History buffer accumulation with velocity rejection
//! - Configurable upscale factors (1.33x, 1.5x, 2.0x)

use ash::vk;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VsrError {
    #[error("VSR not initialized")]
    NotInitialized,

    #[error("Invalid clamping gamma: {0} (must be 1.0-2.0)")]
    InvalidClampingGamma(f32),

    #[error("Invalid history weight: {0} (must be 0.0-1.0)")]
    InvalidHistoryWeight(f32),

    #[error("Invalid velocity threshold: {0} (must be 0.0-1.0)")]
    InvalidVelocityThreshold(f32),

    #[error("Invalid sharpening strength: {0} (must be 0.0-1.0)")]
    InvalidSharpeningStrength(f32),

    #[error("Invalid edge threshold: {0} (must be 0.01-1.0)")]
    InvalidEdgeThreshold(f32),

    #[error("Invalid adaptive flag: {0} (must be 0.0 or 1.0)")]
    InvalidAdaptiveFlag(f32),

    #[error("Sharpening not initialized")]
    SharpeningNotInitialized,

    #[error("Metrics readback failed: {0}")]
    MetricsReadbackFailed(String),

    #[error("Vulkan error: {0}")]
    Vulkan(#[from] crate::AshError),
}

/// Structured readback result from GPU metrics
#[derive(Debug, Clone, Default)]
pub struct VsrMetricsReadback {
    pub rejection_rate: f32,
    pub avg_blend_weight: f32,
    pub ghosting_score: f32,
    pub shimmering_score: f32,
    pub frame_time_ms: f32,
}

/// VSR performance and quality metrics
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VsrMetrics {
    pub rejection_rate: f32,
    pub avg_blend_weight: f32,
    pub ghosting_score: f32,
    pub shimmering_score: f32,
    pub total_time_ms: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityLevel {
    Poor,
    Good,
    Excellent,
}

#[derive(Debug, Clone, Copy)]
pub struct VsrQualityAssessment {
    pub ghosting: QualityLevel,
    pub shimmering: QualityLevel,
    pub overall: QualityLevel,
}

impl VsrMetrics {
    pub fn quality_assessment(&self) -> VsrQualityAssessment {
        let ghosting = if self.ghosting_score < 0.1 {
            QualityLevel::Excellent
        } else if self.ghosting_score < 0.2 {
            QualityLevel::Good
        } else {
            QualityLevel::Poor
        };

        let shimmering = if self.shimmering_score < 0.15 {
            QualityLevel::Excellent
        } else if self.shimmering_score < 0.3 {
            QualityLevel::Good
        } else {
            QualityLevel::Poor
        };

        VsrQualityAssessment {
            ghosting,
            shimmering,
            overall: ghosting.min(shimmering),
        }
    }
}

/// Upscale quality presets
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VsrQuality {
    /// 0.5x resolution - Fastest
    Low,
    /// 0.67x resolution - Balanced
    #[default]
    Medium,
    /// 0.75x resolution - Quality
    High,
}

impl VsrQuality {
    /// Get the upscale factor
    pub const fn factor(self) -> f32 {
        match self {
            Self::Low => 2.0,
            Self::Medium => 1.5,
            Self::High => 1.33,
        }
    }

    pub const fn clamping_gamma(self) -> f32 {
        match self {
            Self::Low => 1.5, // More responsive
            Self::Medium => 1.25,
            Self::High => 1.0, // More stable
        }
    }

    pub const fn history_weight(self) -> f32 {
        match self {
            Self::Low => 0.90,
            Self::Medium => 0.93,
            Self::High => 0.95,
        }
    }

    /// Get render resolution from display resolution
    pub fn render_size(self, display_width: u32, display_height: u32) -> (u32, u32) {
        let factor = self.factor();
        (
            (display_width as f32 / factor) as u32,
            (display_height as f32 / factor) as u32,
        )
    }

    pub const fn default_sharpening(self) -> SharpenConfig {
        match self {
            Self::Low => SharpenConfig {
                strength: 0.6, // More sharpening (lower res needs it)
                edge_threshold: 0.12,
                adaptive: true,
            },
            Self::Medium => SharpenConfig {
                strength: 0.5,
                edge_threshold: 0.12,
                adaptive: true,
            },
            Self::High => SharpenConfig {
                strength: 0.4, // Less sharpening (higher res needs less)
                edge_threshold: 0.15,
                adaptive: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResolutionInfo {
    pub render: (u32, u32),
    pub display: (u32, u32),
    pub factor: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct VsrQualityReport {
    pub resolution: ResolutionInfo,
    pub quality: VsrQuality,
    pub metrics: VsrMetrics,
    pub assessment: VsrQualityAssessment,
}

impl std::fmt::Display for VsrQualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "VSR Quality Report:")?;
        writeln!(
            f,
            "  Resolution: {}x{} -> {}x{} ({:.2}x upscale)",
            self.resolution.render.0,
            self.resolution.render.1,
            self.resolution.display.0,
            self.resolution.display.1,
            self.resolution.factor
        )?;
        writeln!(f, "  Quality Preset: {:?}", self.quality)?;
        writeln!(f, "  Pass Time: {:.2}ms", self.metrics.total_time_ms)?;
        writeln!(
            f,
            "  Rejection Rate: {:.1}%",
            self.metrics.rejection_rate * 100.0
        )?;
        writeln!(
            f,
            "  Ghosting Score: {:.2} ({:?})",
            self.metrics.ghosting_score, self.assessment.ghosting
        )?;
        writeln!(
            f,
            "  Shimmering Score: {:.2} ({:?})",
            self.metrics.shimmering_score, self.assessment.shimmering
        )?;
        writeln!(f, "  Overall Status: {:?}", self.assessment.overall)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharpenConfig {
    pub strength: f32,       // 0.0 - 1.0
    pub edge_threshold: f32, // 0.01 - 1.0
    pub adaptive: bool,      // Enable contrast-adaptive logic
}

impl SharpenConfig {
    pub const fn validate(&self) -> std::result::Result<(), VsrError> {
        if self.strength < 0.0 || self.strength > 1.0 {
            return Err(VsrError::InvalidSharpeningStrength(self.strength));
        }
        if self.edge_threshold < 0.01 || self.edge_threshold > 1.0 {
            return Err(VsrError::InvalidEdgeThreshold(self.edge_threshold));
        }
        Ok(())
    }

    pub const fn subtle() -> Self {
        Self {
            strength: 0.3,
            edge_threshold: 0.15,
            adaptive: true,
        }
    }

    pub const fn moderate() -> Self {
        Self {
            strength: 0.5,
            edge_threshold: 0.12,
            adaptive: true,
        }
    }

    pub const fn strong() -> Self {
        Self {
            strength: 0.7,
            edge_threshold: 0.10,
            adaptive: true,
        }
    }
}

impl Default for SharpenConfig {
    fn default() -> Self {
        Self::moderate()
    }
}

/// VSR configuration
#[derive(Debug, Clone)]
pub struct VsrConfig {
    pub quality: VsrQuality,
    pub sharpening: f32,
    pub anti_ghosting: bool,
    pub clamping_gamma: Option<f32>,
    pub history_weight: Option<f32>,
    pub sharpen_config: Option<SharpenConfig>,
}

impl VsrConfig {
    pub fn clamping_gamma(&self) -> f32 {
        self.clamping_gamma.unwrap_or(self.quality.clamping_gamma())
    }

    pub fn history_weight(&self) -> f32 {
        self.history_weight.unwrap_or(self.quality.history_weight())
    }

    pub fn sharpen_config(&self) -> SharpenConfig {
        self.sharpen_config.unwrap_or_else(|| {
            let mut config = self.quality.default_sharpening();
            config.strength = self.sharpening;
            config
        })
    }
}

impl Default for VsrConfig {
    fn default() -> Self {
        Self {
            quality: VsrQuality::Medium,
            sharpening: 0.0,
            anti_ghosting: true,
            clamping_gamma: None,
            history_weight: None,
            sharpen_config: None,
        }
    }
}

/// Halton sequence for QMC jittering
pub struct HaltonSequence {
    indices: Vec<(f32, f32)>,
    curr_idx: usize,
}

impl HaltonSequence {
    pub fn new(samples: usize) -> Self {
        let mut indices = Vec::with_capacity(samples);
        for i in 1..=samples {
            indices.push((Self::phi(i as u32, 2) - 0.5, Self::phi(i as u32, 3) - 0.5));
        }
        Self {
            indices,
            curr_idx: 0,
        }
    }

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
    pub fn next_sample(&mut self) -> (f32, f32) {
        let res = self.indices[self.curr_idx];
        self.curr_idx = (self.curr_idx + 1) % self.indices.len();
        res
    }

    pub fn reset(&mut self) {
        self.curr_idx = 0;
    }

    pub fn peek(&self) -> (f32, f32) {
        self.indices[self.curr_idx]
    }

    pub fn capacity(&self) -> usize {
        self.indices.len()
    }

    pub fn progress(&self) -> f32 {
        self.curr_idx as f32 / self.indices.len() as f32
    }
}

/// Push constants for VSR upscale shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsrPushConstants {
    /// Current jitter offset
    pub jitter: [f32; 2],
    /// Screen dimensions (render resolution)
    pub render_size: [f32; 2],
    /// Screen dimensions (display resolution)
    pub display_size: [f32; 2],
    /// History blend factor
    pub history_weight: f32,
    /// Frame index for temporal variation
    pub frame_index: u32,
    /// Clamping gamma (1.0 Quality - 1.5 Responsive)
    pub clamping_gamma: f32,
    /// Velocity threshold for rejection
    pub velocity_threshold: f32,
    /// Anti-ghosting toggle (1.0 on, 0.0 off)
    pub anti_ghosting: f32,
}

/// Output selection for VSR
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsrOutput {
    /// Raw temporal output (no sharpening)
    Raw,
    /// Sharpened output (if sharpening is implemented/active)
    Sharpened,
}

impl Default for VsrOutput {
    fn default() -> Self {
        Self::Raw
    }
}

/// VSR input resources for upscaling
#[derive(Debug, Clone, Copy)]
pub struct VsrInputs {
    pub color: vk::ImageView,
    pub depth: vk::ImageView,
    pub motion: vk::ImageView,
    pub jitter: [f32; 2],
}

/// Upscale configuration for a single call
#[derive(Debug, Clone)]
pub struct VsrUpscaleConfig {
    pub velocity_threshold: f32,
    pub history_weight: f32,
    pub clamping_gamma: f32,
    pub anti_ghosting: bool,
}

impl Default for VsrUpscaleConfig {
    fn default() -> Self {
        Self {
            velocity_threshold: 0.05,
            history_weight: 0.95,
            clamping_gamma: 1.0,
            anti_ghosting: true,
        }
    }
}

impl VsrUpscaleConfig {
    pub fn validate(&self) -> Result<(), VsrError> {
        if self.velocity_threshold < 0.0 || self.velocity_threshold > 1.0 {
            return Err(VsrError::InvalidVelocityThreshold(self.velocity_threshold));
        }
        if self.history_weight < 0.0 || self.history_weight > 1.0 {
            return Err(VsrError::InvalidHistoryWeight(self.history_weight));
        }
        if self.clamping_gamma < 1.0 || self.clamping_gamma > 2.0 {
            return Err(VsrError::InvalidClampingGamma(self.clamping_gamma));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SharpenPushConstants {
    pub strength: f32,
    pub edge_threshold: f32,
    pub adaptive: f32,
    pub _padding: [f32; 5],
}

impl SharpenPushConstants {
    pub fn from_config(config: &SharpenConfig) -> Self {
        Self {
            strength: config.strength,
            edge_threshold: config.edge_threshold,
            adaptive: if config.adaptive { 1.0 } else { 0.0 },
            _padding: [0.0; 5],
        }
    }

    pub fn validate(&self) -> std::result::Result<(), VsrError> {
        if self.strength < 0.0 || self.strength > 1.0 {
            return Err(VsrError::InvalidSharpeningStrength(self.strength));
        }
        if self.edge_threshold < 0.01 || self.edge_threshold > 1.0 {
            return Err(VsrError::InvalidEdgeThreshold(self.edge_threshold));
        }
        if self.adaptive != 0.0 && self.adaptive != 1.0 {
            return Err(VsrError::InvalidAdaptiveFlag(self.adaptive));
        }
        Ok(())
    }
}

impl VsrPushConstants {
    pub fn validate(&self) -> Result<(), VsrError> {
        if self.clamping_gamma < 1.0 || self.clamping_gamma > 2.0 {
            return Err(VsrError::InvalidClampingGamma(self.clamping_gamma));
        }
        if self.history_weight < 0.0 || self.history_weight > 1.0 {
            return Err(VsrError::InvalidHistoryWeight(self.history_weight));
        }
        if self.velocity_threshold < 0.0 || self.velocity_threshold > 1.0 {
            return Err(VsrError::InvalidVelocityThreshold(self.velocity_threshold));
        }
        Ok(())
    }
}

/// Virtual Super-Resolution pass
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

    pub config: VsrConfig,
    halton: HaltonSequence,
    frame_idx: u32,

    // Quality metrics
    metrics: VsrMetrics,
    metrics_buffer: vk::Buffer,
    metrics_alloc: Option<vk_mem::Allocation>,
    metrics_readback_buffer: vk::Buffer,
    metrics_readback_alloc: Option<vk_mem::Allocation>,

    // Sharpening resources
    sharpened_img: vk::Image,
    sharpened_alloc: Option<vk_mem::Allocation>,
    sharpened_v: vk::ImageView,
    sharpen_pl: vk::Pipeline,
    sharpen_layout: vk::PipelineLayout,
    sharpen_desc_layout: vk::DescriptorSetLayout,
    sharpen_pool: vk::DescriptorPool,
    sharpen_sets: [vk::DescriptorSet; 2],

    last_output: VsrOutput,
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
            halton: HaltonSequence::new(16),
            frame_idx: 0,
            metrics: VsrMetrics::default(),
            metrics_buffer: vk::Buffer::null(),
            metrics_alloc: None,
            metrics_readback_buffer: vk::Buffer::null(),
            metrics_readback_alloc: None,
            config: VsrConfig::default(),

            // Sharpening
            sharpened_img: vk::Image::null(),
            sharpened_alloc: None,
            sharpened_v: vk::ImageView::null(),
            sharpen_pl: vk::Pipeline::null(),
            sharpen_layout: vk::PipelineLayout::null(),
            sharpen_desc_layout: vk::DescriptorSetLayout::null(),
            sharpen_pool: vk::DescriptorPool::null(),
            sharpen_sets: [vk::DescriptorSet::null(); 2],

            last_output: VsrOutput::Raw,
            initialized: false,
        }
    }

    /// Initialize VSR pass resources
    ///
    /// # Safety
    /// Device and allocator must stay valid for the lifetime of this pass.
    pub unsafe fn init(
        &mut self,
        alloc: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        display_width: u32,
        display_height: u32,
        quality: VsrQuality,
    ) -> Result<(), VsrError> {
        if self.initialized {
            return Ok(());
        }

        // Adversarial Defense: Zero-Sized Resource
        // Minimizing a window on Windows often causes width/height to become 0.
        // Creating Vulkan images with 0 dimensions is invalid and will crash.
        if display_width == 0 || display_height == 0 {
            log::warn!(
                "VsrPass: Skipping initialization with zero dimensions (window likely minimized)"
            );
            return Ok(()); // Return success, assume resize will reinitialize
        }

        let w = display_width;
        let h = display_height;

        self.display_w = w;
        self.display_h = h;
        self.config.quality = quality;

        let (rw, rh) = quality.render_size(w, h);
        self.render_w = rw;
        self.render_h = rh;

        self.create_motion_image(alloc).map_err(VsrError::Vulkan)?;
        self.create_history_images(alloc)
            .map_err(VsrError::Vulkan)?;
        self.create_metrics_buffers(alloc)
            .map_err(VsrError::Vulkan)?;
        self.create_sampler().map_err(VsrError::Vulkan)?;
        self.create_descriptors().map_err(VsrError::Vulkan)?;
        self.create_pipeline().map_err(VsrError::Vulkan)?;

        // Create sharpening resources
        self.create_sharpening_resources(alloc)
            .map_err(VsrError::Vulkan)?;
        self.create_sharpening_pipeline()
            .map_err(VsrError::Vulkan)?;

        self.initialized = true;
        Ok(())
    }

    unsafe fn create_metrics_buffers(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;
        let buffer_size = std::mem::size_of::<u32>() * 4;
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
            .map_err(|e| crate::AshError::VulkanError(format!("Sharpened image: {e:?}")))?;

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

        // Descriptor Pool
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(2)
            .pool_sizes(&pool_sizes);

        self.sharpen_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        // Descriptor Layout
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

        // Allocate Sets
        let layouts = [self.sharpen_desc_layout; 2];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.sharpen_pool)
            .set_layouts(&layouts);

        let sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        self.sharpen_sets = [sets[0], sets[1]];

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
            .size(std::mem::size_of::<VsrPushConstants>() as u32);

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

    pub fn jitter_projection(&mut self, projection: glam::Mat4) -> glam::Mat4 {
        let (jx, jy) = self.halton.next_sample();
        let dx = jx / self.render_w as f32;
        let dy = jy / self.render_h as f32;
        let mat = glam::Mat4::from_translation(glam::Vec3::new(dx * 2.0, dy * 2.0, 0.0));
        mat * projection
    }

    pub fn next_jitter(&mut self) -> (f32, f32) {
        self.halton.next_sample()
    }

    pub fn readback_metrics(
        &mut self,
        cmd: vk::CommandBuffer,
        alloc: &vk_mem::Allocator,
    ) -> Result<VsrMetricsReadback, VsrError> {
        if !self.initialized {
            return Err(VsrError::NotInitialized);
        }

        unsafe { self.readback_metrics_unsafe(cmd, alloc) }
    }

    unsafe fn readback_metrics_unsafe(
        &mut self,
        cmd: vk::CommandBuffer,
        alloc: &vk_mem::Allocator,
    ) -> Result<VsrMetricsReadback, VsrError> {
        // AAA Standard: GPU metrics readback
        // Copy metrics_buffer to metrics_readback_buffer
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(std::mem::size_of::<u32>() as u64 * 4);

        self.device.cmd_copy_buffer(
            cmd,
            self.metrics_buffer,
            self.metrics_readback_buffer,
            &[region],
        );

        // We assume the caller handles the fence/wait before actually reading the data
        // or we use host mapping if it's already finished.
        let alloc_info = alloc.get_allocation_info(self.metrics_readback_alloc.as_ref().unwrap());
        let ptr = alloc_info.mapped_data;
        if !ptr.is_null() {
            let data = std::slice::from_raw_parts(ptr as *const u32, 4);
            let rejection_rate = data[0] as f32 / 1000.0;
            let avg_blend_weight = data[1] as f32 / 1000.0;
            let ghosting_score = data[2] as f32 / 1000.0;
            let shimmering_score = data[3] as f32 / 1000.0;

            // Update internal metrics for quality_report()
            self.metrics.rejection_rate = rejection_rate;
            self.metrics.avg_blend_weight = avg_blend_weight;
            self.metrics.ghosting_score = ghosting_score;
            self.metrics.shimmering_score = shimmering_score;

            return Ok(VsrMetricsReadback {
                rejection_rate,
                avg_blend_weight,
                ghosting_score,
                shimmering_score,
                frame_time_ms: self.metrics.total_time_ms,
            });
        }

        Err(VsrError::MetricsReadbackFailed(
            "Failed to map host buffer".into(),
        ))
    }

    /// Upscale the input image using temporal upscaling
    ///
    /// # Safety
    /// Command buffer must be in recording state and images must be in valid layouts.
    pub unsafe fn upscale(
        &mut self,
        cmd: vk::CommandBuffer,
        input: vk::ImageView,
        depth: vk::ImageView,
        motion: vk::ImageView,
        jitter: [f32; 2],
        velocity_threshold: f32,
    ) -> Result<(), VsrError> {
        self.upscale_with_config(
            cmd,
            input,
            depth,
            motion,
            jitter,
            &VsrUpscaleConfig {
                velocity_threshold,
                history_weight: self.config.history_weight(),
                clamping_gamma: self.config.clamping_gamma(),
                anti_ghosting: self.config.anti_ghosting,
            },
        )
    }

    /// Upscale with custom configuration overrides
    ///
    /// # Safety
    /// Command buffer must be in recording state and images must be in valid layouts.
    pub unsafe fn upscale_with_config(
        &mut self,
        cmd: vk::CommandBuffer,
        input: vk::ImageView,
        depth: vk::ImageView,
        motion: vk::ImageView,
        jitter: [f32; 2],
        upscale_config: &VsrUpscaleConfig,
    ) -> Result<(), VsrError> {
        if !self.initialized {
            return Ok(());
        }

        upscale_config.validate()?;

        let prev = (self.frame_idx % 2) as usize;
        let curr = ((self.frame_idx + 1) % 2) as usize;

        self.update_descriptor_sets(curr, input, depth, motion, prev)
            .map_err(VsrError::Vulkan)?;

        // Transition layouts
        let barriers = [
            vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[prev])
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[curr])
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
        ];

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.upscale_pl);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.upscale_layout,
                0,
                &[self.desc_sets[curr]],
                &[],
            );

            let pc = VsrPushConstants {
                jitter,
                render_size: [self.render_w as f32, self.render_h as f32],
                display_size: [self.display_w as f32, self.display_h as f32],
                history_weight: upscale_config.history_weight,
                frame_index: self.frame_idx,
                clamping_gamma: upscale_config.clamping_gamma,
                velocity_threshold: upscale_config.velocity_threshold,
                anti_ghosting: if upscale_config.anti_ghosting {
                    1.0
                } else {
                    0.0
                },
            };

            pc.validate()?;

            self.device.cmd_push_constants(
                cmd,
                self.upscale_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&pc),
            );
            self.device.cmd_dispatch(
                cmd,
                self.display_w.div_ceil(8),
                self.display_h.div_ceil(8),
                1,
            );
        }

        Ok(())
    }

    unsafe fn update_descriptor_sets(
        &self,
        set_idx: usize,
        input: vk::ImageView,
        depth: vk::ImageView,
        motion: vk::ImageView,
        prev: usize,
    ) -> Result<()> {
        let sampler_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let input_info = [sampler_info.image_view(input)];
        let motion_info = [sampler_info.image_view(motion)];
        let depth_info = [sampler_info.image_view(depth)];
        let history_info = [sampler_info.image_view(self.history_vs[prev])];
        let output_info = [vk::DescriptorImageInfo::default()
            .image_view(self.history_vs[(prev + 1) % 2])
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

    /// Combined upscale and sharpening pass
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn upscale_with_sharpening(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: VsrInputs,
        upscale_config: &VsrUpscaleConfig,
        sharpen_config: Option<&SharpenConfig>,
    ) -> Result<VsrOutput, VsrError> {
        // 1. Perform upscale
        self.upscale_with_config(
            cmd,
            inputs.color,
            inputs.depth,
            inputs.motion,
            inputs.jitter,
            upscale_config,
        )?;

        // 2. Perform sharpening if requested
        let mut output = VsrOutput::Raw;
        if let Some(config) = sharpen_config {
            if config.strength >= 0.01 {
                self.apply_sharpening(cmd, config)?;
                output = VsrOutput::Sharpened;
            }
        }

        self.last_output = output;
        Ok(output)
    }

    /// Apply sharpening to the VSR output
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn apply_sharpening(
        &mut self,
        cmd: vk::CommandBuffer,
        config: &SharpenConfig,
    ) -> Result<(), VsrError> {
        config.validate()?;

        if config.strength < 0.01 {
            return Ok(());
        }

        if !self.initialized {
            return Err(VsrError::NotInitialized);
        }

        if self.sharpen_pl == vk::Pipeline::null() {
            return Err(VsrError::SharpeningNotInitialized);
        }

        self.apply_sharpening_unsafe(cmd, config)
    }

    unsafe fn apply_sharpening_unsafe(
        &mut self,
        cmd: vk::CommandBuffer,
        config: &SharpenConfig,
    ) -> Result<(), VsrError> {
        let curr = (self.frame_idx % 2) as usize;

        // Transition layouts
        let barriers = [
            vk::ImageMemoryBarrier::default()
                .image(self.history_imgs[curr])
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            vk::ImageMemoryBarrier::default()
                .image(self.sharpened_img)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
        ];

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );

        // Update descriptors
        self.update_sharpen_descriptors(curr)
            .map_err(VsrError::Vulkan)?;

        // Bind pipeline and dispatch
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.sharpen_pl);

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.sharpen_layout,
            0,
            &[self.sharpen_sets[curr]],
            &[],
        );

        let pc = SharpenPushConstants::from_config(config);
        pc.validate()?;

        self.device.cmd_push_constants(
            cmd,
            self.sharpen_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
        );

        let gx = self.display_w.div_ceil(8);
        let gy = self.display_h.div_ceil(8);
        self.device.cmd_dispatch(cmd, gx, gy, 1);

        Ok(())
    }

    unsafe fn update_sharpen_descriptors(&self, set_idx: usize) -> Result<()> {
        let sampler_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let input_info = [sampler_info.image_view(self.history_vs[set_idx])];
        let output_info = [vk::DescriptorImageInfo::default()
            .image_view(self.sharpened_v)
            .image_layout(vk::ImageLayout::GENERAL)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.sharpen_sets[set_idx])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&input_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.sharpen_sets[set_idx])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&output_info),
        ];

        self.device.update_descriptor_sets(&writes, &[]);
        Ok(())
    }

    pub fn output_view(&self, output: VsrOutput) -> vk::ImageView {
        match output {
            VsrOutput::Raw => self.history_vs[(self.frame_idx % 2) as usize],
            VsrOutput::Sharpened => {
                if self.sharpened_v == vk::ImageView::null() {
                    self.history_vs[(self.frame_idx % 2) as usize]
                } else {
                    self.sharpened_v
                }
            }
        }
    }

    /// Get the view for the most recent upscale result
    pub fn active_view(&self) -> vk::ImageView {
        self.output_view(self.last_output)
    }

    pub fn raw_output_view(&self) -> vk::ImageView {
        self.output_view(VsrOutput::Raw)
    }

    pub fn sharpened_output_view(&self) -> vk::ImageView {
        self.output_view(VsrOutput::Sharpened)
    }

    pub fn motion_view(&self) -> vk::ImageView {
        self.motion_v
    }
    pub fn render_size(&self) -> (u32, u32) {
        (self.render_w, self.render_h)
    }
    pub fn display_size(&self) -> (u32, u32) {
        (self.display_w, self.display_h)
    }
    pub fn next_frame(&mut self) {
        self.frame_idx = self.frame_idx.wrapping_add(1);
    }

    pub fn quality_report(&self) -> VsrQualityReport {
        VsrQualityReport {
            resolution: ResolutionInfo {
                render: (self.render_w, self.render_h),
                display: (self.display_w, self.display_h),
                factor: self.config.quality.factor(),
            },
            quality: self.config.quality,
            metrics: self.metrics,
            assessment: self.metrics.quality_assessment(),
        }
    }

    /// Destroy VSR pass resources
    ///
    /// # Safety
    /// No GPU commands using these resources must be in flight.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        self.device.destroy_image_view(self.motion_v, None);
        if let Some(mut a) = self.motion_alloc.take() {
            allocator.destroy_image(self.motion_img, &mut a);
        }

        for i in 0..2 {
            self.device.destroy_image_view(self.history_vs[i], None);
            if let Some(mut a) = self.history_allocs[i].take() {
                allocator.destroy_image(self.history_imgs[i], &mut a);
            }
        }

        self.device.destroy_sampler(self.sampler, None);
        self.device.destroy_pipeline(self.upscale_pl, None);
        self.device
            .destroy_pipeline_layout(self.upscale_layout, None);
        self.device
            .destroy_descriptor_pool(self.descriptor_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.desc_layout, None);

        if let Some(mut a) = self.metrics_alloc.take() {
            allocator.destroy_buffer(self.metrics_buffer, &mut a);
        }
        if let Some(mut a) = self.metrics_readback_alloc.take() {
            allocator.destroy_buffer(self.metrics_readback_buffer, &mut a);
        }

        // Sharpening cleanup
        self.device.destroy_image_view(self.sharpened_v, None);
        if let Some(mut a) = self.sharpened_alloc.take() {
            allocator.destroy_image(self.sharpened_img, &mut a);
        }
        self.device.destroy_pipeline(self.sharpen_pl, None);
        self.device
            .destroy_pipeline_layout(self.sharpen_layout, None);
        self.device.destroy_descriptor_pool(self.sharpen_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.sharpen_desc_layout, None);

        self.initialized = false;
    }
}
