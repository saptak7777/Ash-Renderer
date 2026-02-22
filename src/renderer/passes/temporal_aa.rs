//! Temporal Anti-Aliasing (TAA) System
//!
//! Provides high-quality anti-aliasing by blending the current frame with
//! previous frames using motion vectors and color clamping.
//!
//! # Features
//! - Halton jitter sequence
//! - Velocity buffer support
//! - Neighborhood color clamping
//! - Configurable blend factor

use ash::vk;
use glam::{Mat4, Vec2};
use std::fmt;
use std::sync::Arc;

/// Configuration validation error
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValidationError {
    InvalidVelocityThreshold { value: f32 },
    InvalidDepthThreshold { value: f32 },
    InvalidJitterScale { value: f32 },
}

impl fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVelocityThreshold { value } => {
                write!(f, "Invalid velocity threshold: {value} (must be 0.0-1.0)")
            }
            Self::InvalidDepthThreshold { value } => {
                write!(f, "Invalid depth threshold: {value} (must be >= 0.0)")
            }
            Self::InvalidJitterScale { value } => {
                write!(f, "Invalid jitter scale: {value} (must be > 0.0)")
            }
        }
    }
}

/// Config change type (for determining if recreation is needed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigChangeType {
    /// No change
    None,
    /// Minor change (no recreation needed)
    Minor,
    /// Major change (recreation needed)
    Major,
}

impl ConfigChangeType {
    /// Check if recreation is needed
    pub fn needs_recreation(self) -> bool {
        matches!(self, ConfigChangeType::Major)
    }
}

/// Detect config change type (AAA pattern)
pub fn detect_config_change(old: &TaaConfig, new: &TaaConfig) -> ConfigChangeType {
    // Quality change requires recreation (Unreal pattern)
    if old.quality != new.quality {
        return ConfigChangeType::Major;
    }

    // Sharpening mode change requires recreation
    if old.sharpening != new.sharpening {
        return ConfigChangeType::Major;
    }

    // Enable/disable requires recreation
    if old.enabled != new.enabled {
        return ConfigChangeType::Major;
    }

    // Threshold changes are minor (no recreation)
    if old.velocity_threshold != new.velocity_threshold
        || old.depth_threshold != new.depth_threshold
    {
        return ConfigChangeType::Minor;
    }

    // No change
    ConfigChangeType::None
}

impl std::error::Error for ConfigValidationError {}

/// Validation trait (AAA pattern + Rust composable validation)
pub trait Validate {
    /// Validate configuration
    fn validate(&self) -> Result<(), ConfigValidationError>;

    /// Validate and clamp invalid values (Unity pattern)
    fn validate_clamped(&self) -> Self
    where
        Self: Clone;
}

/// Configuration change metrics (AAA-grade profiling)
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ConfigMetrics {
    /// Total number of config changes
    pub change_count: u64,
    /// Number of validation failures
    pub validation_failures: u64,
    /// Number of auto-clamped values
    pub clamped_values: u64,
    /// Last change timestamp (frame number)
    pub last_change_frame: u64,
}

impl ConfigMetrics {
    /// Record a config change
    pub fn record_change(&mut self, frame: u64) {
        self.change_count += 1;
        self.last_change_frame = frame;
    }

    /// Record a validation failure
    pub fn record_validation_failure(&mut self) {
        self.validation_failures += 1;
    }

    /// Record an auto-clamped value
    pub fn record_clamped(&mut self) {
        self.clamped_values += 1;
    }

    /// Get metrics report (for debugging)
    pub fn report(&self) -> ConfigMetricsReport {
        ConfigMetricsReport {
            change_count: self.change_count,
            validation_failures: self.validation_failures,
            clamped_values: self.clamped_values,
            validation_failure_rate: if self.change_count > 0 {
                self.validation_failures as f32 / self.change_count as f32
            } else {
                0.0
            },
        }
    }
}

/// Metrics report (for display)
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigMetricsReport {
    pub change_count: u64,
    pub validation_failures: u64,
    pub clamped_values: u64,
    pub validation_failure_rate: f32,
}

impl fmt::Display for ConfigMetricsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config Metrics:\n\
             - Changes: {}\n\
             - Validation Failures: {} ({:.1}%)\n\
             - Clamped Values: {}",
            self.change_count,
            self.validation_failures,
            self.validation_failure_rate * 100.0,
            self.clamped_values
        )
    }
}

/// TAA quality preset (AAA-grade, Unreal VSR style)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaaQuality {
    /// Responsive (low latency, less stable)
    Responsive,
    /// Balanced (default)
    Balanced,
    /// Quality (most stable)
    Quality,
}

impl TaaQuality {
    /// Get history weight
    pub const fn history_weight(self) -> f32 {
        match self {
            Self::Responsive => 0.7,
            Self::Balanced => 0.85,
            Self::Quality => 0.95,
        }
    }

    /// Get clamping gamma (for AABB clipping)
    pub const fn clamping_gamma(self) -> f32 {
        match self {
            Self::Responsive => 1.5,
            Self::Balanced => 1.2,
            Self::Quality => 1.0,
        }
    }
}

impl Default for TaaQuality {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Sharpening mode (Unreal pattern)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SharpeningMode {
    /// No sharpening
    None,
    /// Subtle (0.2 strength)
    Subtle,
    /// Moderate (0.4 strength) - Unreal default
    Moderate,
    /// Strong (0.6 strength)
    Strong,
}

impl SharpeningMode {
    pub fn strength(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Subtle => 0.2,
            Self::Moderate => 0.4,
            Self::Strong => 0.6,
        }
    }
}

impl Default for SharpeningMode {
    fn default() -> Self {
        Self::Moderate
    }
}

/// TAA configuration
#[derive(Debug, Clone)]
pub struct TaaConfig {
    /// Enable/disable TAA
    pub enabled: bool,
    /// Quality preset (AAA-grade)
    pub quality: TaaQuality,
    /// Sharpening mode
    pub sharpening: SharpeningMode,
    /// Blend factor (0.0 = current only, 1.0 = history only)
    /// Note: Overridden by quality preset if not manually set
    pub blend_factor: f32,
    /// Enable color clamping to reduce ghosting
    pub color_clamp: bool,
    /// Enable velocity rejection
    pub velocity_rejection: bool,
    /// Velocity rejection threshold (higher = more responsive)
    pub velocity_threshold: f32,
    /// Depth rejection threshold
    pub depth_threshold: f32,
    /// Anti-flicker (Unreal VSR feature)
    pub anti_flicker: bool,
    /// Jitter scale (typically 1.0)
    pub jitter_scale: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        let quality = TaaQuality::default();
        Self {
            enabled: true,
            quality,
            sharpening: SharpeningMode::default(),
            blend_factor: quality.history_weight(),
            color_clamp: true,
            velocity_rejection: true,
            velocity_threshold: 0.02, // Unreal default
            depth_threshold: 0.1,     // Unreal default
            anti_flicker: true,
            jitter_scale: 1.0,
        }
    }
}

impl Validate for TaaConfig {
    fn validate(&self) -> Result<(), ConfigValidationError> {
        if self.velocity_threshold < 0.0 || self.velocity_threshold > 1.0 {
            return Err(ConfigValidationError::InvalidVelocityThreshold {
                value: self.velocity_threshold,
            });
        }

        if self.depth_threshold < 0.0 {
            return Err(ConfigValidationError::InvalidDepthThreshold {
                value: self.depth_threshold,
            });
        }

        if self.jitter_scale <= 0.0 {
            return Err(ConfigValidationError::InvalidJitterScale {
                value: self.jitter_scale,
            });
        }

        if !self.enabled {
            log::warn!(
                "TAA is disabled but quality is set to {:?}. Quality will be ignored.",
                self.quality
            );
        }

        Ok(())
    }

    fn validate_clamped(&self) -> Self
    where
        Self: Clone,
    {
        let mut clamped = self.clone();

        clamped.velocity_threshold = clamped.velocity_threshold.clamp(0.0, 1.0);

        if clamped.depth_threshold < 0.0 {
            clamped.depth_threshold = 0.0;
            log::warn!("Depth threshold clamped to 0.0");
        }

        if clamped.jitter_scale <= 0.0 {
            clamped.jitter_scale = 1.0;
            log::warn!("Jitter scale clamped to 1.0");
        }

        clamped
    }
}

use crate::renderer::util::halton::HaltonSequence;

/// TAA push constants for shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TaaPushConstants {
    pub width: f32,
    pub height: f32,
    pub jitter_x: f32, // Current Frame Jitter (NDC)
    pub jitter_y: f32,
    pub prev_jitter_x: f32, // Previous Frame Jitter (NDC)
    pub prev_jitter_y: f32,
    pub blend_factor: f32,
    pub clamping_gamma: f32,
    pub anti_flicker: u32,
}

impl Default for TaaPushConstants {
    fn default() -> Self {
        Self {
            width: 1920.0,
            height: 1080.0,
            jitter_x: 0.0,
            jitter_y: 0.0,
            prev_jitter_x: 0.0,
            prev_jitter_y: 0.0,
            blend_factor: 0.9,
            clamping_gamma: 1.0,
            anti_flicker: 1,
        }
    }
}

/// Temporal Anti-Aliasing manager
pub struct TemporalAA {
    config: TaaConfig,
    halton: HaltonSequence,
    current_jitter: Vec2,
    previous_jitter: Vec2,
    frame_index: u64,
}

impl TemporalAA {
    /// Create a new TAA manager
    pub fn new() -> Self {
        Self::with_config(TaaConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: TaaConfig) -> Self {
        Self {
            config,
            halton: HaltonSequence::new(2, 3),
            current_jitter: Vec2::ZERO,
            previous_jitter: Vec2::ZERO,
            frame_index: 0,
        }
    }

    /// Begin new frame - update jitter
    pub fn begin_frame(&mut self) {
        self.previous_jitter = self.current_jitter;
        self.current_jitter = self.halton.next_sample() * self.config.jitter_scale;
        self.frame_index += 1;
    }

    /// Get jittered projection matrix
    pub fn jitter_projection(&self, projection: Mat4, width: u32, height: u32) -> Mat4 {
        if !self.config.enabled {
            return projection;
        }

        let jitter_x = self.current_jitter.x * 2.0 / width as f32;
        let jitter_y = self.current_jitter.y * 2.0 / height as f32;

        let mut jittered = projection;
        jittered.w_axis.x += jitter_x;
        jittered.w_axis.y += jitter_y;
        jittered
    }

    /// Get push constants for TAA resolve shader
    pub fn push_constants(&self, width: u32, height: u32) -> TaaPushConstants {
        TaaPushConstants {
            width: width as f32,
            height: height as f32,
            jitter_x: self.current_jitter.x,
            jitter_y: self.current_jitter.y,
            prev_jitter_x: self.previous_jitter.x,
            prev_jitter_y: self.previous_jitter.y,
            blend_factor: self.config.blend_factor,
            clamping_gamma: self.config.quality.clamping_gamma(),
            anti_flicker: if self.config.anti_flicker { 1 } else { 0 },
        }
    }

    /// Get current jitter
    pub fn current_jitter(&self) -> Vec2 {
        self.current_jitter
    }

    /// Is TAA enabled?
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get mutable config reference
    pub fn config_mut(&mut self) -> &mut TaaConfig {
        &mut self.config
    }

    /// Get config reference
    pub fn config(&self) -> &TaaConfig {
        &self.config
    }

    /// Reset history (call on camera cut or teleport)
    pub fn reset_history(&mut self) {
        self.halton.reset();
        self.current_jitter = Vec2::ZERO;
        self.previous_jitter = Vec2::ZERO;
    }
}

impl Default for TemporalAA {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TaaPass — GPU-resource-owning TAA pass for 1:1 native resolution.
//
// Distinct from `TemporalAA` (which is a pure-logic jitter/config struct),
// `TaaPass` owns Vulkan resources: two ping-pong history images, a descriptor
// set layout/pool, and the compute pipeline for `taa_resolve.comp`.
//
// Usage:
//   let mut taa = TaaPass::new(device.clone())?;
//   taa.init(&allocator, width, height)?;
//   // each frame:
//   let output_view = taa.resolve(cmd, color_view, depth_view, motion_view, &push_constants)?;
// ─────────────────────────────────────────────────────────────────────────────

use crate::{AshError, Result};
// TaaPushConstants is defined above (line ~380) and reused here for TaaPass.

/// Standalone GPU TAA pass.
///
/// Owns two ping-pong `R16G16B16A16_SFLOAT` history images, a dedicated
/// descriptor set layout/pool, and the `taa_resolve.comp` compute pipeline.
pub struct TaaPass {
    device: Arc<ash::Device>,

    // Ping-pong history images (display resolution, R16G16B16A16_SFLOAT)
    history_images: [vk::Image; 2],
    history_allocs: [Option<vk_mem::Allocation>; 2],
    history_views: [vk::ImageView; 2],

    // Sampler for reading history and color inputs
    sampler: vk::Sampler,

    // Descriptor infrastructure (one set per ping-pong slot)
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; 2],

    // Compute pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    // Dimensions
    width: u32,
    height: u32,

    // Ping-pong frame counter
    frame_index: u64,

    initialized: bool,
}

impl TaaPass {
    /// Create a new (uninitialized) `TaaPass`.
    ///
    /// Call [`init`] before the first frame to allocate GPU resources.
    pub fn new(device: Arc<ash::Device>) -> Result<Self> {
        Ok(Self {
            device,
            history_images: [vk::Image::null(); 2],
            history_allocs: [None, None],
            history_views: [vk::ImageView::null(); 2],
            sampler: vk::Sampler::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: [vk::DescriptorSet::null(); 2],
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            width: 0,
            height: 0,
            frame_index: 0,
            initialized: false,
        })
    }

    /// Allocate GPU resources for the given display resolution.
    ///
    /// Safe to call again after a swapchain resize — old resources are freed first.
    pub fn init(&mut self, allocator: &vk_mem::Allocator, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            log::warn!("TaaPass::init called with zero dimensions; skipping.");
            return Ok(());
        }

        // Free any previously allocated resources before re-initializing.
        if self.initialized {
            unsafe {
                self.destroy_resources(allocator);
            }
        }

        self.width = width;
        self.height = height;

        unsafe {
            self.create_sampler()?;
            self.create_history_images(allocator)?;
            self.create_descriptor_layout()?;
            self.create_descriptor_pool()?;
            self.create_pipeline()?;
            // Write initial descriptor sets (both slots point to the same images
            // initially; they will be swapped each frame via ping-pong).
            self.update_descriptor_sets(
                0,
                vk::ImageView::null(),
                vk::ImageView::null(),
                vk::ImageView::null(),
            );
            self.update_descriptor_sets(
                1,
                vk::ImageView::null(),
                vk::ImageView::null(),
                vk::ImageView::null(),
            );
        }

        self.initialized = true;
        log::info!("TaaPass initialized ({width}x{height})");
        Ok(())
    }

    /// Record all TAA frame commands into `cmd`.
    ///
    /// Synthesizes the full TAA frame lifecycle:
    /// 1. Update descriptor sets with current frame views.
    /// 2. Transition history images for reading.
    /// 3. Dispatch the `taa_resolve.comp` compute shader.
    /// 4. Update the internal ping-pong index for the next frame.
    ///
    /// # Safety
    /// Command buffer must be in recording state and views must be valid.
    pub unsafe fn record_commands(
        &mut self,
        cmd: vk::CommandBuffer,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        motion_view: vk::ImageView,
        push: &TaaPushConstants,
    ) -> Result<()> {
        unsafe { self.resolve(cmd, color_view, depth_view, motion_view, push)? };
        Ok(())
    }

    /// Dispatch the TAA resolve compute shader.
    ///
    /// Bindings expected:
    ///   - `color_view`  : current HDR color image (SHADER_READ_ONLY_OPTIMAL)
    ///   - `depth_view`  : depth buffer (SHADER_READ_ONLY_OPTIMAL)
    ///   - `motion_view` : motion vector buffer (SHADER_READ_ONLY_OPTIMAL)
    ///
    /// Returns the image view of the resolved output (the write history slot),
    /// which can be passed directly to the tonemapper.
    ///
    /// # Safety
    /// The caller must ensure all input images are in the correct layout and
    /// that the command buffer is recording.
    pub unsafe fn resolve(
        &mut self,
        cmd: vk::CommandBuffer,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        motion_view: vk::ImageView,
        push: &TaaPushConstants,
    ) -> Result<vk::ImageView> {
        if !self.initialized {
            return Err(AshError::VulkanError("TaaPass not initialized".into()));
        }

        // Ping-pong: read from slot `read_idx`, write to slot `write_idx`.
        let read_idx = (self.frame_index % 2) as usize;
        let write_idx = ((self.frame_index + 1) % 2) as usize;

        // ── Transition write history image to GENERAL (storage write) ─────────
        let write_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .image(self.history_images[write_idx])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[write_barrier],
            );
        }

        // ── Transition read history image to SHADER_READ_ONLY_OPTIMAL ─────────
        let read_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .image(self.history_images[read_idx])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // Only apply the read barrier after the first frame (frame 0 has no prior write).
        if self.frame_index > 0 {
            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[read_barrier],
                );
            }
        }

        // ── Update descriptor set for this frame's slot ───────────────────────
        // Slot `write_idx` is the set we dispatch with:
        //   binding 0 = output (write_idx history, GENERAL)
        //   binding 1 = color (current frame)
        //   binding 2 = depth
        //   binding 3 = motion
        //   binding 4 = history (read_idx, SHADER_READ_ONLY_OPTIMAL)
        unsafe {
            self.update_descriptor_sets(write_idx, color_view, depth_view, motion_view);
            // Patch binding 4 (history read) to point to the read slot.
            self.update_history_read_binding(write_idx, self.history_views[read_idx]);
        }

        // ── Bind and dispatch ─────────────────────────────────────────────────
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[write_idx]],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(push),
            );
        }

        let groups_x = self.width.div_ceil(8);
        let groups_y = self.height.div_ceil(8);
        unsafe {
            self.device.cmd_dispatch(cmd, groups_x, groups_y, 1);
        }

        // ── Transition write image to SHADER_READ_ONLY for the tonemapper ─────
        let post_barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .image(self.history_images[write_idx])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[post_barrier],
            );
        }

        self.frame_index += 1;
        Ok(self.history_views[write_idx])
    }

    /// Returns the output image view from the most recent `resolve()` call.
    pub fn output_view(&self) -> vk::ImageView {
        let write_idx = ((self.frame_index.saturating_sub(1) + 1) % 2) as usize;
        self.history_views[write_idx]
    }

    /// Whether the pass has been initialized with GPU resources.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    unsafe fn create_sampler(&mut self) -> Result<()> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        self.sampler = unsafe { self.device.create_sampler(&info, None)? };
        Ok(())
    }

    unsafe fn create_history_images(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;
        for i in 0..2 {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
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

            let (img, allocation) = unsafe { allocator.create_image(&image_info, &alloc_info) }
                .map_err(|e| AshError::VulkanError(format!("TAA history image {i}: {e:?}")))?;

            self.history_images[i] = img;
            self.history_allocs[i] = Some(allocation);

            let view_info = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            self.history_views[i] = unsafe { self.device.create_image_view(&view_info, None)? };
        }
        Ok(())
    }

    unsafe fn create_descriptor_layout(&mut self) -> Result<()> {
        // 5 bindings matching taa_resolve.comp:
        //   0: storage image (output)
        //   1: combined image sampler (color)
        //   2: combined image sampler (depth)
        //   3: combined image sampler (motion)
        //   4: combined image sampler (history)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
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
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)?
        };
        Ok(())
    }

    unsafe fn create_descriptor_pool(&mut self) -> Result<()> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2, // one per ping-pong slot
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 8, // 4 samplers × 2 slots
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(2);

        self.descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

        let layouts = [self.descriptor_set_layout; 2];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };
        self.descriptor_sets[0] = sets[0];
        self.descriptor_sets[1] = sets[1];
        Ok(())
    }

    unsafe fn create_pipeline(&mut self) -> Result<()> {
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/taa_resolve.comp.spv"));
        let spv = ash::util::read_spv(&mut std::io::Cursor::new(shader_code))
            .map_err(|e| AshError::VulkanError(format!("TAA SPV parse: {e}")))?;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spv);
        let shader_module = unsafe { self.device.create_shader_module(&module_info, None)? };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<TaaPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));

        self.pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| e)?;
        self.pipeline = pipelines[0];

        unsafe { self.device.destroy_shader_module(shader_module, None) };
        Ok(())
    }

    /// Write all bindings for descriptor set `slot` except binding 4 (history read).
    unsafe fn update_descriptor_sets(
        &self,
        slot: usize,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        motion_view: vk::ImageView,
    ) {
        let set = self.descriptor_sets[slot];

        // Binding 0: output storage image (this slot's history image)
        let output_info = vk::DescriptorImageInfo::default()
            .image_view(self.history_views[slot])
            .image_layout(vk::ImageLayout::GENERAL);

        // Binding 1: current color
        let color_info = vk::DescriptorImageInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler);

        // Binding 2: depth
        let depth_info = vk::DescriptorImageInfo::default()
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler);

        // Binding 3: motion
        let motion_info = vk::DescriptorImageInfo::default()
            .image_view(motion_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&color_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&motion_info)),
        ];

        // Skip writes with null views (happens during init before first frame).
        let valid_writes: Vec<_> = writes
            .iter()
            .filter(|w| {
                let info = unsafe { *w.p_image_info };
                info.image_view != vk::ImageView::null()
            })
            .cloned()
            .collect();

        if !valid_writes.is_empty() {
            unsafe { self.device.update_descriptor_sets(&valid_writes, &[]) };
        }
    }

    /// Update only binding 4 (history read) for the given descriptor set slot.
    unsafe fn update_history_read_binding(&self, slot: usize, history_view: vk::ImageView) {
        if history_view == vk::ImageView::null() {
            return;
        }
        let history_info = vk::DescriptorImageInfo::default()
            .image_view(history_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(self.sampler);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_sets[slot])
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&history_info));

        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    /// Free all GPU resources. Called by `destroy_resources` (pub) and by `init` on resize.
    ///
    /// # Safety
    /// The caller must ensure that the GPU is idle and no resources are currently in use.
    pub unsafe fn destroy_resources(&mut self, allocator: &vk_mem::Allocator) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.descriptor_pool = vk::DescriptorPool::null();
            }
            if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
                self.descriptor_set_layout = vk::DescriptorSetLayout::null();
            }
            for i in 0..2 {
                if self.history_views[i] != vk::ImageView::null() {
                    self.device.destroy_image_view(self.history_views[i], None);
                    self.history_views[i] = vk::ImageView::null();
                }
                if self.history_images[i] != vk::Image::null() {
                    if let Some(mut alloc) = self.history_allocs[i].take() {
                        allocator.destroy_image(self.history_images[i], &mut alloc);
                        self.history_images[i] = vk::Image::null();
                    }
                }
            }
            if self.sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.sampler, None);
                self.sampler = vk::Sampler::null();
            }
        }

        self.initialized = false;
    }
}

// NOTE: TaaPass does not implement Drop automatically because `destroy_resources`
// requires the VMA allocator, which is not stored in the struct.
// The caller (Renderer) must call `destroy_resources` explicitly before dropping.
// This mirrors the pattern used by VsrPass.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_sequence() {
        let mut halton = HaltonSequence::new(2, 3);
        let j1 = halton.next_sample();
        let j2 = halton.next_sample();
        // Should be different
        assert_ne!(j1, j2);
        // Should be in range
        assert!(j1.x >= -0.5 && j1.x <= 0.5);
    }

    #[test]
    fn test_jittered_projection() {
        let taa = TemporalAA::new();
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let jittered = taa.jitter_projection(proj, 1920, 1080);
        // Initial jitter is zero, so should be same
        assert_eq!(proj, jittered);
    }

    #[test]
    fn test_taa_quality_presets() {
        // Responsive: low history weight, high clamping gamma
        assert_eq!(TaaQuality::Responsive.history_weight(), 0.7);
        assert_eq!(TaaQuality::Responsive.clamping_gamma(), 1.5);

        // Balanced: middle ground
        assert_eq!(TaaQuality::Balanced.history_weight(), 0.85);
        assert_eq!(TaaQuality::Balanced.clamping_gamma(), 1.2);

        // Quality: high history weight, low clamping gamma
        assert_eq!(TaaQuality::Quality.history_weight(), 0.95);
        assert_eq!(TaaQuality::Quality.clamping_gamma(), 1.0);
    }

    #[test]
    fn test_sharpening_modes() {
        assert_eq!(SharpeningMode::None.strength(), 0.0);
        assert_eq!(SharpeningMode::Subtle.strength(), 0.2);
        assert_eq!(SharpeningMode::Moderate.strength(), 0.4);
        assert_eq!(SharpeningMode::Strong.strength(), 0.6);
    }

    #[test]
    fn test_taa_config_defaults() {
        let config = TaaConfig::default();
        assert_eq!(config.quality, TaaQuality::Balanced);
        assert_eq!(config.sharpening, SharpeningMode::Moderate);
        assert_eq!(config.blend_factor, 0.85); // Balanced preset
        assert!(config.anti_flicker);
        assert_eq!(config.velocity_threshold, 0.02);
        assert_eq!(config.depth_threshold, 0.1);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = TaaConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_velocity() {
        let config = TaaConfig {
            velocity_threshold: 2.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_depth() {
        let config = TaaConfig {
            depth_threshold: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_jitter() {
        let config = TaaConfig {
            jitter_scale: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_clamped() {
        let config = TaaConfig {
            velocity_threshold: 2.0,
            depth_threshold: -1.0,
            jitter_scale: -0.5,
            ..Default::default()
        };

        let clamped = config.validate_clamped();
        assert_eq!(clamped.velocity_threshold, 1.0);
        assert_eq!(clamped.depth_threshold, 0.0);
        assert_eq!(clamped.jitter_scale, 1.0);
    }

    #[test]
    fn test_detect_config_change_none() {
        let old = TaaConfig::default();
        let new = old.clone();
        assert_eq!(detect_config_change(&old, &new), ConfigChangeType::None);
    }

    #[test]
    fn test_detect_config_change_minor() {
        let old = TaaConfig::default();
        let new = TaaConfig {
            velocity_threshold: 0.5,
            ..old.clone()
        };
        assert_eq!(detect_config_change(&old, &new), ConfigChangeType::Minor);
    }

    #[test]
    fn test_detect_config_change_major() {
        let old = TaaConfig::default();
        let new = TaaConfig {
            quality: TaaQuality::Quality,
            ..old.clone()
        };
        assert_eq!(detect_config_change(&old, &new), ConfigChangeType::Major);
    }

    #[test]
    fn test_config_metrics() {
        let mut metrics = ConfigMetrics::default();
        metrics.record_change(100);
        metrics.record_validation_failure();
        metrics.record_clamped();

        assert_eq!(metrics.change_count, 1);
        assert_eq!(metrics.validation_failures, 1);
        assert_eq!(metrics.clamped_values, 1);
        assert_eq!(metrics.last_change_frame, 100);

        let report = metrics.report();
        assert_eq!(report.change_count, 1);
        assert_eq!(report.validation_failure_rate, 1.0);
    }

    #[test]
    fn test_config_change_type_needs_recreation() {
        assert!(!ConfigChangeType::None.needs_recreation());
        assert!(!ConfigChangeType::Minor.needs_recreation());
        assert!(ConfigChangeType::Major.needs_recreation());
    }
}
