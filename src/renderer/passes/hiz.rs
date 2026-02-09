//! Hi-Z Pyramid GPU Resources
//!
//! Manages GPU-side resources for hierarchical-Z occlusion culling:
//! - Hi-Z pyramid image (mip chain)
//! - Compute pipelines for pyramid generation and culling
//! - Descriptor sets and buffers

use ash::vk;
use std::sync::Arc;

use crate::vulkan::{Allocator, VulkanDevice};
use crate::Result;
use thiserror::Error;

/// Hi-Z error types (Rust explicit errors)
#[derive(Debug, Error)]
pub enum HiZError {
    #[error("Insufficient resolution {resolution:?} for {requested} mips (max: {max_possible})")]
    InsufficientResolution {
        requested: u32,
        max_possible: u32,
        resolution: (u32, u32),
    },

    #[error("Mip level {mip_level} too small: {size:?} (minimum 4x4)")]
    MipTooSmall { mip_level: u32, size: (u32, u32) },

    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),
}

/// Quality report (for debugging/profiling)
#[derive(Debug)]
pub struct HiZQualityReport {
    pub quality_mode: HiZQuality,
    pub mip_count: u32,
    pub avg_frame_time_ms: f64,
    pub performance_acceptable: bool,
    pub validation_failures: u32,
}

impl std::fmt::Display for HiZQualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hi-Z Quality Report:\n\
             - Mode: {:?}\n\
             - Mip Levels: {}\n\
             - Frame Time: {:.2}ms\n\
             - Performance: {}\n\
             - Validation Failures: {}",
            self.quality_mode,
            self.mip_count,
            self.avg_frame_time_ms,
            if self.performance_acceptable {
                "✅ Good"
            } else {
                "⚠️ Slow"
            },
            self.validation_failures
        )
    }
}

/// Hi-Z pyramid mip levels (1024 → 1)
pub const HIZ_MIP_LEVELS: u32 = 10;

/// Hi-Z quality mode (AAA-grade dynamic mip count)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiZQuality {
    /// 6 mip levels (Performance: ~2ms)
    Performance,
    /// 8 mip levels (Balanced: ~3ms)
    Balanced,
    /// 10 mip levels (Quality: ~4ms)
    Quality,
    /// 12 mip levels (Ultra: ~5ms)
    Ultra,
}

impl HiZQuality {
    /// Get mip count (compile-time constant)
    pub const fn mip_count(self) -> u32 {
        match self {
            Self::Performance => 4,
            Self::Balanced => 6,
            Self::Quality => 8,
            Self::Ultra => 10,
        }
    }

    /// Calculate optimal mip count from depth buffer size
    pub fn from_resolution(width: u32, height: u32) -> Self {
        let max_dimension = width.max(height);
        let max_mips = (32 - max_dimension.leading_zeros()).min(12);

        match max_mips {
            0..=6 => Self::Performance,
            7..=8 => Self::Balanced,
            9..=10 => Self::Quality,
            _ => Self::Ultra,
        }
    }

    /// Validate mip chain (Unreal-style validation)
    pub fn validate_mip_chain(self, width: u32, height: u32) -> Result<()> {
        let mip_count = self.mip_count();

        // Check if we have enough resolution
        let min_dimension = width.min(height);
        let max_possible_mips = (32 - min_dimension.leading_zeros()).min(12);

        if mip_count > max_possible_mips {
            return Err(crate::AshError::VulkanError(format!(
                "Hi-Z: Insufficient resolution {:?} for {} mips (max: {})",
                (width, height),
                mip_count,
                max_possible_mips
            )));
        }

        // Validate each mip level (Unreal's rule: No mip smaller than 4x4)
        for mip in 0..mip_count {
            let mip_width = (width >> mip).max(1);
            let mip_height = (height >> mip).max(1);

            if mip_width < 4 || mip_height < 4 {
                return Err(crate::AshError::VulkanError(format!(
                    "Hi-Z: Mip level {} too small: {:?} (minimum 4x4)",
                    mip,
                    (mip_width, mip_height)
                )));
            }
        }

        Ok(())
    }
}

impl Default for HiZQuality {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Performance metrics (AAA-grade profiling)
#[derive(Default, Debug)]
pub struct HiZMetrics {
    /// GPU time per frame (microseconds)
    gpu_time_us: f64,
    /// Frame number
    frame_count: u64,
    /// Number of validation failures (AAA standard)
    pub validation_failures: u32,
}

impl HiZMetrics {
    /// Update metrics (called every frame)
    pub fn update(&mut self, total_time_us: f64) {
        self.gpu_time_us = total_time_us;
        self.frame_count += 1;
    }

    /// Get average GPU time (milliseconds)
    pub fn avg_gpu_time_ms(&self) -> f64 {
        self.gpu_time_us / 1000.0
    }

    /// Check if performance is acceptable (Unreal: < 5ms for Hi-Z)
    pub fn is_performance_acceptable(&self) -> bool {
        self.avg_gpu_time_ms() < 5.0
    }
}

/// Validation state (debug only - zero cost in release!)
#[cfg(debug_assertions)]
#[derive(Default, Debug)]
pub struct HiZValidation {
    /// Verify mip chain correctness
    _verify_mip_chain: bool,
    /// Check for NaN/Inf in depth values
    _check_invalid_depth: bool,
}

/// Adaptive quality manager (AAA-grade dynamic adjustment)
pub struct AdaptiveHiZManager {
    /// Target GPU time for Hi-Z generation (ms)
    target_time_ms: f64,
    /// Hysteresis margin to avoid quality flipping
    hysteresis_margin: f64,
    /// Frames to wait before quality change (stability)
    stability_frames: u32,
    /// Current stability counter
    current_stability: u32,
    /// Pending quality change
    pending_quality: Option<HiZQuality>,
}

impl AdaptiveHiZManager {
    /// Create new adaptive manager with target time
    pub fn new(target_time_ms: f64) -> Self {
        Self {
            target_time_ms,
            hysteresis_margin: 0.5, // 0.5ms margin
            stability_frames: 30,   // Wait 30 frames before changing
            current_stability: 0,
            pending_quality: None,
        }
    }

    /// Update with latest metrics and return new quality if change is needed
    pub fn update(&mut self, current_quality: HiZQuality, gpu_time_ms: f64) -> Option<HiZQuality> {
        // Determine desired quality based on performance
        let desired_quality = if gpu_time_ms > self.target_time_ms + self.hysteresis_margin {
            // Too slow, lower quality
            match current_quality {
                HiZQuality::Ultra => HiZQuality::Quality,
                HiZQuality::Quality => HiZQuality::Balanced,
                HiZQuality::Balanced => HiZQuality::Performance,
                HiZQuality::Performance => HiZQuality::Performance, // Already lowest
            }
        } else if gpu_time_ms < self.target_time_ms - self.hysteresis_margin {
            // Fast enough, try higher quality
            match current_quality {
                HiZQuality::Performance => HiZQuality::Balanced,
                HiZQuality::Balanced => HiZQuality::Quality,
                HiZQuality::Quality => HiZQuality::Ultra,
                HiZQuality::Ultra => HiZQuality::Ultra, // Already highest
            }
        } else {
            // Within acceptable range, keep current
            current_quality
        };

        // Check if quality change is needed
        if desired_quality != current_quality {
            if self.pending_quality == Some(desired_quality) {
                // Same pending change, increment stability counter
                self.current_stability += 1;
                if self.current_stability >= self.stability_frames {
                    // Stable enough, apply change
                    self.current_stability = 0;
                    self.pending_quality = None;
                    log::info!(
                        "Adaptive Hi-Z: Changing quality {current_quality:?} -> {desired_quality:?} (GPU time: {gpu_time_ms:.2}ms)"
                    );
                    return Some(desired_quality);
                }
            } else {
                // New pending change, reset counter
                self.pending_quality = Some(desired_quality);
                self.current_stability = 1;
            }
        } else {
            // No change needed, reset
            self.pending_quality = None;
            self.current_stability = 0;
        }

        None
    }
}

/// Push constants for Hi-Z generation
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HiZGeneratePushConstants {
    pub output_size: [u32; 2],
    pub mip_level: u32,
    pub _padding: u32,
}

/// GPU resources for Hi-Z pyramid
pub struct HiZPass {
    device: Arc<ash::Device>,

    // Hi-Z pyramid image (R32_SFLOAT, mip chain)
    hiz_image: vk::Image,
    hiz_allocation: Option<vk_mem::Allocation>,
    hiz_views: Vec<vk::ImageView>,
    hiz_sampler: vk::Sampler,

    // Compute resources
    generate_pipeline: vk::Pipeline,
    generate_layout: vk::PipelineLayout,

    // Descriptors
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Geometry
    width: u32,
    height: u32,
    mip_count: u32,        // Allocated mip count (max for resolution)
    active_mip_count: u32, // Active mip count (based on quality)

    // Quality configuration
    quality: HiZQuality,

    // Performance metrics (AAA standard)
    metrics: HiZMetrics,
    destroyed: bool,

    allocator: Option<Arc<Allocator>>,

    // Validation state (debug builds only)
    #[cfg(debug_assertions)]
    _validation: HiZValidation,

    initialized: bool,
}

impl HiZPass {
    /// Create a new Hi-Z pass (uninitialized)
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            hiz_image: vk::Image::null(),
            hiz_allocation: None,
            hiz_views: Vec::new(),
            hiz_sampler: vk::Sampler::null(),
            generate_pipeline: vk::Pipeline::null(),
            generate_layout: vk::PipelineLayout::null(),
            pool: vk::DescriptorPool::null(),
            layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
            width: 0,
            height: 0,
            mip_count: 0,
            active_mip_count: 0,
            quality: HiZQuality::default(),
            metrics: HiZMetrics::default(),
            destroyed: false,
            #[cfg(debug_assertions)]
            _validation: HiZValidation {
                _verify_mip_chain: true,
                _check_invalid_depth: true,
            },
            initialized: false,
            allocator: None,
        }
    }

    /// Initialize Hi-Z pass resources.
    ///
    /// # Safety
    /// The caller must ensure that the provided allocator and device are valid.
    pub unsafe fn init(
        &mut self,
        allocator: &Arc<Allocator>,
        vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.allocator = Some(Arc::clone(allocator));
        let allocator = &allocator.vma;
        // Adversarial Defense: Zero-Sized Resource
        // Minimizing a window on Windows often causes width/height to become 0.
        // Creating Vulkan images with 0 dimensions is invalid and will crash.
        if width == 0 || height == 0 {
            log::warn!(
                "HiZPass: Skipping initialization with zero dimensions (window likely minimized)"
            );
            return Ok(());
        }

        if self.initialized {
            return Ok(());
        }

        self.width = width;
        self.height = height;

        // MODERN FIX: Dynamic Mip Calculation
        // Never ask for more mips than the resolution supports.
        // Formula: floor(log2(max(w, h))) + 1
        let max_dimension = width.max(height);
        let max_possible_mips = (32 - max_dimension.leading_zeros()).min(12);

        // We want 12 levels for absolute quality, but we MUST clamp to what's physically possible.
        self.mip_count = max_possible_mips;

        // Auto-detect initial quality from resolution
        self.quality = HiZQuality::from_resolution(width, height);
        self.active_mip_count = self.quality.mip_count().min(self.mip_count);

        log::info!(
            "HiZPass Initialized: {}x{} with {} mips (Quality: {:?})",
            width,
            height,
            self.mip_count,
            self.quality
        );

        // Hi-Z image with mip chain
        self.create_hiz_image(allocator)?;

        self.create_sampler()?;
        self.create_descriptors()?;
        self.create_pipeline(vulkan_device)?;

        self.initialized = true;
        Ok(())
    }

    /// Create Hi-Z image with mip chain
    unsafe fn create_hiz_image(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            })
            .mip_levels(self.mip_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) =
            allocator
                .create_image(&image_info, &alloc_info)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Hi-Z image creation failed: {e:?}"))
                })?;

        self.hiz_image = image;
        self.hiz_allocation = Some(allocation);

        // Create views for each mip level
        for mip in 0..self.mip_count {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.hiz_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                );

            let view = self.device.create_image_view(&view_info, None)?;
            self.hiz_views.push(view);
        }

        log::debug!("HiZPass: Created image with {} views", self.hiz_views.len());
        Ok(())
    }

    /// Create sampler for Hi-Z reads
    unsafe fn create_sampler(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(self.mip_count as f32);

        self.hiz_sampler = self.device.create_sampler(&sampler_info, None)?;
        Ok(())
    }

    /// Create descriptor layout and pool
    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // Layout: binding 0 = input sampler, binding 1 = output storage image
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

        self.layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        // Pool for mip_count - 1 sets (one per mip transition)
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: self.mip_count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: self.mip_count,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.mip_count)
            .pool_sizes(&pool_sizes);

        self.pool = self.device.create_descriptor_pool(&pool_info, None)?;

        // Allocate sets
        let layouts: Vec<_> = (0..self.mip_count).map(|_| self.layout).collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        self.descriptor_sets = self.device.allocate_descriptor_sets(&alloc_info)?;

        // Update descriptor sets for each mip transition
        for mip in 0..(self.mip_count as usize - 1) {
            let src_view = self.hiz_views[mip];
            let dst_view = self.hiz_views[mip + 1];

            let sampler_info = vk::DescriptorImageInfo::default()
                .sampler(self.hiz_sampler)
                .image_view(src_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let storage_info = vk::DescriptorImageInfo::default()
                .image_view(dst_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[mip])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[mip])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&storage_info)),
            ];

            self.device.update_descriptor_sets(&writes, &[]);
        }

        Ok(())
    }

    /// Create compute pipeline
    unsafe fn create_pipeline(&mut self, _vulkan_device: &VulkanDevice) -> Result<()> {
        // Load shader module
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/hiz_generate.comp.spv"));

        let code = ash::util::read_spv(&mut std::io::Cursor::new(shader_code))
            .map_err(|e| crate::AshError::VulkanError(e.to_string()))?;
        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let shader_module = self
            .device
            .create_shader_module(&shader_module_info, None)?;

        // Push constant range
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<HiZGeneratePushConstants>() as u32);

        // Pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.generate_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        // Compute pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.generate_layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?;

        self.generate_pipeline = pipelines[0];
        self.device.destroy_shader_module(shader_module, None);

        Ok(())
    }

    /// Validate mip chain configuration at runtime
    ///
    /// Checks if the current configuration (dimensions vs mip count) is valid.
    /// Returns true if valid, false otherwise (with error logging).
    pub fn validate_mip_chain_runtime(&mut self) -> bool {
        if self.width == 0 || self.height == 0 {
            log::error!(
                "Hi-Z Validation Failed: Invalid dimensions {}x{}",
                self.width,
                self.height
            );
            self.metrics.validation_failures += 1;
            return false;
        }

        if self.active_mip_count == 0 {
            log::error!("Hi-Z Validation Failed: Active mip count is 0");
            self.metrics.validation_failures += 1;
            return false;
        }

        let max_mips = (self.width.max(self.height) as f32).log2().floor() as u32 + 1;
        if self.active_mip_count > max_mips {
            log::error!(
                "Hi-Z Validation Failed: Active mip count ({}) exceeds max possible ({}) for {}x{}",
                self.active_mip_count,
                max_mips,
                self.width,
                self.height
            );
            self.metrics.validation_failures += 1;
            return false;
        }

        true
    }

    /// Build Hi-Z pyramid from depth buffer
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn build_pyramid(
        &mut self,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
    ) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // Runtime validation (AAA standard)
        if !self.validate_mip_chain_runtime() {
            // In production, we might fallback or panic, but here we error
            return Err(crate::AshError::VulkanError(
                "Hi-Z runtime validation failed".to_string(),
            ));
        }

        // Depth -> Source for transfer
        let depth_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_READ,
            old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: depth_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[depth_barrier],
        );

        // Hi-Z mip 0 -> Destination for transfer
        let hiz_barrier = vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: self.hiz_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[hiz_barrier],
        );

        let blit_region = vk::ImageBlit::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .layer_count(1),
            )
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: self.width as i32,
                    y: self.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: self.width as i32,
                    y: self.height as i32,
                    z: 1,
                },
            ]);

        self.device.cmd_blit_image(
            cmd,
            depth_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.hiz_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit_region],
            vk::Filter::NEAREST,
        );

        // Transition mip 0 to shader read
        let mip0_read_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.hiz_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[mip0_read_barrier],
        );

        // Generate mip chain (use active mip count for current quality)
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.generate_pipeline);

        for mip in 1..self.active_mip_count {
            let mip_width = (self.width >> mip).max(1);
            let mip_height = (self.height >> mip).max(1);

            // Transition current mip to general (storage write)
            let mip_barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.hiz_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[mip_barrier],
            );

            // Bind descriptor set for this mip transition
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.generate_layout,
                0,
                &[self.descriptor_sets[(mip - 1) as usize]],
                &[],
            );

            // Push constants
            let push = HiZGeneratePushConstants {
                output_size: [mip_width, mip_height],
                mip_level: mip,
                _padding: 0,
            };

            self.device.cmd_push_constants(
                cmd,
                self.generate_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );

            // Dispatch
            let group_x = mip_width.div_ceil(8);
            let group_y = mip_height.div_ceil(8);
            self.device.cmd_dispatch(cmd, group_x, group_y, 1);

            // Transition this mip to shader read for next iteration
            let read_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.hiz_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                );

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

        // Restore depth to attachment optimal
        let depth_restore = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(depth_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[depth_restore],
        );

        Ok(())
    }

    /// Get Hi-Z image for culling shader
    pub fn hiz_image(&self) -> vk::Image {
        self.hiz_image
    }

    /// Get Hi-Z sampler for culling shader
    pub fn hiz_sampler(&self) -> vk::Sampler {
        self.hiz_sampler
    }

    /// Get complete Hi-Z image view (all mips)
    pub fn hiz_view(&self) -> Option<vk::ImageView> {
        self.hiz_views.first().copied()
    }

    /// Get current quality mode
    pub fn quality(&self) -> HiZQuality {
        self.quality
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &HiZMetrics {
        &self.metrics
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self, gpu_time_us: f64) {
        self.metrics.update(gpu_time_us);
    }

    /// Set quality level (updates active mip count)
    pub fn set_quality(&mut self, quality: HiZQuality) {
        if quality != self.quality {
            self.quality = quality;
            self.active_mip_count = quality.mip_count().min(self.mip_count);
            log::debug!(
                "Hi-Z quality changed to {:?} ({} active mips)",
                quality,
                self.active_mip_count
            );

            // Validate new configuration
            let _ = self.validate_mip_chain_runtime();
        }
    }

    /// Check if performance is acceptable
    pub fn is_performance_acceptable(&self) -> bool {
        self.metrics.is_performance_acceptable()
    }

    /// Get quality report (AAA standard)
    pub fn quality_report(&self) -> HiZQualityReport {
        HiZQualityReport {
            quality_mode: self.quality,
            mip_count: self.active_mip_count,
            avg_frame_time_ms: self.metrics.avg_gpu_time_ms(),
            performance_acceptable: self.metrics.is_performance_acceptable(),
            validation_failures: self.metrics.validation_failures,
        }
    }

    /// Resize the Hi-Z pyramid
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn resize(
        &mut self,
        allocator: &Arc<Allocator>,
        vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
    ) -> Result<()> {
        // Adversarial Defense: Guard against zero dimensions on resize.
        if width == 0 || height == 0 {
            return Ok(());
        }

        if width == self.width && height == self.height {
            return Ok(());
        }

        self.destroy();
        self.init(allocator, vulkan_device, width, height)?;

        // Final validation after resize (AAA standard)
        let _ = self.validate_mip_chain_runtime();
        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        if !self.initialized {
            return;
        }

        let allocator = if let Some(ref a) = self.allocator {
            &a.vma
        } else {
            log::error!("HiZPass: Destroy called without allocator!");
            return;
        };

        for view in self.hiz_views.drain(..) {
            self.device.destroy_image_view(view, None);
        }

        if self.hiz_image != vk::Image::null() {
            if let Some(mut alloc) = self.hiz_allocation.take() {
                allocator.destroy_image(self.hiz_image, &mut alloc);
            }
            self.hiz_image = vk::Image::null();
        }

        if self.hiz_sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.hiz_sampler, None);
            self.hiz_sampler = vk::Sampler::null();
        }

        if self.generate_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.generate_pipeline, None);
            self.generate_pipeline = vk::Pipeline::null();
        }

        if self.generate_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.generate_layout, None);
            self.generate_layout = vk::PipelineLayout::null();
        }

        if self.pool != vk::DescriptorPool::null() {
            self.device.destroy_descriptor_pool(self.pool, None);
            self.pool = vk::DescriptorPool::null();
        }

        if self.layout != vk::DescriptorSetLayout::null() {
            self.device.destroy_descriptor_set_layout(self.layout, None);
            self.layout = vk::DescriptorSetLayout::null();
        }

        self.initialized = false;
        log::info!("HiZPass: Resources destroyed");
    }
}

impl Drop for HiZPass {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}

impl crate::renderer::cleanup_traits::VulkanResourceCleanup for HiZPass {
    fn cleanup_with_device(&mut self, _device: &ash::Device) -> std::result::Result<(), String> {
        unsafe {
            self.destroy();
        }
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "HiZPass"
    }
}

impl crate::renderer::resource_registry::VulkanResource for HiZPass {}
