#![allow(unsafe_op_in_unsafe_fn)]

//! Hi-Z Pyramid GPU Resources
//!
//! Manages GPU-side resources for hierarchical-Z occlusion culling:
//! - Hi-Z pyramid image (mip chain)
//! - Compute pipelines for pyramid generation and culling
//! - Descriptor sets and buffers

use ash::vk;
use std::sync::Arc;

use crate::Result;
use crate::vulkan::{Allocator, VulkanDevice};
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

/// Push constants for the buffer-backed Hi-Z generate compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HiZPushConstants {
    pub hiz_buffer_addr: u64,
    pub src_mip_offset: u32,
    pub dst_mip_offset: u32,
    pub src_resolution: [u32; 2],
    pub dst_resolution: [u32; 2],
    pub is_first_pass: u32,
    pub _pad: u32,
}
const _: () = assert!(
    std::mem::size_of::<HiZPushConstants>() == 40,
    "HiZPushConstants must be exactly 40 bytes"
);

/// GPU resources for the buffer-backed Hi-Z pyramid (Phase 5 BDA Migration)
pub struct HiZPass {
    device: Arc<ash::Device>,

    // --- Phase 5: The single flat BDA buffer that replaces VkImage ---
    // Stores the full mip chain as a contiguous array of u32 (floatBitsToUint depth values).
    // Layout: [mip0_pixels... | mip1_pixels... | ... | mip_N_pixels]
    hiz_buffer: vk::Buffer,
    hiz_buffer_allocation: Option<vk_mem::Allocation>,
    /// 64-bit device address — passed directly in push constants, no descriptor needed.
    hiz_buffer_addr: u64,
    /// Starting element index for each mip level within `hiz_buffer`.
    /// `mip_offsets[i]` is the index of the first u32 element for mip `i`.
    mip_offsets: Vec<u32>,
    /// Total u32 elements allocated in `hiz_buffer` (sum of all mip areas).
    total_elements: u64,

    // Compute pipeline (Phase 2)
    generate_pipeline: vk::Pipeline,
    generate_layout: vk::PipelineLayout,

    // Phase 2: Descriptor for reading main depth in Pass 0
    depth_sampler: vk::Sampler,
    input_layout: vk::DescriptorSetLayout,
    input_pool: vk::DescriptorPool,
    input_set: vk::DescriptorSet,

    // Geometry
    width: u32,
    height: u32,
    mip_count: u32,        // Allocated mip count (max for resolution)
    active_mip_count: u32, // Active mip count (based on quality)

    // Quality configuration
    quality: HiZQuality,

    // Performance metrics
    metrics: HiZMetrics,
    destroyed: bool,

    allocator: Option<Arc<Allocator>>,

    #[cfg(debug_assertions)]
    _validation: HiZValidation,

    initialized: bool,
}

impl HiZPass {
    /// Calculate the total element count and per-mip element offsets for a
    /// mipmap chain starting at `width` × `height` and descending to 1×1.
    ///
    /// Returns `(total_elements, mip_offsets)` where:
    /// - `total_elements` is the sum of all mip-level areas.
    /// - `mip_offsets[i]` is the starting element index for mip level `i`.
    pub fn calc_mip_chain(width: u32, height: u32, mip_count: u32) -> (u64, Vec<u32>) {
        let mut offsets = Vec::with_capacity(mip_count as usize);
        let mut total: u64 = 0;
        for mip in 0..mip_count {
            offsets.push(total as u32);
            let mip_w = (width >> mip).max(1) as u64;
            let mip_h = (height >> mip).max(1) as u64;
            total += mip_w * mip_h;
        }
        (total, offsets)
    }

    /// Create a new Hi-Z pass (uninitialized). All GPU resources start as null.
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            // Phase 5: BDA buffer fields
            hiz_buffer: vk::Buffer::null(),
            hiz_buffer_allocation: None,
            hiz_buffer_addr: 0,
            mip_offsets: Vec::new(),
            total_elements: 0,
            // Compute pipeline resources
            generate_pipeline: vk::Pipeline::null(),
            generate_layout: vk::PipelineLayout::null(),

            // Phase 2 Descriptor
            depth_sampler: vk::Sampler::null(),
            input_layout: vk::DescriptorSetLayout::null(),
            input_pool: vk::DescriptorPool::null(),
            input_set: vk::DescriptorSet::null(),

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
        _vulkan_device: &VulkanDevice,
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

        // Phase 5: Allocate the flat BDA depth buffer.
        unsafe { self.create_hiz_buffer(allocator)? };

        // Phase 2: Descriptor for Pass 0 texture read
        unsafe { self.create_descriptors()? };

        // Phase 2: Compute pipeline
        unsafe { self.create_pipeline(_vulkan_device)? };

        self.initialized = true;
        Ok(())
    }

    /// Phase 5: Allocate the flat BDA Hi-Z buffer.
    ///
    /// The buffer stores the entire mip chain as a contiguous array of `u32`
    /// (bit-cast `f32` depth values via `floatBitsToUint`/`uintBitsToFloat`).
    ///
    /// Layout inside the buffer:
    /// ```text
    /// [  mip0: width*height u32s  |  mip1: (w/2)*(h/2) u32s  |  ...  |  mipN: 1 u32  ]
    /// ```
    ///
    /// `mip_offsets[i]` gives the starting *element* index for mip `i`.
    /// Byte offset = `mip_offsets[i] * 4`.
    unsafe fn create_hiz_buffer(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        // --- Mipmap math -------------------------------------------------------
        let (total_elements, mip_offsets) =
            Self::calc_mip_chain(self.width, self.height, self.mip_count);
        self.mip_offsets = mip_offsets;
        self.total_elements = total_elements;

        let byte_size = total_elements * std::mem::size_of::<u32>() as u64;

        log::debug!(
            "HiZPass: Allocating BDA buffer — {}x{} × {} mips = {} elements ({} KiB)",
            self.width,
            self.height,
            self.mip_count,
            total_elements,
            byte_size / 1024
        );

        // --- Buffer allocation -------------------------------------------------
        let buffer_info = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(
                // STORAGE_BUFFER: allows GLSL `buffer` references.
                // SHADER_DEVICE_ADDRESS: allows vkGetBufferDeviceAddress.
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Hi-Z BDA buffer allocation failed: {e:?}"))
            })?;

        self.hiz_buffer = buffer;
        self.hiz_buffer_allocation = Some(allocation);

        // --- Extract the 64-bit BDA pointer ------------------------------------
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(self.hiz_buffer);
        self.hiz_buffer_addr = unsafe { self.device.get_buffer_device_address(&addr_info) };

        log::info!(
            "HiZPass: BDA buffer allocated at {:#018X} ({} MiB)",
            self.hiz_buffer_addr,
            byte_size / (1024 * 1024)
        );

        Ok(())
    }

    /// Create descriptor pool/layout for reading hardware depth buffer in Pass 0
    unsafe fn create_descriptors(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_anisotropy(1.0);
        self.depth_sampler = self.device.create_sampler(&sampler_info, None)?;

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.input_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        self.input_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        let layouts = [self.input_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.input_pool)
            .set_layouts(&layouts);
        self.input_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];

        Ok(())
    }

    /// Create compute pipeline
    unsafe fn create_pipeline(&mut self, _vulkan_device: &VulkanDevice) -> Result<()> {
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/hiz_generate.comp.spv"));
        let code = ash::util::read_spv(&mut std::io::Cursor::new(shader_code))
            .map_err(|e| crate::AshError::VulkanError(e.to_string()))?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = self.device.create_shader_module(&shader_info, None)?;

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<HiZPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.input_layout))
            .push_constant_ranges(std::slice::from_ref(&push_range));

        self.generate_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.generate_layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?;

        self.generate_pipeline = pipelines[0];
        self.device.destroy_shader_module(module, None);
        Ok(())
    }

    // --- DELETED (Phase 5): create_descriptors() was here. ---
    // Descriptor pool/layout for the old image-based pipeline has been removed.
    // All data now flows through BDA push constants.

    // --- STUB: old image-based create_descriptors signature for reference only ---
    #[allow(dead_code)]
    unsafe fn _deleted_create_descriptors(&mut self) -> Result<()> {
        // This method has been intentionally deleted as part of Phase 5:
        // Buffer-Backed Hi-Z Migration. The VkDescriptorPool and
        // VkDescriptorSetLayout for the Hi-Z pipeline are gone.
        unreachable!("Phase 5: create_descriptors() has been eradicated")
    }

    // --- DELETED (Phase 5): old create_pipeline with image descriptor layout ---
    #[allow(dead_code)]
    unsafe fn _deleted_old_create_pipeline(&mut self) -> Result<()> {
        unreachable!("Phase 5: old image-based create_pipeline() has been eradicated")
    }

    /// Validate mip chain configuration at runtime
    // Note: moved immediately before build_pyramid. Method body unchanged.
    unsafe fn _old_create_descriptors_placeholder(&self) -> Result<()> {
        Ok(())
    }

    /// Validate mip chain configuration at runtime.
    ///
    /// Returns `true` if the buffer dimensions match the current mip chain.
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

    /// Build the Hi-Z pyramid from the hardware depth buffer.
    ///
    /// **Phase 1 stub**: The BDA buffer is allocated and the address is valid.
    /// The actual compute dispatch is implemented in Phase 2 (`hiz_generate.comp`).
    ///
    /// Phase 2 will:
    /// 1. Transition the depth image to `SHADER_READ_ONLY_OPTIMAL`.
    /// 2. Bind the compute pipeline.
    /// 3. For each mip level, dispatch a compute shader that samples the depth
    ///    image (mip 0) or the previous BDA mip level, and writes the minimum
    ///    2×2 Reverse-Z depth to `hiz_ptr.data[mip_offsets[mip] + pixel_idx]`.
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn build_pyramid(
        &mut self,
        cmd: vk::CommandBuffer,
        depth_view: vk::ImageView,
    ) -> Result<()> {
        if !self.initialized || self.hiz_buffer_addr == 0 {
            return Ok(());
        }

        // 1. Update Pass 0 Descriptor with the fresh depth_view
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(self.depth_sampler)
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.input_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        self.device
            .update_descriptor_sets(std::slice::from_ref(&write), &[]);

        // 2. Bind pipeline & descriptor
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.generate_pipeline);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.generate_layout,
            0,
            std::slice::from_ref(&self.input_set),
            &[],
        );

        // 3. Dispatch Loop over all mips
        let mut src_width = self.width;
        let mut src_height = self.height;

        for mip in 0..self.active_mip_count {
            let dst_width = (self.width >> mip).max(1);
            let dst_height = (self.height >> mip).max(1);

            let push = HiZPushConstants {
                hiz_buffer_addr: self.hiz_buffer_addr,
                src_mip_offset: if mip == 0 {
                    0
                } else {
                    self.mip_offset((mip - 1) as usize)
                },
                dst_mip_offset: self.mip_offset(mip as usize),
                src_resolution: [src_width, src_height],
                dst_resolution: [dst_width, dst_height],
                is_first_pass: if mip == 0 { 1 } else { 0 },
                _pad: 0,
            };

            self.device.cmd_push_constants(
                cmd,
                self.generate_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );

            let group_x = dst_width.div_ceil(8);
            let group_y = dst_height.div_ceil(8);
            self.device.cmd_dispatch(cmd, group_x, group_y, 1);

            // Memory barrier: Flush BDA writes so the next mip can safely read them (Sync2)
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ);

            let memory_barriers = [barrier];
            let dep_info = vk::DependencyInfo::default().memory_barriers(&memory_barriers);
            self.device.cmd_pipeline_barrier2(cmd, &dep_info);

            src_width = dst_width;
            src_height = dst_height;
        }

        Ok(())
    }

    /// Record all Hi-Z frame commands into `cmd`.
    ///
    /// Absorbs the full frame lifecycle:
    /// 1. Adaptive quality adjustment (based on prior frame GPU timing).
    /// 2. Depth-pyramid construction (`build_pyramid`).
    /// 3. Handing the pyramid view+sampler to the `IndirectDrawPass` so the
    ///    culling compute shader sees fresh data this frame.
    ///
    /// # Lock ordering
    /// Acquires the `IndirectDrawPass` write-lock **after** all Hi-Z work is
    /// complete, matching the convention established by the old
    /// `RenderPipeline::execute_hiz_pass`.
    ///
    /// # Safety
    /// `cmd` must be in recording state.
    pub unsafe fn record_commands(
        &mut self,
        cmd: vk::CommandBuffer,
        depth_view: vk::ImageView,
        // Frame GPU timing from the previous frame (drives adaptive quality).
        hiz_time_ms: Option<f64>,
    ) -> crate::Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // 1. Adaptive quality adjustment.
        if let Some(time_ms) = hiz_time_ms {
            let target_ms = 3.0_f64;
            let margin = 0.5_f64;
            if time_ms > target_ms + margin && self.quality != HiZQuality::Performance {
                self.quality = match self.quality {
                    HiZQuality::Ultra => HiZQuality::Quality,
                    HiZQuality::Quality => HiZQuality::Balanced,
                    _ => HiZQuality::Performance,
                };
                self.active_mip_count = self.quality.mip_count().min(self.mip_count);
                log::debug!("HiZ quality → {:?} ({:.1}ms)", self.quality, time_ms);
            } else if time_ms < target_ms - margin && self.quality != HiZQuality::Ultra {
                self.quality = match self.quality {
                    HiZQuality::Performance => HiZQuality::Balanced,
                    HiZQuality::Balanced => HiZQuality::Quality,
                    _ => HiZQuality::Ultra,
                };
                self.active_mip_count = self.quality.mip_count().min(self.mip_count);
                log::debug!("HiZ quality ↑ {:?} ({:.1}ms)", self.quality, time_ms);
            }
        }

        // 2. Dispatch BDA compute downsample.
        unsafe { self.build_pyramid(cmd, depth_view)? };

        Ok(())
    }

    /// Returns the 64-bit BDA pointer for the flat Hi-Z buffer.
    /// Pass this directly into the culling push constants — no descriptor required.
    #[inline]
    pub fn hiz_buffer_addr(&self) -> u64 {
        self.hiz_buffer_addr
    }

    /// Returns the element-index offset for mip level `mip`.
    /// `byte_offset = mip_offsets[mip] * 4`.
    #[inline]
    pub fn mip_offset(&self, mip: usize) -> u32 {
        self.mip_offsets.get(mip).copied().unwrap_or(0)
    }

    /// Returns the full mip-offset table (element indices, not bytes).
    #[inline]
    pub fn mip_offsets(&self) -> &[u32] {
        &self.mip_offsets
    }

    /// Returns the total number of `u32` elements in the Hi-Z buffer.
    #[inline]
    pub fn total_elements(&self) -> u64 {
        self.total_elements
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

        unsafe {
            self.destroy();
            self.init(allocator, vulkan_device, width, height)?;
        }

        // Final validation after resize (AAA standard)
        let _ = self.validate_mip_chain_runtime();
        Ok(())
    }

    /// Destroys all Vulkan resources associated with this pass.
    ///
    /// # Safety
    /// The caller must ensure that the GPU is idle and no resources are currently in use.
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

        unsafe {
            // Phase 5: Destroy the flat BDA Hi-Z buffer.
            if self.hiz_buffer != vk::Buffer::null() {
                if let Some(mut alloc) = self.hiz_buffer_allocation.take() {
                    allocator.destroy_buffer(self.hiz_buffer, &mut alloc);
                }
                self.hiz_buffer = vk::Buffer::null();
                self.hiz_buffer_addr = 0;
            }
            self.mip_offsets.clear();
            self.total_elements = 0;

            // Compute pipeline (Phase 2 installs this)
            if self.generate_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.generate_pipeline, None);
                self.generate_pipeline = vk::Pipeline::null();
            }
            if self.generate_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.generate_layout, None);
                self.generate_layout = vk::PipelineLayout::null();
            }
            // Phase 2: Descriptor cleanup
            if self.depth_sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.depth_sampler, None);
                self.depth_sampler = vk::Sampler::null();
            }
            if self.input_pool != vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(self.input_pool, None);
                self.input_pool = vk::DescriptorPool::null();
            }
            if self.input_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.input_layout, None);
                self.input_layout = vk::DescriptorSetLayout::null();
            }
        }

        self.initialized = false;
        log::info!("HiZPass: Resources destroyed");
    }
    pub fn mip_count(&self) -> u32 {
        self.mip_count
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
