//! Post-Processing System
//!
//! Self-contained system for HDR rendering, tonemapping, and bloom effects.
//! Fully owns all GPU resources (framebuffers, pipelines, descriptors).

use ash::vk;
use std::sync::Arc;

use crate::{
    renderer::{fullscreen_pass, fullscreen_pass::PostProcessPushConstants, hdr_framebuffer},
    vulkan::{self, Allocator},
    AshError, Result,
};

/// Post-processing system managing HDR, tonemapping, and bloom
pub struct PostProcessingSystem {
    // Core resources
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,

    // HDR rendering
    hdr_framebuffer: Option<hdr_framebuffer::HdrFramebuffer>,

    // Fullscreen pass infrastructure
    fullscreen_pass: Option<fullscreen_pass::FullscreenPass>,

    // Tonemapping pipeline
    post_pipeline: Option<vulkan::Pipeline>,

    // Descriptor management
    post_descriptor_pool: vk::DescriptorPool,
    post_descriptor_sets: Vec<vk::DescriptorSet>,
    post_sampler: vk::Sampler,

    // Post-processing settings
    pub tonemapping_enabled: bool,
    tonemapping_exposure: f32,
    tonemapping_gamma: f32,
    pub bloom_enabled: bool,
    bloom_intensity: f32,

    // Black texture for bloom placeholder
    black_texture_view: vk::ImageView,

    // Swapchain info
    _swapchain_format: vk::Format,
    frames_in_flight: usize,
    _is_headless: bool,
    extent: vk::Extent2D,
}

impl PostProcessingSystem {
    /// Create a new post-processing system
    ///
    /// # Safety
    /// - `device` and `allocator` must remain valid for the lifetime of this system
    /// - `black_texture_view` must be a valid 1x1 black texture view
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        resolution: vk::Extent2D,
        swapchain_format: vk::Format,
        frames_in_flight: usize,
        black_texture_view: vk::ImageView,
        is_headless: bool,
    ) -> Result<Self> {
        log::info!(
            "Initializing PostProcessingSystem ({}x{})",
            resolution.width,
            resolution.height
        );

        // Create post-processing sampler
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
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR);

        let post_sampler = device
            .create_sampler(&sampler_info, None)
            .map_err(|e| AshError::VulkanError(format!("Failed to create post sampler: {e}")))?;

        // Create HDR framebuffer
        let hdr_framebuffer = hdr_framebuffer::HdrFramebuffer::new(
            Arc::clone(&device),
            Arc::clone(&allocator),
            resolution.width,
            resolution.height,
        )?;

        // Create fullscreen pass
        let final_layout = if is_headless {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::PRESENT_SRC_KHR
        };

        let fullscreen_pass = fullscreen_pass::FullscreenPass::new(
            Arc::clone(&device),
            swapchain_format,
            final_layout,
        )?;

        let mut system = Self {
            device: Arc::clone(&device),
            allocator: Arc::clone(&allocator),
            hdr_framebuffer: Some(hdr_framebuffer),
            fullscreen_pass: Some(fullscreen_pass),
            post_pipeline: None,
            post_descriptor_pool: vk::DescriptorPool::null(),
            post_descriptor_sets: Vec::new(),
            post_sampler,
            tonemapping_enabled: true,
            tonemapping_exposure: 1.2,
            tonemapping_gamma: 2.2,
            bloom_enabled: true,
            bloom_intensity: 0.1,
            black_texture_view,
            _swapchain_format: swapchain_format,
            frames_in_flight,
            _is_headless: is_headless,
            extent: resolution,
        };

        // Create descriptor pool and sets
        system.create_post_descriptors()?;

        // Create tonemapping pipeline
        system.recreate_post_pipeline()?;

        log::info!("PostProcessingSystem initialized successfully");
        Ok(system)
    }

    /// Recreate resources for a new resolution
    ///
    /// # Safety
    /// - GPU must be idle
    /// - All command buffers using old resources must be complete
    pub unsafe fn resize(&mut self, resolution: vk::Extent2D) -> Result<()> {
        log::info!(
            "Resizing PostProcessingSystem to {}x{}",
            resolution.width,
            resolution.height
        );

        // Recreate HDR framebuffer
        let hdr = hdr_framebuffer::HdrFramebuffer::new(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution.width,
            resolution.height,
        )?;
        self.hdr_framebuffer = Some(hdr);
        self.extent = resolution;

        // Update descriptor sets to point to new HDR buffer
        self.update_post_descriptors()?;

        log::info!("PostProcessingSystem resized successfully");
        Ok(())
    }

    /// Render post-processing effects
    ///
    /// # Safety
    /// - `cmd` must be in recording state
    /// - `input_view` must be a valid image view in SHADER_READ_ONLY_OPTIMAL or GENERAL layout
    /// - `framebuffer` must be compatible with the fullscreen pass render pass
    #[inline]
    pub unsafe fn render(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        input_view: vk::ImageView,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
    ) -> Result<()> {
        // Update descriptor set for this frame with the current input
        self.update_descriptor_for_frame(frame_index, input_view)?;

        let fullscreen_pass = self.fullscreen_pass.as_ref().ok_or_else(|| {
            AshError::RenderPassMissing("Fullscreen pass not initialized".to_string())
        })?;

        let pipeline = self
            .post_pipeline
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Post pipeline not initialized".to_string()))?;

        // Begin render pass
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(fullscreen_pass.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        self.device
            .cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

        // Bind pipeline
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

        // Bind descriptor set
        if frame_index < self.post_descriptor_sets.len() {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                fullscreen_pass.pipeline_layout(),
                0,
                &[self.post_descriptor_sets[frame_index]],
                &[],
            );
        }

        // Push constants
        let push_constants = PostProcessPushConstants {
            exposure: self.tonemapping_exposure,
            gamma: self.tonemapping_gamma,
            bloom_intensity: self.bloom_intensity,
            tonemapping_enabled: if self.tonemapping_enabled { 1.0 } else { 0.0 },
        };

        let push_bytes = bytemuck::bytes_of(&push_constants);
        self.device.cmd_push_constants(
            cmd,
            fullscreen_pass.pipeline_layout(),
            vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        // Draw fullscreen triangle
        self.device.cmd_draw(cmd, 3, 1, 0, 0);

        self.device.cmd_end_render_pass(cmd);

        Ok(())
    }

    /// Get HDR image view for rendering
    pub fn hdr_image_view(&self) -> Option<vk::ImageView> {
        self.hdr_framebuffer.as_ref().map(|hdr| hdr.view())
    }

    /// Get HDR image for rendering
    pub fn hdr_image(&self) -> Option<vk::Image> {
        self.hdr_framebuffer.as_ref().map(|hdr| hdr.image())
    }

    /// Get HDR format
    pub fn hdr_format(&self) -> Option<vk::Format> {
        self.hdr_framebuffer.as_ref().map(|hdr| hdr.format())
    }

    /// Get the render pass for post-processing
    pub fn render_pass(&self) -> vk::RenderPass {
        self.fullscreen_pass
            .as_ref()
            .map(|p| p.render_pass())
            .unwrap_or(vk::RenderPass::null())
    }

    // Exposure control
    pub fn set_tonemapping_exposure(&mut self, exposure: f32) {
        self.tonemapping_exposure = exposure.max(0.0);
    }

    pub fn tonemapping_exposure(&self) -> f32 {
        self.tonemapping_exposure
    }

    // Gamma control
    pub fn set_tonemapping_gamma(&mut self, gamma: f32) {
        self.tonemapping_gamma = gamma.max(0.1);
    }

    pub fn tonemapping_gamma(&self) -> f32 {
        self.tonemapping_gamma
    }

    // Bloom control
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        self.bloom_intensity = intensity.clamp(0.0, 2.0);
    }

    pub fn bloom_intensity(&self) -> f32 {
        self.bloom_intensity
    }

    /// Create descriptor pool and allocate descriptor sets
    fn create_post_descriptors(&mut self) -> Result<()> {
        // Cleanup old pool if exists
        if self.post_descriptor_pool != vk::DescriptorPool::null() {
            unsafe {
                self.device
                    .destroy_descriptor_pool(self.post_descriptor_pool, None);
            }
        }

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: (self.frames_in_flight * 3) as u32, // 3 bindings per frame
        }];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.frames_in_flight as u32)
            .pool_sizes(&pool_sizes);

        self.post_descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create post descriptor pool: {e}"))
                })?
        };

        let fullscreen_pass = self
            .fullscreen_pass
            .as_ref()
            .ok_or_else(|| AshError::RenderPassMissing("Fullscreen pass missing".to_string()))?;

        let layouts = vec![fullscreen_pass.descriptor_set_layout(); self.frames_in_flight];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.post_descriptor_pool)
            .set_layouts(&layouts);

        self.post_descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to allocate post descriptor sets: {e}"))
                })?
        };

        self.update_post_descriptors()?;

        Ok(())
    }

    /// Update all descriptor sets with current HDR buffer
    fn update_post_descriptors(&mut self) -> Result<()> {
        let hdr = self
            .hdr_framebuffer
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("HDR framebuffer not initialized".to_string()))?;

        let color_view = hdr.view();
        let sampler = hdr.sampler();
        let bloom_view = self.black_texture_view; // Placeholder until bloom is implemented

        for descriptor_set in &self.post_descriptor_sets {
            let color_info = vk::DescriptorImageInfo {
                sampler,
                image_view: color_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let bloom_info = vk::DescriptorImageInfo {
                sampler,
                image_view: bloom_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let color_infos = [color_info];
            let bloom_infos = [bloom_info];

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&color_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&bloom_infos),
            ];

            unsafe {
                self.device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        Ok(())
    }

    /// Update a single descriptor set for a specific frame
    fn update_descriptor_for_frame(
        &self,
        frame_index: usize,
        input_view: vk::ImageView,
    ) -> Result<()> {
        if frame_index >= self.post_descriptor_sets.len() {
            return Err(AshError::VulkanError(format!(
                "Invalid frame index: {} >= {}",
                frame_index,
                self.post_descriptor_sets.len()
            )));
        }

        let descriptor_set = self.post_descriptor_sets[frame_index];
        let bloom_view = self.black_texture_view;

        let color_info = vk::DescriptorImageInfo {
            sampler: self.post_sampler,
            image_view: input_view,
            image_layout: vk::ImageLayout::GENERAL, // VSR outputs to GENERAL
        };

        let bloom_info = vk::DescriptorImageInfo {
            sampler: self.post_sampler,
            image_view: bloom_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let color_infos = [color_info];
        let bloom_infos = [bloom_info];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&color_infos),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&bloom_infos),
        ];

        unsafe {
            self.device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        Ok(())
    }

    /// Recreate the tonemapping pipeline
    fn recreate_post_pipeline(&mut self) -> Result<()> {
        let fullscreen_pass = self
            .fullscreen_pass
            .as_ref()
            .ok_or_else(|| AshError::RenderPassMissing("Fullscreen pass missing".to_string()))?;

        let mut builder = vulkan::Pipeline::builder(Arc::clone(&self.device))
            .with_layout(fullscreen_pass.pipeline_layout())
            .with_render_pass(fullscreen_pass.render_pass())
            .with_extent(self.extent) // Dynamic extent from resize/new
            .with_cull_mode(vk::CullModeFlags::NONE);

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/postprocess.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/tonemapping.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

        let pipeline = builder.build()?;
        self.post_pipeline = Some(pipeline);

        Ok(())
    }
}

impl Drop for PostProcessingSystem {
    fn drop(&mut self) {
        unsafe {
            log::debug!("Destroying PostProcessingSystem");

            // Pipeline is dropped automatically via RAII
            self.post_pipeline = None;

            // Destroy descriptor pool (frees all sets)
            if self.post_descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.post_descriptor_pool, None);
            }

            // Destroy sampler
            self.device.destroy_sampler(self.post_sampler, None);

            // Fullscreen pass and HDR framebuffer are dropped automatically via RAII
            self.fullscreen_pass = None;
            self.hdr_framebuffer = None;
        }
    }
}
