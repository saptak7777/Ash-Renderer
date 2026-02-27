use crate::Result;
use crate::renderer::passes::fullscreen::FullscreenPass;
use crate::renderer::passes::fullscreen::PostProcessPushConstants;
use crate::renderer::passes::temporal_aa::{ConfigMetrics, TaaPass, TaaPushConstants};
use crate::renderer::resources::HdrSystem;
use crate::renderer::resources::Resources;
use crate::renderer::util::frame_state::FrameState;
use crate::vulkan;
use crate::vulkan::SwapchainWrapper;
use ash::vk;
use bytemuck;
use std::sync::Arc;

use crate::renderer::frame::Frame;

/// Configuration for post-processing effects.
#[derive(Clone, Copy, Debug)]
pub struct PostProcessConfig {
    pub exposure: f32,
    pub gamma: f32,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            gamma: 2.2,
        }
    }
}

/// High-level system for managing post-processing effects and resources.
/// Wraps FullscreenPass and manages its own pipelines and descriptors.
pub struct PostProcessSystem {
    device: Arc<ash::Device>,
    fullscreen_pass: FullscreenPass,
    pub config: PostProcessConfig,

    // Resource Management
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline: Option<vulkan::Pipeline>,

    // Native TAA pass (1:1 resolution, no upscaling).
    // Initialized on first resize() call when dimensions are known.
    pub taa_pass: Option<TaaPass>,
}

impl PostProcessSystem {
    pub fn new(
        device: Arc<ash::Device>,
        image_count: usize,
        _swapchain_extent: vk::Extent2D,
        output_format: vk::Format,
    ) -> Result<Self> {
        log::info!("Initializing PostProcessSystem");

        let fullscreen_pass = unsafe { FullscreenPass::new(Arc::clone(&device), output_format)? };

        Ok(Self {
            device,
            fullscreen_pass,
            config: PostProcessConfig::default(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::with_capacity(image_count),
            pipeline: None,
            taa_pass: None, // Initialized in resize() once dimensions are known
        })
    }

    pub fn resize(&mut self, image_count: usize, extent: vk::Extent2D) -> Result<()> {
        // Pipeline cleanup handled by RAII in vulkan::Pipeline
        self.pipeline = None;

        let pipeline = match self
            .fullscreen_pass
            .create_pipeline(&self.device, extent, "main")
        {
            Ok(p) => p,
            Err(e) => {
                log::error!("Failed to create post-process pipeline: {e}");
                return Err(e);
            }
        };
        self.pipeline = Some(pipeline);

        if self.descriptor_sets.len() != image_count {
            self.reallocate_descriptors(image_count)?;
        }

        Ok(())
    }

    /// Initialize or re-initialize the post-processing system.
    ///
    /// Must be called after resize() whenever the render resolution changes.
    /// Separate from resize() because it requires the VMA allocator.
    pub fn init(&mut self, allocator: &vk_mem::Allocator, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        match &mut self.taa_pass {
            Some(taa) => {
                // Re-initialize existing pass with new dimensions
                taa.init(allocator, width, height)?;
            }
            None => {
                // First-time creation
                let mut taa = TaaPass::new(Arc::clone(&self.device))?;
                taa.init(allocator, width, height)?;
                self.taa_pass = Some(taa);
            }
        }

        log::info!("TAA pass initialized/resized to {width}x{height}");
        Ok(())
    }

    /// Explicitly destroy GPU resources that require the VMA allocator.
    ///
    /// Must be called during shutdown before the allocator is destroyed.
    pub fn destroy_resources(&mut self, allocator: &vk_mem::Allocator) {
        if let Some(mut taa) = self.taa_pass.take() {
            unsafe {
                taa.destroy_resources(allocator);
            }
        }
        log::info!("PostProcessSystem resources destroyed.");
    }

    fn reallocate_descriptors(&mut self, count: usize) -> Result<()> {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }

            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (count * 3) as u32,
            }];

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(count as u32);

            self.descriptor_pool = self.device.create_descriptor_pool(&pool_info, None)?;

            let layouts = vec![self.fullscreen_pass.descriptor_set_layout(); count];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);

            self.descriptor_sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        }
        Ok(())
    }

    pub fn update_descriptor_set(
        &self,
        image_index: usize,
        input_view: vk::ImageView,
        bloom_view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        if image_index >= self.descriptor_sets.len() {
            return;
        }

        let input_info = vk::DescriptorImageInfo::default()
            .image_view(input_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler);

        let bloom_info = vk::DescriptorImageInfo::default()
            .image_view(bloom_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[image_index])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&input_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[image_index])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&bloom_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[image_index])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    // ─── PostProcessConfig Setters ────────────────────────────────────────────

    /// Replace the entire post-process configuration in one call.
    pub fn set_config(&mut self, config: PostProcessConfig) {
        self.config = config;
    }

    /// Set the exposure value (clamped to `[0.1, 10.0]`).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.config.exposure = exposure.clamp(0.1, 10.0);
    }

    /// Query the current exposure value.
    pub fn exposure(&self) -> f32 {
        self.config.exposure
    }

    /// Set the gamma value (clamped to `[1.0, 3.0]`).
    pub fn set_gamma(&mut self, gamma: f32) {
        self.config.gamma = gamma.clamp(1.0, 3.0);
    }

    /// Query the current gamma value.
    pub fn gamma(&self) -> f32 {
        self.config.gamma
    }
}

/// Context for post-processing and upscaling.
pub struct PostProcessContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub frame_index: usize,
    pub image_index: usize,
    pub resources: &'a Resources,
    pub frame: &'a Frame,
    pub swapchain: &'a SwapchainWrapper,
    pub hdr: Option<&'a HdrSystem>,
    pub taa_config: crate::renderer::passes::temporal_aa::TaaConfig,
    pub taa_metrics: Option<&'a mut ConfigMetrics>,
    /// Authoritative per-frame temporal state — single source of truth for jitter.
    pub frame_state: &'a FrameState,
    /// Direct depth image view (SHADER_READ_ONLY_OPTIMAL) for TAA
    pub depth_view: vk::ImageView,
    /// Direct motion vector image view (SHADER_READ_ONLY_OPTIMAL) for TAA
    pub motion_view: vk::ImageView,
    /// Authoritative bloom settings from BloomFeature
    pub bloom_view: vk::ImageView,
    pub bloom_intensity: f32,
}

impl PostProcessSystem {
    /// Record the complete post-processing chain commands.
    ///
    /// Pipeline: Geometry → TAA Resolve → Tone Mapping → Swapchain
    ///
    /// VSR is intentionally bypassed while we validate native TAA stability.
    pub fn record_commands(&mut self, ctx: PostProcessContext) -> Result<()> {
        let hdr = ctx
            .hdr
            .expect("HdrSystem is mandatory for Pure Renderer remediation");
        let raw_hdr_view = hdr.view();

        // ── 1. TAA Resolve Phase (STRICT NATIVE) ──────────────────────────
        // Run the TAA compute shader before tonemapping. The resolved output
        // is in SHADER_READ_ONLY_OPTIMAL after resolve() returns.
        let resolved_view = if let Some(taa) = &mut self.taa_pass {
            if taa.is_initialized()
                && ctx.depth_view != vk::ImageView::null()
                && ctx.motion_view != vk::ImageView::null()
            {
                let extent = ctx.swapchain.extent;
                let push = TaaPushConstants {
                    width: extent.width as f32,
                    height: extent.height as f32,
                    jitter_x: ctx.frame_state.jitter.x,
                    jitter_y: ctx.frame_state.jitter.y,
                    prev_jitter_x: ctx.frame_state.prev_jitter.x,
                    prev_jitter_y: ctx.frame_state.prev_jitter.y,
                    blend_factor: ctx.taa_config.blend_factor,
                    clamping_gamma: ctx.taa_config.quality.clamping_gamma(),
                    depth_threshold: ctx.taa_config.depth_threshold,
                    anti_flicker: if ctx.taa_config.anti_flicker { 1 } else { 0 },
                };

                unsafe {
                    taa.record_commands(
                        ctx.command_buffer,
                        raw_hdr_view,
                        ctx.depth_view,
                        ctx.motion_view,
                        &push,
                    )?;
                }
                taa.output_view()
            } else {
                raw_hdr_view
            }
        } else {
            raw_hdr_view
        };

        // ── 3. Tonemapping / Swapchain Phase ─────────────────────────────
        self.update_descriptor_set(
            ctx.image_index,
            resolved_view,
            ctx.bloom_view,
            hdr.sampler(),
        );

        let swapchain_extent = vk::Extent2D {
            width: ctx.swapchain.extent.width,
            height: ctx.swapchain.extent.height,
        };

        let target_image = ctx.swapchain.images[ctx.image_index];
        let target_view = ctx.swapchain.image_views[ctx.image_index];

        self.render(
            ctx.command_buffer,
            ctx.image_index,
            swapchain_extent,
            target_image,
            target_view,
            ctx.bloom_intensity,
        )?;

        Ok(())
    }

    pub fn render(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        extent: vk::Extent2D,
        target_image: vk::Image,
        target_view: vk::ImageView,
        _bloom_intensity: f32,
    ) -> Result<()> {
        if self.pipeline.is_none() || self.descriptor_sets.is_empty() {
            return Ok(());
        }

        let pipeline_wrapper = self.pipeline.as_ref().unwrap();
        let pipeline = pipeline_wrapper.pipeline;

        if image_index >= self.descriptor_sets.len() {
            return Ok(());
        }
        let descriptor_set = self.descriptor_sets[image_index % self.descriptor_sets.len()];

        unsafe {
            // 1. Pre-Render Barrier: Transition Swapchain Image to COLOR_ATTACHMENT_OPTIMAL (Sync2)
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_barriers = [barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
            self.device.cmd_pipeline_barrier2(command_buffer, &dep_info);

            // 2. Begin Rendering
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(target_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            self.device
                .cmd_begin_rendering(command_buffer, &rendering_info);

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.fullscreen_pass.pipeline_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let push_constants = PostProcessPushConstants {
                exposure: self.config.exposure,
                bloom_intensity: 0.0, // Forced zero to bypass Ghost Bloom (Phase 2)
                tonemapper_type: 0,   // Linear (Bypass AgX desaturation)
                gamma: self.config.gamma,
            };

            self.device.cmd_push_constants(
                command_buffer,
                self.fullscreen_pass.pipeline_layout(),
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            self.device.cmd_end_rendering(command_buffer);

            // 3. Post-Render Barrier: Transition Swapchain Image to PRESENT_SRC_KHR (Sync2)
            let final_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::empty())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_barriers = [final_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
            self.device.cmd_pipeline_barrier2(command_buffer, &dep_info);
        }

        Ok(())
    }
}

impl Drop for PostProcessSystem {
    fn drop(&mut self) {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            // Pipeline is dropped automatically by RAII.
            // taa_pass: TaaPass does not impl Drop (requires allocator).
            // The Renderer must call taa_pass.destroy_resources(&allocator) before drop.
        }
    }
}
