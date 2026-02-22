use crate::Result;
use crate::renderer::passes::fullscreen::FullscreenPass;
use crate::renderer::passes::fullscreen::PostProcessPushConstants;
use crate::renderer::passes::temporal_aa::{ConfigMetrics, TaaPass, TaaPushConstants};
use crate::renderer::resources::HdrSystem;
use crate::renderer::resources::Resources;
use crate::vulkan;
use crate::vulkan::SwapchainWrapper;
use ash::vk;
use bytemuck;
use std::sync::Arc;

use crate::renderer::frame::Frame;

/// Configuration for post-processing effects.
#[derive(Clone, Copy, Debug)]
pub struct PostProcessConfig {
    pub tonemapping_enabled: bool,
    pub exposure: f32,
    pub gamma: f32,
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            tonemapping_enabled: true,
            exposure: 1.2,
            gamma: 1.0,
            bloom_enabled: true,
            bloom_intensity: 0.1,
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
        ssgi_view: vk::ImageView,
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

        let ssgi_info = vk::DescriptorImageInfo::default()
            .image_view(ssgi_view)
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
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&ssgi_info)),
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

    /// Enable or disable tonemapping.
    pub fn set_tonemapping_enabled(&mut self, enabled: bool) {
        self.config.tonemapping_enabled = enabled;
    }

    /// Query tonemapping enabled state.
    pub fn tonemapping_enabled(&self) -> bool {
        self.config.tonemapping_enabled
    }

    /// Set the exposure value (clamped to `>= 0.0`).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.config.exposure = exposure.max(0.0);
    }

    /// Query the current exposure value.
    pub fn exposure(&self) -> f32 {
        self.config.exposure
    }

    /// Set the gamma value (clamped to `>= 0.1` to avoid gamma = 0).
    pub fn set_gamma(&mut self, gamma: f32) {
        self.config.gamma = gamma.max(0.1);
    }

    /// Query the current gamma value.
    pub fn gamma(&self) -> f32 {
        self.config.gamma
    }

    /// Enable or disable bloom.
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.config.bloom_enabled = enabled;
    }

    /// Query bloom enabled state.
    pub fn bloom_enabled(&self) -> bool {
        self.config.bloom_enabled
    }

    /// Set the bloom intensity (clamped to `[0.0, 2.0]`).
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        self.config.bloom_intensity = intensity.clamp(0.0, 2.0);
    }

    /// Query the current bloom intensity.
    pub fn bloom_intensity(&self) -> f32 {
        self.config.bloom_intensity
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
    pub jitter_uv: [f32; 2],
    pub prev_jitter_uv: [f32; 2],
    /// Direct depth image view (SHADER_READ_ONLY_OPTIMAL) for TAA
    pub depth_view: vk::ImageView,
    /// Direct motion vector image view (SHADER_READ_ONLY_OPTIMAL) for TAA
    pub motion_view: vk::ImageView,
}

impl PostProcessSystem {
    /// Record the complete post-processing chain commands.
    ///
    /// Pipeline: Geometry → TAA Resolve → Tone Mapping → Swapchain
    ///
    /// VSR is intentionally bypassed while we validate native TAA stability.
    pub fn record_commands(&mut self, ctx: PostProcessContext) -> Result<()> {
        if let Some(hdr) = ctx.hdr {
            let raw_hdr_view = hdr.view();
            let black_view = ctx.resources.black_texture.view();

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
                        jitter_x: ctx.jitter_uv[0],
                        jitter_y: ctx.jitter_uv[1],
                        prev_jitter_x: ctx.prev_jitter_uv[0],
                        prev_jitter_y: ctx.prev_jitter_uv[1],
                        blend_factor: ctx.taa_config.blend_factor,
                        clamping_gamma: ctx.taa_config.quality.clamping_gamma(),
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
                black_view, // bloom placeholder
                black_view, // ssgi placeholder
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
                u32::from(ctx.swapchain.is_hdr),
            )?;
        }

        Ok(())
    }

    pub fn render(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        extent: vk::Extent2D,
        target_image: vk::Image,
        target_view: vk::ImageView,
        is_hdr: u32,
    ) -> Result<()> {
        if self.pipeline.is_none() || self.descriptor_sets.is_empty() {
            return Ok(());
        }

        let pipeline_wrapper = self.pipeline.as_ref().unwrap();
        let pipeline = pipeline_wrapper.pipeline;

        if image_index >= self.descriptor_sets.len() {
            return Ok(());
        }
        let descriptor_set = self.descriptor_sets[image_index];

        unsafe {
            // 1. Pre-Render Barrier: Transition Swapchain Image to COLOR_ATTACHMENT_OPTIMAL
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

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
                bloom_intensity: if self.config.bloom_enabled {
                    self.config.bloom_intensity
                } else {
                    0.0
                },
                tonemapper_type: if self.config.tonemapping_enabled {
                    1
                } else {
                    0
                },
                is_hdr,
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

            // 3. Post-Render Barrier: Transition Swapchain Image to PRESENT_SRC_KHR
            let final_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[final_barrier],
            );
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
