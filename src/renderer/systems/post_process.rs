use crate::renderer::passes::fullscreen::FullscreenPass;
use crate::renderer::passes::fullscreen::PostProcessPushConstants;
use crate::vulkan;
use crate::Result;
use ash::vk;
use bytemuck;
use std::sync::Arc;

use crate::renderer::frame::Frame;
use crate::renderer::passes::temporal_aa::ConfigMetrics;
use crate::renderer::passes::vsr::{SharpenConfig, VsrConfig, VsrPass};
use crate::renderer::resources::HdrSystem;
use crate::renderer::resources::Resources;
use crate::vulkan::SwapchainWrapper;

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
            gamma: 2.2,
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
                log::error!("Failed to create post-process pipeline: {}", e);
                return Err(e);
            }
        };
        self.pipeline = Some(pipeline);

        if self.descriptor_sets.len() != image_count {
            self.reallocate_descriptors(image_count)?;
        }

        Ok(())
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
    pub vsr: Option<&'a mut VsrPass>,
    pub hdr: Option<&'a HdrSystem>,
    pub taa_metrics: Option<&'a mut ConfigMetrics>,
    pub vsr_config: VsrConfig,
    pub sharpen_config: Option<SharpenConfig>,
    pub jitter_uv: [f32; 2],
}

impl PostProcessSystem {
    /// Execute the complete post-processing chain.
    ///
    /// This method orchestrates:
    /// 1. VSR (upscaling) dispatch if enabled.
    /// 2. Tone Mapping (HDR to SDR) with swapchain barriers.
    pub fn execute(&mut self, mut ctx: PostProcessContext) -> Result<()> {
        // 1. VSR / Upscaling Phase
        if let Some(ref mut vsr) = ctx.vsr {
            let upscale_config = crate::renderer::passes::vsr::VsrUpscaleConfig {
                velocity_threshold: 0.05, // Standard threshold
                history_weight: ctx.vsr_config.history_weight(),
                clamping_gamma: ctx.vsr_config.clamping_gamma(),
                anti_ghosting: ctx.vsr_config.anti_ghosting,
            };

            let vsr_inputs = crate::renderer::passes::vsr::VsrInputs {
                color_index: ctx.frame.hdr_image_index.ok_or_else(|| {
                    crate::AshError::VulkanError(
                        "HDR image index not initialized for VSR".to_string(),
                    )
                })?,
                depth_index: {
                    let gbuffer_indices = ctx.frame.gbuffer_indices.as_ref().unwrap();
                    if gbuffer_indices.depth_index == u32::MAX {
                        0 // Fallback to default white texture index
                    } else {
                        gbuffer_indices.depth_index
                    }
                },
                motion_index: {
                    let gbuffer_indices = ctx.frame.gbuffer_indices.as_ref().unwrap();
                    if gbuffer_indices.motion_index == u32::MAX {
                        0 // Fallback to default white texture index
                    } else {
                        gbuffer_indices.motion_index
                    }
                },
                jitter: ctx.jitter_uv,
            };

            unsafe {
                vsr.upscale_with_sharpening(
                    ctx.command_buffer,
                    vsr_inputs,
                    &upscale_config,
                    ctx.sharpen_config.as_ref(),
                )
                .map_err(|e| crate::AshError::VulkanError(format!("VSR upscale failed: {e}")))?;
            }
            vsr.next_frame();
        }

        // 2. Tonemapping / Swapchain Resolution Phase
        if let Some(hdr) = ctx.hdr {
            // Resolve input view (VSR output vs raw HDR output)
            let input_view = ctx
                .vsr
                .as_ref()
                .map(|vsr| vsr.active_view())
                .unwrap_or_else(|| hdr.view());

            let black_view = ctx.resources.black_texture.view();

            self.update_descriptor_set(
                ctx.image_index,
                input_view,
                black_view, // bloom_view placeholder
                black_view, // ssgi_view placeholder
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
    ) -> Result<()> {
        if self.pipeline.is_none() || self.descriptor_sets.is_empty() {
            return Ok(());
        }

        let pipeline_wrapper = self.pipeline.as_ref().unwrap();
        let pipeline = pipeline_wrapper.pipeline; // helper from wrapper

        // Handle case where image_index is out of bounds
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
                gamma: self.config.gamma,
                bloom_intensity: if self.config.bloom_enabled {
                    self.config.bloom_intensity
                } else {
                    0.0
                },
                tonemapping_enabled: if self.config.tonemapping_enabled {
                    1.0
                } else {
                    0.0
                },
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
            // Pipeline is dropped automatically by RAII
        }
    }
}
