//! VSM Shadow Pass - Renders geometry into physical cache pages

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::resources::{PageAllocation, VsmResources};

/// VSM shadow rendering pass
pub struct VsmShadowPass {
    device: Arc<ash::Device>,

    /// Render pass for depth-only rendering
    pub render_pass: vk::RenderPass,

    /// Framebuffer for physical cache
    pub framebuffer: vk::Framebuffer,

    /// Pipeline for shadow rendering
    shadow_pipeline: Option<vk::Pipeline>,
    shadow_pipeline_layout: Option<vk::PipelineLayout>,
}

impl VsmShadowPass {
    /// Create a new VSM shadow pass
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this pass.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        _allocator: Arc<Allocator>,
        resources: &VsmResources,
    ) -> Result<Self> {
        log::info!("Creating VSM shadow pass");

        // Create render pass (depth-only)
        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::R32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::LOAD, // Load existing data
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::GENERAL,
            final_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        };

        let depth_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&depth_ref));

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let attachments = [depth_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("VSM render pass creation failed: {e:?}"))
            })?;

        // Create framebuffer
        let attachments = [resources.physical_cache_view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(resources.config().physical_resolution)
            .height(resources.config().physical_resolution)
            .layers(1);

        let framebuffer = device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("VSM framebuffer creation failed: {e:?}"))
            })?;

        log::info!("VSM shadow pass created successfully");

        Ok(Self {
            device,
            render_pass,
            framebuffer,
            shadow_pipeline: None,
            shadow_pipeline_layout: None,
        })
    }

    /// Render shadows for allocated pages
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn render_shadows<F>(
        &self,
        cmd: vk::CommandBuffer,
        allocations: &[PageAllocation],
        page_size: u32,
        mut draw_fn: F,
    ) where
        F: FnMut(vk::CommandBuffer, &PageAllocation),
    {
        if allocations.is_empty() {
            return;
        }

        log::debug!("Rendering {} shadow pages", allocations.len());

        // For each allocated page, set viewport and render
        for allocation in allocations {
            // Calculate viewport for this physical page
            let x = (allocation.physical_x * page_size) as f32;
            let y = (allocation.physical_y * page_size) as f32;

            let viewport = vk::Viewport {
                x,
                y,
                width: page_size as f32,
                height: page_size as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D {
                    width: page_size,
                    height: page_size,
                },
            };

            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Call user draw function for this page
            draw_fn(cmd, allocation);
        }
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying VSM shadow pass");

        if let Some(pipeline) = self.shadow_pipeline.take() {
            self.device.destroy_pipeline(pipeline, None);
        }

        if let Some(layout) = self.shadow_pipeline_layout.take() {
            self.device.destroy_pipeline_layout(layout, None);
        }

        if self.framebuffer != vk::Framebuffer::null() {
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.framebuffer = vk::Framebuffer::null();
        }

        if self.render_pass != vk::RenderPass::null() {
            self.device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }

        log::debug!("VSM shadow pass destroyed");
    }
}

impl Drop for VsmShadowPass {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
