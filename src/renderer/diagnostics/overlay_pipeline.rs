//! Overlay pipeline for rendering diagnostics text

use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Overlay rendering pipeline
pub struct OverlayPipeline {
    device: Arc<ash::Device>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// Vertex buffer for overlay (recreated each frame)
    vertex_buffer: Option<vk::Buffer>,
    vertex_buffer_memory: Option<vk::DeviceMemory>,
}

impl OverlayPipeline {
    /// Create overlay pipeline
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this pipeline.
    pub unsafe fn new(device: Arc<ash::Device>, swapchain_format: vk::Format) -> Result<Self> {
        log::info!("[OverlayPipeline] Creating overlay pipeline");

        // Create render pass for overlay (no depth, alpha blending)
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::LOAD, // Preserve existing framebuffer
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let attachments = [color_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| AshError::VulkanError(format!("Overlay render pass failed: {e}")))?;

        // Create pipeline layout (no descriptors, no push constants)
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| AshError::VulkanError(format!("Overlay layout failed: {e}")))?;

        log::info!("[OverlayPipeline] Overlay pipeline created");

        Ok(Self {
            device,
            render_pass,
            pipeline_layout,
            pipeline: vk::Pipeline::null(),
            vertex_buffer: None,
            vertex_buffer_memory: None,
        })
    }

    /// Check if pipeline needs to be created
    pub fn needs_pipeline(&self) -> bool {
        self.pipeline == vk::Pipeline::null()
    }

    /// Get render pass handle
    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    /// Get pipeline handle
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Get pipeline layout
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for OverlayPipeline {
    fn drop(&mut self) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            if let Some(buffer) = self.vertex_buffer.take() {
                self.device.destroy_buffer(buffer, None);
            }
            if let Some(memory) = self.vertex_buffer_memory.take() {
                self.device.free_memory(memory, None);
            }

            log::info!("[OverlayPipeline] Overlay pipeline destroyed");
        }
    }
}
