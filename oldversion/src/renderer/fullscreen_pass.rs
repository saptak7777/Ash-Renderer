//! Fullscreen Quad for Post-Processing
//!
//! Provides infrastructure for fullscreen post-processing passes.

use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Fullscreen pass for post-processing effects
///
/// Uses a single triangle that covers the entire screen (more efficient than a quad).
/// No vertex buffer needed - vertices are generated in the shader.
pub struct FullscreenPass {
    device: Arc<ash::Device>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl FullscreenPass {
    /// Creates a new fullscreen pass
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this pass.
    pub unsafe fn new(device: Arc<ash::Device>, output_format: vk::Format) -> Result<Self> {
        log::info!("Creating fullscreen pass");

        // Create render pass for fullscreen output
        let color_attachment = vk::AttachmentDescription {
            format: output_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
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
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| AshError::VulkanError(format!("Fullscreen render pass failed: {e}")))?;

        // Create descriptor set layout for input texture
        // Create descriptor set layout for input textures
        let bindings = [
            // Binding 0: HDR input
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // Binding 1: Bloom input
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| AshError::VulkanError(format!("Descriptor layout failed: {e}")))?;

        // Create push constant range for post-process parameters
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<PostProcessPushConstants>() as u32,
        };

        // Create pipeline layout
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| AshError::VulkanError(format!("Pipeline layout failed: {e}")))?;

        log::info!("Fullscreen pass created successfully");

        Ok(Self {
            device,
            render_pass,
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    /// Returns the render pass
    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    /// Returns the pipeline layout
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Returns the descriptor set layout
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }
}

impl Drop for FullscreenPass {
    fn drop(&mut self) {
        unsafe {
            log::debug!("Destroying fullscreen pass");
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

/// Push constants for post-processing shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PostProcessPushConstants {
    /// Exposure multiplier for tonemapping
    pub exposure: f32,
    /// Gamma correction value
    pub gamma: f32,
    /// Bloom intensity
    pub bloom_intensity: f32,
    /// Reserved for future use
    pub _padding: f32,
}
