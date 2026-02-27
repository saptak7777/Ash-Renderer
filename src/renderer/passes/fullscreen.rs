//! Fullscreen Quad for Post-Processing
//!
//! Provides infrastructure for fullscreen post-processing passes.

use ash::vk;
use std::sync::Arc;

use crate::vulkan;
use crate::{AshError, Result};

/// Fullscreen pass for post-processing effects
///
/// Uses a single triangle that covers the entire screen (more efficient than a quad).
/// No vertex buffer needed - vertices are generated in the shader.
pub struct FullscreenPass {
    device: Arc<ash::Device>,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    output_format: vk::Format,
}

impl FullscreenPass {
    /// Creates a new fullscreen pass
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this pass.
    pub unsafe fn new(device: Arc<ash::Device>, output_format: vk::Format) -> Result<Self> {
        log::info!("Creating fullscreen pass");

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
            // Binding 2: Unused
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None) }
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

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .map_err(|e| AshError::VulkanError(format!("Pipeline layout failed: {e}")))?;

        log::info!("Fullscreen pass created successfully");

        Ok(Self {
            device,
            pipeline_layout,
            descriptor_set_layout,
            output_format,
        })
    }

    /// Returns the output format
    pub fn output_format(&self) -> vk::Format {
        self.output_format
    }

    /// Returns the pipeline layout
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Returns the descriptor set layout
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn create_pipeline(
        &self,
        device: &Arc<ash::Device>,
        extent: vk::Extent2D,
        entry_point: &str,
    ) -> Result<vulkan::Pipeline> {
        let mut builder = vulkan::Pipeline::builder(Arc::clone(device))
            .with_layout(self.pipeline_layout)
            .with_dynamic_rendering(&[self.output_format], None, None)
            .with_extent(extent)
            .with_cull_mode(vk::CullModeFlags::NONE);

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/postprocess.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            entry_point,
        )?;

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/tonemapping.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            entry_point,
        )?;

        builder.build()
    }
}

impl Drop for FullscreenPass {
    fn drop(&mut self) {
        unsafe {
            log::debug!("Destroying fullscreen pass");
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Push constants for post-processing shaders.
///
/// Layout is `repr(C)` and kept at exactly 16 bytes for Vulkan
/// push-constant alignment rules and `bytemuck::Pod` compliance.
///
/// Byte map: `[exposure(4), bloom_intensity(4), tonemapper_type(4), gamma(4)]` → **16 bytes total**
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessPushConstants {
    /// Exposure multiplier applied before tonemapping.
    pub exposure: f32,
    /// Bloom composite intensity (0.0 = bloom disabled).
    pub bloom_intensity: f32,
    /// Which tonemapper to use.
    pub tonemapper_type: u32,
    /// Target gamma for display calibration (e.g. 2.2).
    pub gamma: f32,
}

impl Default for PostProcessPushConstants {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            bloom_intensity: 0.0,
            tonemapper_type: 1,
            gamma: 2.2,
        }
    }
}
