//! Compute pipeline abstraction for GPU compute shaders.

use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Compute pipeline for dispatching compute shaders.
pub struct ComputePipeline {
    device: Arc<ash::Device>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    owns_layout: bool,
}

impl ComputePipeline {
    /// Create a compute pipeline with an existing layout.
    ///
    /// # Safety
    /// Shader module and layout must be valid.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        layout: vk::PipelineLayout,
        shader_module: vk::ShaderModule,
        entry_point: &std::ffi::CStr,
    ) -> Result<Self> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let pipelines = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_, e)| {
                AshError::VulkanError(format!("Failed to create compute pipeline: {e}"))
            })?;

        let pipeline = pipelines[0];

        log::info!("Created compute pipeline");

        Ok(Self {
            device,
            pipeline,
            layout,
            owns_layout: false,
        })
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn builder(device: Arc<ash::Device>) -> ComputePipelineBuilder {
        ComputePipelineBuilder::new(device)
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            if self.owns_layout {
                self.device.destroy_pipeline_layout(self.layout, None);
            }
        }
        log::debug!("ComputePipeline destroyed");
    }
}

/// Builder for compute pipelines.
pub struct ComputePipelineBuilder {
    device: Arc<ash::Device>,
    layout: Option<vk::PipelineLayout>,
    shader_module: Option<vk::ShaderModule>,
    entry_point: String,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl ComputePipelineBuilder {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            layout: None,
            shader_module: None,
            entry_point: "main".to_string(),
            set_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
        }
    }

    pub fn with_layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_shader(mut self, module: vk::ShaderModule) -> Self {
        self.shader_module = Some(module);
        self
    }

    pub fn with_entry_point(mut self, entry: &str) -> Self {
        self.entry_point = entry.to_string();
        self
    }

    pub fn add_set_layout(mut self, layout: vk::DescriptorSetLayout) -> Self {
        self.set_layouts.push(layout);
        self
    }

    pub fn add_push_constant(mut self, range: vk::PushConstantRange) -> Self {
        self.push_constant_ranges.push(range);
        self
    }

    /// Build the compute pipeline.
    ///
    /// # Safety
    /// Shader module must be valid.
    pub unsafe fn build(self) -> Result<ComputePipeline> {
        let shader_module = self.shader_module.ok_or_else(|| {
            AshError::VulkanError("Compute pipeline requires a shader module".into())
        })?;

        let entry_point = std::ffi::CString::new(self.entry_point.as_str())
            .map_err(|e| AshError::VulkanError(format!("Invalid entry point: {e}")))?;

        let (layout, owns_layout) = if let Some(layout) = self.layout {
            (layout, false)
        } else {
            // Create layout from set layouts and push constants
            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&self.set_layouts)
                .push_constant_ranges(&self.push_constant_ranges);

            let layout = self
                .device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create pipeline layout: {e}"))
                })?;
            (layout, true)
        };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_, e)| {
                AshError::VulkanError(format!("Failed to create compute pipeline: {e}"))
            })?;

        let pipeline = pipelines[0];

        log::info!("Created compute pipeline via builder");

        Ok(ComputePipeline {
            device: self.device,
            pipeline,
            layout,
            owns_layout,
        })
    }
}
