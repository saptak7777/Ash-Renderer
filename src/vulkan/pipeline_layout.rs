use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// RAII wrapper for Vulkan pipeline layouts with a fluent builder API.
pub struct PipelineLayout {
    layout: vk::PipelineLayout,
    device: Arc<ash::Device>,
    managed_by_registry: bool,
}

impl PipelineLayout {
    pub fn builder(device: Arc<ash::Device>) -> PipelineLayoutBuilder {
        PipelineLayoutBuilder::new(device)
    }

    pub fn handle(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn mark_managed_by_registry(&mut self) {
        self.managed_by_registry = true;
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }

        unsafe {
            if self.layout != vk::PipelineLayout::null() {
                self.device.destroy_pipeline_layout(self.layout, None);
            }
        }
    }
}

/// Builder that mirrors Vulkan's pipeline layout creation while staying safe.
pub struct PipelineLayoutBuilder {
    device: Arc<ash::Device>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    push_constants: Vec<vk::PushConstantRange>,
}

impl PipelineLayoutBuilder {
    fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            set_layouts: Vec::new(),
            push_constants: Vec::new(),
        }
    }

    /// Adds a descriptor set layout to the layout.
    pub fn add_set_layout(mut self, layout: vk::DescriptorSetLayout) -> Self {
        self.set_layouts.push(layout);
        self
    }

    /// Adds a push constant range.
    pub fn add_push_constant(mut self, range: vk::PushConstantRange) -> Self {
        self.push_constants.push(range);
        self
    }

    pub fn build(self) -> Result<PipelineLayout> {
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&self.set_layouts)
            .push_constant_ranges(&self.push_constants);

        let layout = unsafe {
            self.device
                .create_pipeline_layout(&info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create pipeline layout: {e}"))
                })?
        };

        Ok(PipelineLayout {
            layout,
            device: Arc::clone(&self.device),
            managed_by_registry: false,
        })
    }
}
