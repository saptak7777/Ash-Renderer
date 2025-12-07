use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// RAII wrapper for a descriptor set layout with optional bindless flags
pub struct DescriptorSetLayout {
    device: Arc<ash::Device>,
    layout: vk::DescriptorSetLayout,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorSetLayout {
    pub fn new(device: Arc<ash::Device>, bindings: &[vk::DescriptorSetLayoutBinding]) -> Result<Self> {
        let binding_flags: Vec<_> = bindings
            .iter()
            .map(|binding| {
                if binding.descriptor_count > 1 {
                    vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                        | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                } else {
                    vk::DescriptorBindingFlags::empty()
                }
            })
            .collect();

        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
            .binding_flags(&binding_flags);

        let mut create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);

        create_info = create_info.push_next(&mut flags_info);

        let layout = unsafe {
            device
                .create_descriptor_set_layout(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!(
                    "Failed to create descriptor set layout: {e}"
                )))?
        };

        Ok(Self {
            device,
            layout,
            bindings: bindings.to_vec(),
        })
    }

    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn bindings(&self) -> &[vk::DescriptorSetLayoutBinding] {
        &self.bindings
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

pub struct DescriptorSetLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl Default for DescriptorSetLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DescriptorSetLayoutBuilder {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
    }

    pub fn add_binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
        count: u32,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(count)
                .stage_flags(stage_flags)
                .build(),
        );
        self
    }

    pub fn add_bindless_binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
        max_count: u32,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(max_count)
                .stage_flags(stage_flags)
                .build(),
        );
        self
    }

    pub fn build(self, device: Arc<ash::Device>) -> Result<DescriptorSetLayout> {
        DescriptorSetLayout::new(device, &self.bindings)
    }
}
