use crate::vulkan::descriptor_layout::{DescriptorSetLayout, DescriptorSetLayoutBuilder};
use crate::Result;
use ash::vk;
use std::sync::Arc;

pub struct BindlessDescriptorSet {
    pub layout: DescriptorSetLayout,
}

impl BindlessDescriptorSet {
    pub fn new(device: Arc<ash::Device>) -> Result<Self> {
        let layout = DescriptorSetLayoutBuilder::new()
            .add_bindless_binding(
                1, // global_cubemaps
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                1024,
            )
            .build(device)?;

        Ok(Self { layout })
    }
}
