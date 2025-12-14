use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

use super::descriptor_allocator::DescriptorAllocator;
use super::descriptor_layout::{DescriptorSetLayout, DescriptorSetLayoutBuilder};
use super::descriptor_set::DescriptorSet;

/// Manages bindless descriptor resources (images/buffers) with variable descriptor counts.
pub struct BindlessManager {
    layout: DescriptorSetLayout,
    descriptor_set: DescriptorSet,
    max_resources: u32,
    next_index: u32,
}

impl BindlessManager {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: &mut DescriptorAllocator,
        max_resources: u32,
    ) -> Result<Self> {
        let layout = DescriptorSetLayoutBuilder::new()
            .add_bindless_binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_resources,
            )
            .add_bindless_binding(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_resources,
            )
            .add_bindless_binding(
                2,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_resources,
            )
            .build(Arc::clone(&device))?;

        // Bindless descriptors must be allocated from a pool with UPDATE_AFTER_BIND bit
        let descriptor_set =
            allocator.allocate_bindless_set(layout.handle(), layout.bindings(), max_resources)?;

        Ok(Self {
            layout,
            descriptor_set,
            max_resources,
            next_index: 0,
        })
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set.handle()
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout.handle()
    }

    pub fn add_sampled_image(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<u32> {
        let index = self.allocate_index()?;
        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        self.descriptor_set.update_image_at(
            0,
            index,
            info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;
        Ok(index)
    }

    pub fn add_storage_image(&mut self, image_view: vk::ImageView) -> Result<u32> {
        let index = self.allocate_index()?;
        let info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        self.descriptor_set
            .update_image_at(1, index, info, vk::DescriptorType::STORAGE_IMAGE)?;
        Ok(index)
    }

    pub fn add_storage_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        let index = self.allocate_index()?;
        self.descriptor_set.update_buffer_at(
            2,
            index,
            buffer,
            offset,
            range,
            vk::DescriptorType::STORAGE_BUFFER,
        )?;
        Ok(index)
    }

    fn allocate_index(&mut self) -> Result<u32> {
        if self.next_index >= self.max_resources {
            return Err(AshError::VulkanError(
                "Exceeded maximum number of bindless resources".into(),
            ));
        }
        let index = self.next_index;
        self.next_index += 1;
        Ok(index)
    }
}
