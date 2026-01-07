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
    next_image_index: u32,
    next_material_index: u32,
    next_instance_index: u32,
    next_indirect_index: u32,
}

impl BindlessManager {
    pub const DEFAULT_MAX_TEXTURES: u32 = 16384;
    pub const DEFAULT_MAX_BUFFERS: u32 = 1024;

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
                vk::DescriptorType::STORAGE_BUFFER, // Materials
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_resources,
            )
            .add_bindless_binding(
                2,
                vk::DescriptorType::STORAGE_BUFFER, // Instance Data
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_resources,
            )
            .add_bindless_binding(
                3,
                vk::DescriptorType::STORAGE_BUFFER, // Indirect/Joints
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
            next_image_index: 0,
            next_material_index: 0,
            next_instance_index: 0,
            next_indirect_index: 0,
        })
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout.handle()
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set.handle()
    }

    pub fn add_sampled_image(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<u32> {
        let index = self.allocate_image_index()?;
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

    pub fn add_material_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        let index = self.allocate_material_index()?;
        self.descriptor_set.update_buffer_at(
            1,
            index,
            buffer,
            offset,
            range,
            vk::DescriptorType::STORAGE_BUFFER,
        )?;
        Ok(index)
    }

    pub fn add_instance_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        let index = self.allocate_instance_index()?;
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

    pub fn add_indirect_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        let index = self.allocate_indirect_index()?;
        self.descriptor_set.update_buffer_at(
            3,
            index,
            buffer,
            offset,
            range,
            vk::DescriptorType::STORAGE_BUFFER,
        )?;
        Ok(index)
    }

    /// Backwards compatibility method - maps to texture binding
    pub fn add_storage_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        self.add_instance_buffer(buffer, offset, range)
    }

    pub fn validate_index(&self, index: u32) -> Result<()> {
        if index >= self.max_resources {
            return Err(AshError::VulkanError(format!(
                "Invalid bindless index: {} (max: {})",
                index, self.max_resources
            )));
        }
        Ok(())
    }

    fn allocate_image_index(&mut self) -> Result<u32> {
        if self.next_image_index >= self.max_resources {
            return Err(AshError::VulkanError(format!(
                "Exceeded maximum number of bindless images: {}/{}",
                self.next_image_index, self.max_resources
            )));
        }
        let index = self.next_image_index;
        self.next_image_index += 1;
        Ok(index)
    }

    fn allocate_material_index(&mut self) -> Result<u32> {
        if self.next_material_index >= self.max_resources {
            return Err(AshError::VulkanError(format!(
                "Exceeded maximum number of bindless materials: {}/{}",
                self.next_material_index, self.max_resources
            )));
        }
        let index = self.next_material_index;
        self.next_material_index += 1;
        Ok(index)
    }

    fn allocate_instance_index(&mut self) -> Result<u32> {
        if self.next_instance_index >= self.max_resources {
            return Err(AshError::VulkanError(format!(
                "Exceeded maximum number of bindless instances: {}/{}",
                self.next_instance_index, self.max_resources
            )));
        }
        let index = self.next_instance_index;
        self.next_instance_index += 1;
        Ok(index)
    }

    fn allocate_indirect_index(&mut self) -> Result<u32> {
        if self.next_indirect_index >= self.max_resources {
            return Err(AshError::VulkanError(format!(
                "Exceeded maximum number of bindless indirect commands: {}/{}",
                self.next_indirect_index, self.max_resources
            )));
        }
        let index = self.next_indirect_index;
        self.next_indirect_index += 1;
        Ok(index)
    }
}
