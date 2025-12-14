use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

use super::descriptor_allocator::DescriptorAllocator;
use super::descriptor_layout::DescriptorSetLayoutBuilder;
use super::descriptor_set::DescriptorSet;

/// Manages bindless descriptor resources (images/buffers) with variable descriptor counts.
pub struct BindlessManager {
    descriptor_set: DescriptorSet,
    max_resources: u32,
    next_index: u32,
    // Keep default texture alive
    default_texture: Option<crate::renderer::Texture>,
}

impl BindlessManager {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: &mut DescriptorAllocator,
        max_resources: u32,
        vma_allocator: Arc<crate::vulkan::Allocator>,
        queue_family_index: u32,
        graphics_queue: vk::Queue,
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

        // Bindless descriptors are long-lived and should be allocated from the dedicated bindless pool
        let descriptor_set =
            allocator.allocate_bindless_set(layout.handle(), layout.bindings(), max_resources)?;

        let mut manager = Self {
            descriptor_set,
            max_resources,
            next_index: 0,
            default_texture: None,
        };

        // Create temporary command pool for transfer
        // Create temporary command pool for transfer
        let pool_create_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            ..Default::default()
        };
        let command_pool =
            unsafe { device.create_command_pool(&pool_create_info, None) }.map_err(|e| {
                AshError::VulkanError(format!("Failed to create bindless transfer pool: {e}"))
            })?;

        // Create and register default white texture at index 0
        let white_pixels =
            crate::renderer::resources::texture::TextureData::solid_color([255, 255, 255, 255]);

        // SAFETY: Only locally created handles used, and we own the device references
        let default_texture = unsafe {
            crate::renderer::Texture::from_data(
                vma_allocator,
                Arc::clone(&device),
                command_pool,
                graphics_queue,
                &white_pixels,
                vk::Format::R8G8B8A8_UNORM,
                Some("Default Bindless Texture"),
            )?
        };

        unsafe { device.destroy_command_pool(command_pool, None) };

        // Manually add it to index 0
        let index = manager.allocate_index()?; // Should be 0
        if index != 0 {
            return Err(AshError::VulkanError(
                "Bindless index 0 reserved for default texture but not returned".into(),
            ));
        }

        let info = vk::DescriptorImageInfo {
            sampler: default_texture.sampler(),
            image_view: default_texture.view(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        manager.descriptor_set.update_image_at(
            0,
            index,
            info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        manager.default_texture = Some(default_texture);

        Ok(manager)
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set.handle()
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set.layout()
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
