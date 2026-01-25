use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

use super::descriptor_allocator::DescriptorAllocator;
use super::descriptor_layout::{DescriptorSetLayout, DescriptorSetLayoutBuilder};
use super::descriptor_set::DescriptorSet;

#[derive(Clone)]
struct RegisteredImage {
    index: u32,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

#[derive(Clone)]
struct RegisteredBuffer {
    index: u32,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    range: vk::DeviceSize,
}

/// Manages bindless descriptor resources (images) with variable descriptor counts.
pub struct BindlessManager {
    #[allow(dead_code)]
    device: Arc<ash::Device>,
    layout: DescriptorSetLayout,
    descriptor_set: DescriptorSet,
    max_resources: u32,
    next_image_index: u32,
    next_buffer_index: u32,
    // Resource tracking for recreation
    registered_images: Vec<RegisteredImage>,
    registered_buffers: Vec<RegisteredBuffer>,
}

impl BindlessManager {
    pub const DEFAULT_MAX_TEXTURES: u32 = 65536; // Increased for modern hardware

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
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_resources,
            )
            .build(Arc::clone(&device))?;

        // Bindless descriptors must be allocated from a pool with UPDATE_AFTER_BIND bit
        let descriptor_set =
            allocator.allocate_bindless_set(layout.handle(), layout.bindings(), max_resources)?;

        Ok(Self {
            device,
            layout,
            descriptor_set,
            max_resources,
            next_image_index: 0,
            next_buffer_index: 0,
            registered_images: Vec::new(),
            registered_buffers: Vec::new(),
        })
    }

    /// Recreate the descriptor set (e.g., after swapchain resize)
    pub fn recreate(&mut self, allocator: &mut DescriptorAllocator) -> Result<()> {
        log::info!("Recreating bindless descriptor set...");

        // Allocate new descriptor set with same layout
        let new_descriptor_set = allocator.allocate_bindless_set(
            self.layout.handle(),
            self.layout.bindings(),
            self.max_resources,
        )?;

        // Free the old descriptor set to prevent memory leaks
        allocator.free_bindless_set(self.descriptor_set.handle())?;

        // Replace old descriptor set
        self.descriptor_set = new_descriptor_set;

        // Re-register all previously registered resources
        self.re_register_all()?;

        log::info!("Bindless descriptor set recreated successfully");
        Ok(())
    }

    /// Re-register all previously registered resources
    fn re_register_all(&mut self) -> Result<()> {
        // Re-register images
        for img in &self.registered_images {
            let info = vk::DescriptorImageInfo {
                sampler: img.sampler,
                image_view: img.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            self.descriptor_set.update_image_at(
                0,
                img.index,
                info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            )?;
        }

        // Re-register buffers
        for buf in &self.registered_buffers {
            self.descriptor_set.update_buffer_at(
                1,
                buf.index,
                buf.buffer,
                buf.offset,
                buf.range,
                vk::DescriptorType::STORAGE_BUFFER,
            )?;
        }

        log::debug!(
            "Re-registered {} images and {} buffers",
            self.registered_images.len(),
            self.registered_buffers.len()
        );

        Ok(())
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

        // Track for recreation
        self.registered_images.push(RegisteredImage {
            index,
            view: image_view,
            sampler,
        });

        Ok(index)
    }

    pub fn add_storage_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        let index = self.allocate_buffer_index()?;
        self.descriptor_set.update_buffer_at(
            1,
            index,
            buffer,
            offset,
            range,
            vk::DescriptorType::STORAGE_BUFFER,
        )?;

        // Track for recreation
        self.registered_buffers.push(RegisteredBuffer {
            index,
            buffer,
            offset,
            range,
        });

        Ok(index)
    }

    /// Get the descriptor set layout
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.layout.handle()
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

    fn allocate_buffer_index(&mut self) -> Result<u32> {
        if self.next_buffer_index >= self.max_resources {
            return Err(AshError::VulkanError(format!(
                "Exceeded maximum number of bindless buffers: {}/{}",
                self.next_buffer_index, self.max_resources
            )));
        }
        let index = self.next_buffer_index;
        self.next_buffer_index += 1;
        Ok(index)
    }
}
