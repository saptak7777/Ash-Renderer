use ash::vk;
use std::sync::Arc;

use crate::Result;

/// Lightweight wrapper around a Vulkan descriptor set with helper update methods.
pub struct DescriptorSet {
    device: Arc<ash::Device>,
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
}

impl DescriptorSet {
    pub fn new(
        device: Arc<ash::Device>,
        set: vk::DescriptorSet,
        layout: vk::DescriptorSetLayout,
        bindings: &[vk::DescriptorSetLayoutBinding<'static>],
    ) -> Result<Self> {
        Ok(Self {
            device,
            set,
            layout,
            bindings: bindings.to_vec(),
        })
    }

    pub fn handle(&self) -> vk::DescriptorSet {
        self.set
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn bindings(&self) -> &[vk::DescriptorSetLayoutBinding<'static>] {
        &self.bindings
    }

    pub fn update_buffer(
        &self,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
        descriptor_type: vk::DescriptorType,
    ) -> Result<()> {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            self.device.update_descriptor_sets(&[write], &[]);
        }

        Ok(())
    }

    pub fn update_buffer_at(
        &self,
        binding: u32,
        array_index: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
        descriptor_type: vk::DescriptorType,
    ) -> Result<()> {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .dst_array_element(array_index)
            .descriptor_type(descriptor_type)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            self.device.update_descriptor_sets(&[write], &[]);
        }

        Ok(())
    }

    pub fn update_image(
        &self,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) -> Result<()> {
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .descriptor_type(descriptor_type)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            self.device.update_descriptor_sets(&[write], &[]);
        }

        Ok(())
    }

    pub fn update_image_at(
        &self,
        binding: u32,
        array_index: u32,
        info: vk::DescriptorImageInfo,
        descriptor_type: vk::DescriptorType,
    ) -> Result<()> {
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .dst_array_element(array_index)
            .descriptor_type(descriptor_type)
            .image_info(std::slice::from_ref(&info));

        unsafe {
            self.device.update_descriptor_sets(&[write], &[]);
        }

        Ok(())
    }
}
