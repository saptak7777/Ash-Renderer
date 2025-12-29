use ash::vk;
use log::info;

use std::sync::Arc;

use crate::renderer::resource_registry::ResourceRegistry;
use crate::{AshError, Result};

use super::descriptor_allocator::DescriptorAllocator;
use super::descriptor_layout::DescriptorSetLayoutBuilder;
use super::descriptor_set::DescriptorSet;

const EXTRA_TEXTURE_SETS: u32 = 2048;

/// Manages descriptor layouts and descriptor sets for frame, material, and texture resources.
pub struct DescriptorManager {
    allocator: DescriptorAllocator,
    frame_layout: super::descriptor_layout::DescriptorSetLayout,
    material_layout: super::descriptor_layout::DescriptorSetLayout,
    shadow_layout: super::descriptor_layout::DescriptorSetLayout,
    joint_layout: super::descriptor_layout::DescriptorSetLayout,
    frame_sets: Vec<DescriptorSet>,
    m_sets: Vec<DescriptorSet>,
    shadow_sets: Vec<DescriptorSet>,
    joint_sets: Vec<DescriptorSet>,
}

impl DescriptorManager {
    pub fn new(
        device: Arc<ash::Device>,
        frame_count: u32,
        material_worker_count: u32,
        resource_registry: Option<Arc<ResourceRegistry>>,
    ) -> Result<Self> {
        info!("Creating descriptor manager for {frame_count} frames");

        let mut allocator =
            DescriptorAllocator::new(Arc::clone(&device), EXTRA_TEXTURE_SETS, resource_registry)?;

        let frame_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .build(Arc::clone(&device))?;

        let material_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .build(Arc::clone(&device))?;

        // Shadow map layout (set 3, binding 0 - depth texture sampler)
        let shadow_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .build(Arc::clone(&device))?;

        let frame_sets = Self::create_descriptor_sets(frame_count, &frame_layout, &mut allocator)?;
        let m_sets =
            Self::create_descriptor_sets(material_worker_count, &material_layout, &mut allocator)?;
        let shadow_sets =
            Self::create_descriptor_sets(frame_count, &shadow_layout, &mut allocator)?;

        // Joint matrices layout (Storage buffer)
        let joint_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
                1,
            )
            .build(Arc::clone(&device))?;

        let joint_sets = Self::create_descriptor_sets(frame_count, &joint_layout, &mut allocator)?;

        info!(
            "Allocated descriptor sets (frame: {}, material: {})",
            frame_sets.len(),
            m_sets.len()
        );

        Ok(Self {
            allocator,
            frame_layout,
            material_layout,
            shadow_layout,
            joint_layout,
            frame_sets,
            m_sets,
            shadow_sets,
            joint_sets,
        })
    }

    pub fn next_frame(&mut self) {
        self.allocator.next_frame();
    }

    pub fn bind_frame_uniform(
        &self,
        frame_index: usize,
        buffer: vk::Buffer,
        buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        let descriptor = self.frame_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Frame descriptor set index out of bounds".into())
        })?;

        descriptor.update_buffer(
            0,
            buffer,
            0,
            buffer_size,
            vk::DescriptorType::UNIFORM_BUFFER,
        )
    }

    pub fn bind_material_uniform(
        &self,
        worker_id: u32,
        buffer: vk::Buffer,
        buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        let descriptor = self.m_sets.get(worker_id as usize).ok_or_else(|| {
            AshError::VulkanError("Material descriptor set index out of bounds".into())
        })?;

        descriptor.update_buffer(
            0,
            buffer,
            0,
            buffer_size,
            vk::DescriptorType::UNIFORM_BUFFER,
        )
    }

    // Methods bind_material_textures, default_texture_set, default_texture_array_set,
    // allocate_texture_set, allocate_material_texture_set, material_texture_layout removed.

    pub fn frame_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.frame_sets.get(index).map(|set| set.handle())
    }

    pub fn material_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.m_sets.get(index).map(|set| set.handle())
    }

    pub fn frame_set_count(&self) -> usize {
        self.frame_sets.len()
    }

    pub fn material_set_count(&self) -> usize {
        self.m_sets.len()
    }

    /// Get mutable access to the allocator for external allocation (e.g., bindless)
    pub fn allocator_mut(&mut self) -> &mut DescriptorAllocator {
        &mut self.allocator
    }

    pub fn shadow_layout(&self) -> vk::DescriptorSetLayout {
        self.shadow_layout.handle()
    }

    pub fn shadow_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.shadow_sets.get(index).map(|set| set.handle())
    }

    /// Bind shadow map texture to shadow descriptor set for given frame
    pub fn bind_shadow_map(
        &self,
        frame_index: usize,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        let descriptor = self.shadow_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Shadow descriptor set index out of bounds".into())
        })?;

        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };
        descriptor.update_image_at(0, 0, info, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)?;
        Ok(())
    }

    pub fn recreate_frame_sets(&mut self, frame_count: u32) -> Result<()> {
        self.frame_sets =
            Self::create_descriptor_sets(frame_count, &self.frame_layout, &mut self.allocator)?;
        Ok(())
    }

    pub fn recreate_shadow_sets(&mut self, frame_count: u32) -> Result<()> {
        self.shadow_sets =
            Self::create_descriptor_sets(frame_count, &self.shadow_layout, &mut self.allocator)?;
        Ok(())
    }

    pub fn recreate_joint_sets(&mut self, frame_count: u32) -> Result<()> {
        self.joint_sets =
            Self::create_descriptor_sets(frame_count, &self.joint_layout, &mut self.allocator)?;
        Ok(())
    }

    // material_texture_descriptor method removed.

    pub fn frame_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_layout.handle()
    }

    pub fn material_layout(&self) -> vk::DescriptorSetLayout {
        self.material_layout.handle()
    }

    pub fn joint_layout(&self) -> vk::DescriptorSetLayout {
        self.joint_layout.handle()
    }

    pub fn joint_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.joint_sets.get(index).map(|set| set.handle())
    }

    /// Get the descriptor set for joint matrices for a specific frame
    pub fn get_joint_descriptor_set(&self, frame_index: usize) -> Result<vk::DescriptorSet> {
        let descriptor_set = self
            .joint_sets
            .get(frame_index)
            .map(|set| set.handle())
            .ok_or_else(|| {
                AshError::VulkanError(format!(
                    "Joint descriptor set index {frame_index} exceeds buffer count {}",
                    self.joint_sets.len()
                ))
            })?;

        if descriptor_set == vk::DescriptorSet::null() {
            return Err(AshError::VulkanError(format!(
                "Joint descriptor set is null for frame {frame_index}"
            )));
        }

        Ok(descriptor_set)
    }

    /// Bind joint matrices buffer to joint descriptor set for given frame
    pub fn bind_joint_buffer(
        &self,
        frame_index: usize,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let descriptor = self.joint_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Joint descriptor set index out of bounds".into())
        })?;

        descriptor.update_buffer(0, buffer, 0, size, vk::DescriptorType::STORAGE_BUFFER)
    }

    fn create_descriptor_sets(
        count: u32,
        layout: &super::descriptor_layout::DescriptorSetLayout,
        allocator: &mut DescriptorAllocator,
    ) -> Result<Vec<DescriptorSet>> {
        let mut sets = Vec::with_capacity(count as usize);
        for _ in 0..count {
            // Use static pool - these sets persist for renderer lifetime
            sets.push(allocator.allocate_static_set(&layout.handle(), layout.bindings())?);
        }
        Ok(sets)
    }
}
