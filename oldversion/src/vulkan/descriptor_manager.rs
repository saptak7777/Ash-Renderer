use ash::vk;
use log::info;
use std::collections::HashMap;
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
    texture_layout: super::descriptor_layout::DescriptorSetLayout,
    material_texture_layout: super::descriptor_layout::DescriptorSetLayout,
    shadow_layout: super::descriptor_layout::DescriptorSetLayout,
    frame_sets: Vec<DescriptorSet>,
    material_sets: Vec<DescriptorSet>,
    shadow_sets: Vec<DescriptorSet>,
    default_texture_set: DescriptorSet,
    default_texture_array_set: DescriptorSet,
    dynamic_sets: HashMap<vk::DescriptorSet, DescriptorSet>,
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

        let texture_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .build(Arc::clone(&device))?;

        let material_texture_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                2,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                3,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                4,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
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
        let material_sets =
            Self::create_descriptor_sets(material_worker_count, &material_layout, &mut allocator)?;
        let shadow_sets =
            Self::create_descriptor_sets(frame_count, &shadow_layout, &mut allocator)?;
        let default_texture_set =
            allocator.allocate_static_set(&texture_layout.handle(), texture_layout.bindings())?;
        let default_texture_array_set = allocator.allocate_static_set(
            &material_texture_layout.handle(),
            material_texture_layout.bindings(),
        )?;

        info!(
            "Allocated descriptor sets (frame: {}, material + texture: 2)",
            frame_sets.len()
        );

        Ok(Self {
            allocator,
            frame_layout,
            material_layout,
            texture_layout,
            material_texture_layout,
            shadow_layout,
            frame_sets,
            material_sets,
            shadow_sets,
            default_texture_set,
            default_texture_array_set,
            dynamic_sets: HashMap::new(),
        })
    }

    pub fn next_frame(&mut self) {
        self.allocator.next_frame();
        // NOTE: We no longer clear dynamic_sets here because texture descriptors
        // are now allocated from the static pool and should persist.
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
        worker_index: usize,
        buffer: vk::Buffer,
        buffer_size: vk::DeviceSize,
    ) -> Result<()> {
        let descriptor = self.material_sets.get(worker_index).ok_or_else(|| {
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

    pub fn bind_material_textures(
        &self,
        set: vk::DescriptorSet,
        bindings: &[(u32, vk::ImageView, vk::Sampler)],
    ) -> Result<()> {
        let descriptor = self.material_texture_descriptor(set)?;
        for &(binding, view, sampler) in bindings {
            let info = vk::DescriptorImageInfo {
                sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            descriptor.update_image_at(
                binding,
                0,
                info,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            )?;
        }
        Ok(())
    }

    pub fn default_texture_set(&self) -> vk::DescriptorSet {
        self.default_texture_set.handle()
    }

    pub fn default_texture_array_set(&self) -> vk::DescriptorSet {
        self.default_texture_array_set.handle()
    }

    pub fn allocate_texture_set(&mut self) -> Result<vk::DescriptorSet> {
        // Use static pool - textures live for mesh lifetime
        let descriptor = self.allocator.allocate_static_set(
            &self.texture_layout.handle(),
            self.texture_layout.bindings(),
        )?;
        let handle = descriptor.handle();
        self.dynamic_sets.insert(handle, descriptor);
        Ok(handle)
    }

    pub fn allocate_material_texture_set(&mut self) -> Result<vk::DescriptorSet> {
        // Use static pool - material textures live for mesh lifetime
        let descriptor = self.allocator.allocate_static_set(
            &self.material_texture_layout.handle(),
            self.material_texture_layout.bindings(),
        )?;
        let handle = descriptor.handle();
        self.dynamic_sets.insert(handle, descriptor);
        Ok(handle)
    }

    pub fn frame_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_layout.handle()
    }

    pub fn material_layout(&self) -> vk::DescriptorSetLayout {
        self.material_layout.handle()
    }

    pub fn material_texture_layout(&self) -> vk::DescriptorSetLayout {
        self.material_texture_layout.handle()
    }

    pub fn frame_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.frame_sets.get(index).map(|set| set.handle())
    }

    pub fn material_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.material_sets.get(index).map(|set| set.handle())
    }

    pub fn frame_set_count(&self) -> usize {
        self.frame_sets.len()
    }

    pub fn material_set_count(&self) -> usize {
        self.material_sets.len()
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

    fn material_texture_descriptor(&self, set: vk::DescriptorSet) -> Result<&DescriptorSet> {
        if set == self.default_texture_array_set.handle() {
            return Ok(&self.default_texture_array_set);
        }

        self.dynamic_sets.get(&set).ok_or_else(|| {
            AshError::VulkanError("Descriptor set not managed by DescriptorManager".into())
        })
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
