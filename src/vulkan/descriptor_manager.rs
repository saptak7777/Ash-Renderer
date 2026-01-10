use ash::vk;
use log::info;

use std::sync::Arc;

use crate::renderer::resource_registry::ResourceRegistry;
use crate::{AshError, Result};

use super::descriptor_allocator::DescriptorAllocator;
use super::descriptor_layout::DescriptorSetLayoutBuilder;
use super::descriptor_set::DescriptorSet;

const EXTRA_TEXTURE_SETS: u32 = 2048;

/// Manages descriptor layouts and descriptor sets for frame and environment resources.
pub struct DescriptorManager {
    allocator: DescriptorAllocator,
    frame_layout: super::descriptor_layout::DescriptorSetLayout,
    environment_layout: super::descriptor_layout::DescriptorSetLayout,
    frame_sets: Vec<DescriptorSet>,
    environment_sets: Vec<DescriptorSet>,
}

impl DescriptorManager {
    pub fn new(
        device: Arc<ash::Device>,
        frame_count: u32,
        resource_registry: Option<Arc<ResourceRegistry>>,
    ) -> Result<Self> {
        info!("Creating simplified descriptor manager for {frame_count} frames");

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

        // Environment layout (Set 2)
        // 0: Irradiance Map
        // 1: Prefiltered Map
        // 2: BRDF LUT
        // 3: Skybox Map
        // 4: Shadow Map
        let environment_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                0, // Irradiance Map
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                1, // Prefiltered Map
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                2, // BRDF LUT
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                3, // Skybox Map
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                4, // Shadow Map
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .build(Arc::clone(&device))?;

        let frame_sets = Self::create_descriptor_sets(frame_count, &frame_layout, &mut allocator)?;
        let environment_sets =
            Self::create_descriptor_sets(frame_count, &environment_layout, &mut allocator)?;

        info!(
            "Allocated descriptor sets (frame: {}, environment: {})",
            frame_sets.len(),
            environment_sets.len()
        );

        Ok(Self {
            allocator,
            frame_layout,
            environment_layout,
            frame_sets,
            environment_sets,
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

    // Materials are now in bindless Set 1, Binding 1 - no material uniform binding needed
    // Legacy methods removed: bind_material_uniform, bind_material_textures, etc.

    pub fn frame_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.frame_sets.get(index).map(|set| set.handle())
    }

    pub fn frame_set_count(&self) -> usize {
        self.frame_sets.len()
    }

    /// Get mutable access to the allocator for external allocation (e.g., bindless)
    pub fn allocator_mut(&mut self) -> &mut DescriptorAllocator {
        &mut self.allocator
    }

    pub fn environment_layout(&self) -> vk::DescriptorSetLayout {
        self.environment_layout.handle()
    }

    pub fn environment_set(&self, index: usize) -> Option<vk::DescriptorSet> {
        self.environment_sets.get(index).map(|set| set.handle())
    }

    /// Bind shadow map texture to shadow descriptor set for given frame
    pub fn bind_shadow_map(
        &self,
        frame_index: usize,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        let descriptor = self.environment_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Environment descriptor set index out of bounds".into())
        })?;

        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };
        descriptor.update_image_at(4, 0, info, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)?;
        Ok(())
    }

    /// Bind IBL resources to the environment descriptor set
    pub fn bind_ibl_resources(
        &self,
        frame_index: usize,
        resources: &crate::vulkan::IBLResources,
    ) -> Result<()> {
        let descriptor = self.environment_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Environment descriptor set index out of bounds".into())
        })?;

        crate::vulkan::IBLDescriptorSet::update(descriptor, resources)
    }

    pub fn recreate_frame_sets(&mut self, frame_count: u32) -> Result<()> {
        self.frame_sets =
            Self::create_descriptor_sets(frame_count, &self.frame_layout, &mut self.allocator)?;
        Ok(())
    }

    pub fn recreate_environment_sets(&mut self, frame_count: u32) -> Result<()> {
        self.environment_sets = Self::create_descriptor_sets(
            frame_count,
            &self.environment_layout,
            &mut self.allocator,
        )?;
        Ok(())
    }

    pub fn frame_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_layout.handle()
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
