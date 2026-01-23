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
        // 5: VSM Page Table
        // 6: VSM Physical Cache
        let environment_layout = DescriptorSetLayoutBuilder::new()
            .add_binding(
                5, // VSM Page Table
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .add_binding(
                6, // VSM Physical Cache
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

    /// Bind VSM resources (Page Table and Physical Cache) to environment descriptor set
    pub fn bind_vsm_resources(
        &self,
        frame_index: usize,
        page_table_view: vk::ImageView,
        page_table_sampler: vk::Sampler,
        physical_cache_view: vk::ImageView,
        physical_cache_sampler: vk::Sampler,
    ) -> Result<()> {
        let descriptor = self.environment_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Environment descriptor set index out of bounds".into())
        })?;

        // Binding 5: VSM Page Table (R32_UINT)
        let page_table_info = vk::DescriptorImageInfo {
            sampler: page_table_sampler,
            image_view: page_table_view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        descriptor.update_image_at(
            5,
            0,
            page_table_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        // Binding 6: VSM Physical Cache (R32_FLOAT)
        let physical_cache_info = vk::DescriptorImageInfo {
            sampler: physical_cache_sampler,
            image_view: physical_cache_view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        descriptor.update_image_at(
            6,
            0,
            physical_cache_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        Ok(())
    }

    pub fn recreate_frame_sets(&mut self, frame_count: u32) -> Result<()> {
        if self.frame_sets.len() == frame_count as usize {
            return Ok(());
        }
        self.frame_sets =
            Self::create_descriptor_sets(frame_count, &self.frame_layout, &mut self.allocator)?;
        Ok(())
    }

    pub fn recreate_environment_sets(&mut self, frame_count: u32) -> Result<()> {
        if self.environment_sets.len() == frame_count as usize {
            return Ok(());
        }
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

    pub fn bind_defaults(
        &self,
        frame_index: usize,
        default_uint_2d: &vk::DescriptorImageInfo, // R32_UINT for VSM page table
        default_2d: &vk::DescriptorImageInfo,
    ) -> Result<()> {
        let descriptor = self.environment_sets.get(frame_index).ok_or_else(|| {
            AshError::VulkanError("Environment descriptor set index out of bounds".into())
        })?;

        // 4: Shadow Map - REMOVED

        // 5: VSM Page Table (2D UINT) - Default to 0xFFFFFFFF (invalid page)
        descriptor.update_image_at(
            5,
            0,
            *default_uint_2d,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;
        // 6: VSM Physical Cache (2D) - Default to white (no shadow)
        let cache_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            ..*default_2d
        };
        descriptor.update_image_at(6, 0, cache_info, vk::DescriptorType::COMBINED_IMAGE_SAMPLER)?;

        Ok(())
    }

    pub fn environment_set_count(&self) -> usize {
        self.environment_sets.len()
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
