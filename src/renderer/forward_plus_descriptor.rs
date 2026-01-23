//! Forward+ Descriptor Set Management
//!
//! Manages Descriptor Set 3 for Forward+ lighting in fragment shaders.
//! This binds the light buffer, tile indices, and forward+ info UBO.

use ash::vk;
use std::sync::Arc;

use crate::Result;

/// Forward+ descriptor set layout bindings
///
/// Matches the shader layout:
/// - binding 0: LightBuffer (storage buffer)
/// - binding 1: TileLightIndices (storage buffer)  
/// - binding 2: ForwardPlusInfo (uniform buffer)
pub struct ForwardPlusDescriptor {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    device: Arc<ash::Device>,
}

impl ForwardPlusDescriptor {
    /// Create the Forward+ descriptor set layout and allocate a set
    ///
    /// # Safety
    /// Device must be valid.
    pub unsafe fn new(device: Arc<ash::Device>, frame_count: u32) -> Result<Self> {
        // Layout bindings
        let bindings = [
            // Binding 0: Light buffer (storage)
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE),
            // Binding 1: Tile indices (storage)
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE),
            // Binding 2: ForwardPlusInfo (uniform)
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Forward+ layout creation failed: {e:?}"))
            })?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2 * frame_count, // Light + Tile
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1 * frame_count, // ForwardPlusInfo
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(frame_count)
            .pool_sizes(&pool_sizes);

        let pool = device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Forward+ pool creation failed: {e:?}"))
            })?;

        // Allocate descriptor sets
        let mut layouts = Vec::with_capacity(frame_count as usize);
        for _ in 0..frame_count {
            layouts.push(layout);
        }

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let sets = device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
            crate::AshError::VulkanError(format!("Forward+ set allocation failed: {e:?}"))
        })?;

        log::info!("Forward+ descriptor sets created (count: {})", sets.len());

        Ok(Self {
            layout,
            pool,
            descriptor_sets: sets,
            device,
        })
    }

    /// Update the descriptor set with buffer bindings
    ///
    /// # Arguments
    /// * `light_buffer` - Storage buffer containing GpuLight array
    /// * `light_buffer_size` - Size in bytes
    /// * `tile_buffer` - Storage buffer containing tile light indices
    /// * `tile_buffer_size` - Size in bytes
    /// * `info_buffer` - Uniform buffer containing ForwardPlusInfo
    /// * `info_buffer_size` - Size in bytes
    ///
    /// # Safety
    /// All buffers must be valid and allocated.
    pub unsafe fn update(
        &self,
        frame_index: usize,
        light_buffer: vk::Buffer,
        light_buffer_size: u64,
        tile_buffer: vk::Buffer,
        tile_buffer_size: u64,
        info_buffer: vk::Buffer,
        info_buffer_size: u64,
    ) {
        let light_info = vk::DescriptorBufferInfo {
            buffer: light_buffer,
            offset: 0,
            range: light_buffer_size,
        };

        let tile_info = vk::DescriptorBufferInfo {
            buffer: tile_buffer,
            offset: 0,
            range: tile_buffer_size,
        };

        let fp_info = vk::DescriptorBufferInfo {
            buffer: info_buffer,
            offset: 0,
            range: info_buffer_size,
        };

        let descriptor_set = self.descriptor_sets[frame_index];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&light_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&tile_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&fp_info)),
        ];

        self.device.update_descriptor_sets(&writes, &[]);
        log::debug!("Forward+ descriptors updated");
    }

    /// Get the descriptor set layout
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Get the allocated descriptor set
    pub fn descriptor_set(&self, frame_index: usize) -> vk::DescriptorSet {
        self.descriptor_sets[frame_index]
    }

    /// Record bind command for Set 4
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn bind(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        frame_index: usize,
    ) {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            3, // Set 3
            &[self.descriptor_sets[frame_index]],
            &[],
        );
    }
}

impl Drop for ForwardPlusDescriptor {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
        log::info!("Forward+ descriptor set destroyed");
    }
}

#[cfg(test)]
mod tests {
    // GPU tests would require Vulkan context
}
