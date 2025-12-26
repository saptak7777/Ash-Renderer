//! Complete Forward+ Lighting Integration
//!
//! This module provides a single struct that manages all Forward+ resources:
//! - Light buffer management via LightManager
//! - Descriptor set management via ForwardPlusDescriptor
//! - ForwardPlusInfo UBO for shader data
//!
//! # Usage
//! ```ignore
//! // Create during renderer initialization
//! let forward_plus = ForwardPlusIntegration::new(device, allocator)?;
//!
//! // Each frame:
//! forward_plus.update_lights(&point_lights, &directional_lights);
//! forward_plus.on_resize(width, height);
//! forward_plus.upload_to_gpu(allocator)?;
//!
//! // During render:
//! forward_plus.bind(device, cmd, pipeline_layout);
//! ```

use ash::vk;
use std::sync::Arc;
use vk_mem::Alloc;

use crate::renderer::features::{DirectionalLight, ForwardPlusInfo, LightManager, PointLight};
use crate::renderer::forward_plus_descriptor::ForwardPlusDescriptor;
use crate::Result;

/// Complete Forward+ integration for the renderer
///
/// This struct manages:
/// - `LightManager` for CPU-side light logic and GPU buffer management
/// - `ForwardPlusDescriptor` for Set 4 descriptor binding
/// - `ForwardPlusInfo` UBO for shader constants
pub struct ForwardPlusIntegration {
    /// Light manager (owns light and tile buffers)
    lights: LightManager,
    /// Descriptor set for Set 4
    descriptor: ForwardPlusDescriptor,
    /// ForwardPlusInfo UBO buffer
    info_buf: vk::Buffer,
    info_alloc: vk_mem::Allocation,
    /// Whether the integration is initialized
    initialized: bool,
    /// Cached info data
    cached_info: ForwardPlusInfo,
}

impl ForwardPlusIntegration {
    /// Create a new Forward+ integration
    ///
    /// # Safety
    /// Device and allocator must be valid.
    pub unsafe fn new(device: Arc<ash::Device>, allocator: &vk_mem::Allocator) -> Result<Self> {
        let lights = LightManager::new();
        let descriptor = ForwardPlusDescriptor::new(device)?;

        // Create ForwardPlusInfo UBO
        let info_size = std::mem::size_of::<ForwardPlusInfo>() as u64;
        let info_buffer_info = vk::BufferCreateInfo::default()
            .size(info_size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let info_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (info_buf, info_alloc) = allocator
            .create_buffer(&info_buffer_info, &info_alloc_info)
            .map_err(|e| {
                crate::AshError::VulkanError(format!(
                    "ForwardPlusInfo buffer creation failed: {e:?}"
                ))
            })?;

        Ok(Self {
            lights,
            descriptor,
            info_buf,
            info_alloc,
            initialized: false,
            cached_info: ForwardPlusInfo::default(),
        })
    }

    /// Initialize GPU resources for Forward+ lighting.
    ///
    /// # Safety
    /// The caller must ensure that the provided allocator remains valid for the duration
    /// of the renderer's lifetime or until `destroy` is called.
    pub unsafe fn init(&mut self, allocator: &vk_mem::Allocator) {
        if self.initialized {
            return;
        }

        // Assertive: this should not fail during normal operation
        self.lights
            .create_buffers(allocator)
            .expect("Forward+ buffer allocation failed");
        self.initialized = true;
    }

    pub fn update_lights(
        &mut self,
        point_lights: &[PointLight],
        directional_lights: &[DirectionalLight],
    ) {
        // Just forward to light manager - it handles the internal slicing
        self.lights.update_lights(point_lights, directional_lights);
    }

    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.lights.on_resize(width, height);
        self.cached_info = self.lights.get_forward_plus_info();
    }

    /// Upload current light data to GPU buffers.
    ///
    /// # Safety
    /// The caller must ensure that the allocator is valid and that no concurrent
    /// access to the light buffers occurs during this operation.
    pub unsafe fn upload_to_gpu(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // Upload lights
        self.lights.upload_lights(allocator)?;

        // Upload ForwardPlusInfo (direct UBO update)
        self.cached_info = self.lights.get_forward_plus_info();
        let info_data = allocator.get_allocation_info(&self.info_alloc);
        let mapped_ptr = info_data.mapped_data;
        if !mapped_ptr.is_null() {
            // Avoid extra copies by writing directly if possible, though info is small
            std::ptr::copy_nonoverlapping(
                &self.cached_info as *const ForwardPlusInfo as *const u8,
                mapped_ptr as *mut u8,
                std::mem::size_of::<ForwardPlusInfo>(),
            );
        }

        // Update descriptor bindings - we only do this if buffers exist
        if let (Some(l_buf), Some(t_buf)) = (
            self.lights.get_light_buffer(),
            self.lights.get_tile_buffer(),
        ) {
            let info_size = std::mem::size_of::<ForwardPlusInfo>() as u64;
            self.descriptor.update(
                l_buf,
                self.lights.get_tile_buffer_size() as u64,
                t_buf,
                self.lights.get_tile_buffer_size() as u64,
                self.info_buf,
                info_size,
            );
        }

        Ok(())
    }

    /// Bind Set 4 for rendering
    ///
    /// Call this before issuing draw commands that use Forward+ lighting.
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn bind(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
    ) {
        if self.initialized {
            self.descriptor
                .bind(device, command_buffer, pipeline_layout);
        }
    }

    /// Get the descriptor set layout for pipeline creation
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor.layout()
    }

    /// Check if Forward+ is enabled (has lights)
    pub fn is_enabled(&self) -> bool {
        self.lights.is_enabled()
    }

    /// Get light count
    pub fn light_count(&self) -> usize {
        self.lights.light_count()
    }

    /// Get dispatch dimensions for light culling compute
    pub fn get_dispatch_dimensions(&self) -> (u32, u32, u32) {
        self.lights.get_dispatch_dimensions()
    }

    pub fn lights(&self) -> &LightManager {
        &self.lights
    }

    pub fn lights_mut(&mut self) -> &mut LightManager {
        &mut self.lights
    }

    /// Destroy all GPU resources
    /// Destroy all GPU resources.
    ///
    /// # Safety
    /// The caller must ensure that no GPU commands using these resources are
    /// currently executing on the device.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        self.lights.destroy_buffers(allocator);
        allocator.destroy_buffer(self.info_buf, &mut self.info_alloc);
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        // Unit tests are limited without active Vulkan context.
        let info = ForwardPlusInfo::default();
        assert_eq!(info.num_tiles, [0, 0]);
    }
}

impl Drop for ForwardPlusIntegration {
    fn drop(&mut self) {
        // LightManager will be dropped automatically via its own Drop impl
        log::debug!("ForwardPlusIntegration: Drop called (cleanup requires explicit destroy)");
    }
}
