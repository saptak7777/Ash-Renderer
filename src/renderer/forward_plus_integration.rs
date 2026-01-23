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
use bytemuck;
use std::sync::Arc;
use vk_mem::Alloc;

use crate::renderer::features::light_culling::{CullingCameraData, LightCullingPushConstants};
use crate::renderer::features::{
    DirectionalLight, ForwardPlusInfo, LightManager, PointLight, SpotLight,
};
use crate::renderer::forward_plus_descriptor::ForwardPlusDescriptor;
use crate::vulkan::{ComputePipeline, ShaderModule};
use crate::{AshError, Result};

/// Complete Forward+ integration for the renderer
///
/// This struct manages:
/// - `LightManager` for CPU-side light logic and GPU buffer management
/// - `ForwardPlusDescriptor` for Set 3 descriptor binding (Fragment Shader)
/// - `ForwardPlusInfo` UBO for shader constants
/// - Compute Pipeline for Light Culling
pub struct ForwardPlusIntegration {
    /// Light manager (owns light and tile buffers)
    lights: LightManager,
    /// Descriptor set for Set 3 (Fragment)
    descriptor: ForwardPlusDescriptor,
    /// ForwardPlusInfo UBO buffers (per-frame)
    info_bufs: Vec<vk::Buffer>,
    info_allocs: Vec<vk_mem::Allocation>,

    /// Camera Data UBO for Compute Shader (per-frame)
    camera_bufs: Vec<vk::Buffer>,
    camera_allocs: Vec<vk_mem::Allocation>,

    /// Compute Pipeline Resources
    compute_pipeline: Option<ComputePipeline>,
    compute_descriptor_pool: vk::DescriptorPool,
    compute_descriptor_sets: Vec<vk::DescriptorSet>,
    compute_descriptor_layout: vk::DescriptorSetLayout,

    /// Number of frames in flight
    frame_count: usize,

    /// Whether the integration is initialized
    initialized: bool,
    /// Cached info data
    cached_info: ForwardPlusInfo,
    // Whether this is the first frame (needs full descriptor update)
    // first_frame: bool,
}

impl ForwardPlusIntegration {
    /// Create a new Forward+ integration
    ///
    /// # Safety
    /// Device and allocator must be valid.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: &vk_mem::Allocator,
        frame_count: u32,
    ) -> Result<Self> {
        let lights = LightManager::new(frame_count as usize);
        let descriptor = ForwardPlusDescriptor::new(device, frame_count)?;

        // Create per-frame ForwardPlusInfo UBOs
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

        let mut info_bufs = Vec::with_capacity(frame_count as usize);
        let mut info_allocs = Vec::with_capacity(frame_count as usize);

        for _ in 0..frame_count {
            let (buf, alloc) = allocator
                .create_buffer(&info_buffer_info, &info_alloc_info)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!(
                        "ForwardPlusInfo buffer creation failed: {e:?}"
                    ))
                })?;
            info_bufs.push(buf);
            info_allocs.push(alloc);
        }

        // Create per-frame Camera Data UBOs for Compute Shader
        let camera_size = std::mem::size_of::<CullingCameraData>() as u64;
        let camera_buffer_info = vk::BufferCreateInfo::default()
            .size(camera_size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let mut camera_bufs = Vec::with_capacity(frame_count as usize);
        let mut camera_allocs = Vec::with_capacity(frame_count as usize);

        for _ in 0..frame_count {
            let (buf, alloc) = allocator
                .create_buffer(&camera_buffer_info, &info_alloc_info)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!(
                        "Camera data buffer creation failed: {e:?}"
                    ))
                })?;
            camera_bufs.push(buf);
            camera_allocs.push(alloc);
        }

        Ok(Self {
            lights,
            descriptor,
            info_bufs,
            info_allocs,
            camera_bufs,
            camera_allocs,
            compute_pipeline: None,
            compute_descriptor_pool: vk::DescriptorPool::null(),
            compute_descriptor_sets: Vec::new(),
            compute_descriptor_layout: vk::DescriptorSetLayout::null(),
            frame_count: frame_count as usize,
            initialized: false,
            cached_info: ForwardPlusInfo::default(),
            // first_frame: true,
        })
    }

    /// Initialize the compute pipeline.
    /// Must be called after depth buffer creation.
    ///
    /// # Safety
    /// All provided Vulkan handles (device, depth_sampler, depth_image_view) must be valid
    /// and remain valid for the duration of this call. The compute shader bytecode must
    /// be available in the OUT_DIR.
    pub unsafe fn init_pipeline(
        &mut self,
        device: Arc<ash::Device>,
        depth_sampler: vk::Sampler,
        depth_image_view: vk::ImageView,
    ) -> Result<()> {
        // Load shader
        let code = include_bytes!(concat!(env!("OUT_DIR"), "/light_culling.comp.spv"));
        let shader_module =
            ShaderModule::load_from_bytes(&device, code, vk::ShaderStageFlags::COMPUTE)?;

        // 1. Create Descriptor Set Layout (Set 0 for Compute)
        // NOTE: LightBuffer moved to Set 3 to match fragment shader
        // Set 0 now only contains DepthBuffer and CameraData
        let bindings = [
            // Binding 0: Depth buffer (sampler)
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            // Binding 1: Camera data (UBO)
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        self.compute_descriptor_layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!(
                    "Failed to create compute descriptor set layout: {e}"
                ))
            })?;

        // 2. Create Descriptor Pool (sized for frame_count sets)
        // Each set contains: 1 COMBINED_IMAGE_SAMPLER (depth) + 1 UNIFORM_BUFFER (camera)
        // Total pool must accommodate: frame_count sets × descriptors_per_set
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: self.frame_count as u32, // 1 depth sampler per set × frame_count sets
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: self.frame_count as u32, // 1 camera UBO per set × frame_count sets
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(self.frame_count as u32);

        self.compute_descriptor_pool =
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create compute descriptor pool: {e}"))
                })?;

        // 3. Allocate Descriptor Sets (one per frame)
        let layouts = vec![self.compute_descriptor_layout; self.frame_count];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.compute_descriptor_pool)
            .set_layouts(&layouts);

        self.compute_descriptor_sets =
            device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
                AshError::VulkanError(format!("Failed to allocate compute descriptor sets: {e}"))
            })?;

        // 4. Create Pipeline
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<LightCullingPushConstants>() as u32,
        };

        // We use the helper ComputePipeline from crate::vulkan which simplifies creation
        // NOTE: Set 0 contains DepthBuffer and CameraData
        // Set 3 contains Forward+ descriptor (LightBuffer, TileLightIndices, ForwardPlusInfo)
        // We need placeholder sets at 1 and 2 so Set 3 is at the correct index
        self.compute_pipeline = Some(
            ComputePipeline::builder(Arc::clone(&device))
                .with_shader(shader_module.module)
                .add_set_layout(self.compute_descriptor_layout) // Set 0: Depth buffer, camera
                .add_set_layout(vk::DescriptorSetLayout::null()) // Set 1: Placeholder
                .add_set_layout(vk::DescriptorSetLayout::null()) // Set 2: Placeholder
                .add_set_layout(self.descriptor.layout()) // Set 3: Forward+ (LightBuffer, TileLightIndices, ForwardPlusInfo)
                .add_push_constant(push_constant_range)
                .build()?,
        );

        // 5. Update all descriptor sets with depth buffer and their respective camera buffers
        let depth_image_info = vk::DescriptorImageInfo {
            sampler: depth_sampler,
            image_view: depth_image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        for frame_idx in 0..self.frame_count {
            let camera_info = vk::DescriptorBufferInfo {
                buffer: self.camera_bufs[frame_idx],
                offset: 0,
                range: std::mem::size_of::<CullingCameraData>() as u64,
            };

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.compute_descriptor_sets[frame_idx])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&depth_image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.compute_descriptor_sets[frame_idx])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&camera_info)),
            ];
            device.update_descriptor_sets(&writes, &[]);
        }

        log::info!(
            "Forward+ Compute Pipeline initialized with {} descriptor sets",
            self.frame_count
        );
        Ok(())
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
        spot_lights: &[SpotLight],
    ) {
        // Just forward to light manager - it handles the internal slicing
        self.lights
            .update_lights(point_lights, directional_lights, spot_lights);
    }

    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.lights.on_resize(width, height);
        self.cached_info = self.lights.get_forward_plus_info();
    }

    /// Update camera data for light culling.
    ///
    /// # Safety
    /// Allocator must be valid. The internal camera buffer must have been successfully
    /// allocated during `new()`.
    pub unsafe fn update_camera(
        &mut self,
        allocator: &vk_mem::Allocator,
        frame_index: usize,
        view: &[[f32; 4]; 4],
        projection: &[[f32; 4]; 4],
        camera_pos: &[f32; 4],
    ) -> Result<()> {
        let inv_projection = glam::Mat4::from_cols_array_2d(projection)
            .inverse()
            .to_cols_array_2d();
        let data = CullingCameraData {
            view: *view,
            projection: *projection,
            inv_projection,
            camera_pos: *camera_pos,
        };

        let info = allocator.get_allocation_info(&self.camera_allocs[frame_index]);
        let ptr = info.mapped_data;
        if !ptr.is_null() {
            std::ptr::copy_nonoverlapping(&data, ptr as *mut CullingCameraData, 1);
        }
        Ok(())
    }

    /// Upload current light data to GPU buffers.
    ///
    /// # Safety
    /// The caller must ensure that the allocator is valid and that no concurrent
    /// access to the light buffers occurs during this operation.
    pub unsafe fn upload_to_gpu(
        &mut self,
        allocator: &vk_mem::Allocator,
        _device: &ash::Device,
        frame_index: usize,
    ) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // CRITICAL: Recreate tile buffer if screen size changed
        // Must happen before upload_lights and descriptor update
        // This also handles min size 1024 logic internally now
        let buffers_recreated = self.lights.recreate_tile_buffer_if_needed(allocator)?;

        // Upload lights
        self.lights.upload_lights(allocator, frame_index)?;

        // Upload ForwardPlusInfo to the current frame's UBO
        self.cached_info = self.lights.get_forward_plus_info();
        let info_data = allocator.get_allocation_info(&self.info_allocs[frame_index]);
        let mapped_ptr = info_data.mapped_data;
        if !mapped_ptr.is_null() {
            std::ptr::copy_nonoverlapping(
                &self.cached_info as *const ForwardPlusInfo as *const u8,
                mapped_ptr as *mut u8,
                std::mem::size_of::<ForwardPlusInfo>(),
            );
        }

        let info_size = std::mem::size_of::<ForwardPlusInfo>() as u64;
        let light_buffer_size = (crate::renderer::features::MAX_LIGHTS
            * std::mem::size_of::<crate::renderer::features::GpuLight>())
            as u64;

        // If buffers were recreated, update all descriptor sets to point to new buffers
        // Recreate happens on resize (idle) so it is safe to update all sets.
        // Otherwise, update only the current frame's set to avoid touching in-flight sets.
        let update_range = if buffers_recreated {
            0..self.lights.frame_count()
        } else {
            frame_index..frame_index + 1
        };

        for idx in update_range {
            // Update descriptor bindings - we only do this if buffers exist
            if let (Some(l_buf), Some(t_buf)) = (
                self.lights.get_light_buffer(idx),
                self.lights.get_tile_buffer(idx),
            ) {
                // Update Fragment Shader Descriptor (Set 3) with frame-specific info buffer
                self.descriptor.update(
                    idx,
                    l_buf,
                    light_buffer_size,
                    t_buf,
                    self.lights.get_tile_buffer_size() as u64,
                    self.info_bufs[idx],
                    info_size,
                );
            }
        }

        // Update Compute Shader Descriptor (Set 0) - only for current frame
        // NOTE: Camera buffer is already bound in init_pipeline, but we may need to update it
        // if the camera data changes. For now, we rely on update_camera_data() being called separately.

        Ok(())
    }

    /// Dispatch light culling compute shader.
    ///
    /// # Safety
    /// Command buffer must be in recording state. Device must be valid. All internal
    /// buffers (light, tile, info, camera) must have been initialized via `init()`
    /// and `upload_to_gpu()`.
    pub unsafe fn dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        device: &ash::Device,
        frame_index: usize,
    ) {
        if let Some(pipeline) = &self.compute_pipeline {
            if self.compute_descriptor_sets.is_empty() {
                return;
            }

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle(),
            );

            // Bind Set 0: Depth buffer, camera buffer (frame-specific)
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[self.compute_descriptor_sets[frame_index]],
                &[],
            );

            // Bind Set 3: Forward+ descriptor (LightBuffer, TileLightIndices, ForwardPlusInfo)
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                3,
                &[self.descriptor.descriptor_set(frame_index)],
                &[],
            );

            let (tx, ty, tz) = self.lights.get_dispatch_dimensions();

            // Reconstruct screen size for PC
            // We use the cached ForwardPlusInfo to determine total size
            let width = self.cached_info.num_tiles[0] * self.cached_info.tile_size;
            let height = self.cached_info.num_tiles[1] * self.cached_info.tile_size;

            let push_constants = LightCullingPushConstants {
                screen_size: [width, height],
                light_count: self.lights.light_count() as u32,
                _padding: 0,
            };

            device.cmd_push_constants(
                command_buffer,
                pipeline.layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            device.cmd_dispatch(command_buffer, tx, ty, tz);

            // Pipeline barrier to ensure writes are visible to fragment shader
            // LightManager owns tile buffer, used in Set 2 binding 2.
            if let Some(t_buf) = self.lights.get_tile_buffer(frame_index) {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(t_buf)
                    .offset(0)
                    .size(vk::WHOLE_SIZE); // Or proper size

                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            }
        }
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
        frame_index: usize,
    ) {
        if self.initialized {
            self.descriptor
                .bind(device, command_buffer, pipeline_layout, frame_index);
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
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator, device: &ash::Device) {
        if !self.initialized {
            return;
        }

        if self.compute_descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.compute_descriptor_pool, None);
        }
        if self.compute_descriptor_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.compute_descriptor_layout, None);
        }
        // ComputePipeline drops itself

        self.lights.destroy_buffers(allocator);

        // Destroy all per-frame info buffers
        for (buf, alloc) in self.info_bufs.iter().zip(self.info_allocs.iter_mut()) {
            allocator.destroy_buffer(*buf, alloc);
        }

        // Destroy all per-frame camera buffers
        for (buf, alloc) in self.camera_bufs.iter().zip(self.camera_allocs.iter_mut()) {
            allocator.destroy_buffer(*buf, alloc);
        }

        self.initialized = false;
    }
}

impl Drop for ForwardPlusIntegration {
    fn drop(&mut self) {
        // LightManager will be dropped automatically via its own Drop impl
        log::debug!("ForwardPlusIntegration: Drop called (cleanup requires explicit destroy)");
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
