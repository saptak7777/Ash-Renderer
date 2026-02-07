//! Shadow System - High-level abstraction for Virtual Shadow Maps
//!
//! This module provides a clean interface for shadow rendering, encapsulating
//! the complexity of VSM (Virtual Shadow Maps) from the main renderer.

use ash::vk;
use std::sync::Arc;

use crate::renderer::features::vsm::{VsmConfig, VsmFeature};
use crate::renderer::resources::Texture;
use crate::vulkan::{Allocator, BindlessManager};
use crate::Result;

/// Render context containing BDA pointers and geometry data
pub struct ShadowRenderContext {
    pub light_direction: glam::Vec3,
    pub object_count: u32,
    pub frame_index: usize,
    pub frame_descriptor_set: vk::DescriptorSet,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub frame_ptr: u64,
    pub vertex_ptr: u64,
    pub instance_ptr: u64,
    pub material_ptr: u64,
    pub index_ptr: u64,
    pub light_ptr: u64,
    pub tile_ptr: u64,
    pub transform_ptr: u64,
    pub transform_index: u32,
}

/// Shadow System - Encapsulates all shadow mapping logic
pub struct ShadowSystem {
    /// Core VSM implementation
    vsm_feature: VsmFeature,

    /// Bindless indices for VSM resources
    vsm_page_index: u32,
    vsm_cache_index: u32,

    /// Default textures for VSM
    _default_uint_texture: Texture,
    _default_array_texture: Texture,

    /// Device handle for barriers
    device: Arc<ash::Device>,

    /// Allocator for cleanup
    _allocator: Arc<Allocator>,
}

impl ShadowSystem {
    /// Create a new shadow system
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this system.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        bindless_manager: &mut BindlessManager,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        config: VsmConfig,
        frame_count: u32,
    ) -> Result<Self> {
        log::info!("Creating Shadow System");

        // Create VSM feature
        let vsm_feature = VsmFeature::new(
            Arc::clone(&device),
            Arc::clone(&allocator),
            config.clone(),
            frame_count,
        )?;

        // Create default textures for VSM bindless slots
        let default_uint_texture = Texture::create_vsm_default_uint(
            Arc::clone(&allocator),
            Arc::clone(&device),
            command_pool,
            queue,
        )?;

        let default_array_texture = Texture::create_vsm_default_array(
            Arc::clone(&allocator),
            Arc::clone(&device),
            command_pool,
            queue,
            config.clipmap_levels,
        )?;

        // Register VSM resources with bindless manager
        let vsm_page_index = bindless_manager.add_page_table(
            vsm_feature.page_table_view(),
            vsm_feature.page_table_sampler(),
        )?;

        let vsm_cache_index = bindless_manager.add_sampled_image(
            vsm_feature.physical_cache_view(),
            vsm_feature.physical_cache_sampler(),
        )?;

        log::info!(
            "Shadow System created: page_index={}, cache_index={}",
            vsm_page_index,
            vsm_cache_index
        );

        Ok(Self {
            vsm_feature,
            vsm_page_index,
            vsm_cache_index,
            _default_uint_texture: default_uint_texture,
            _default_array_texture: default_array_texture,
            device,
            _allocator: allocator,
        })
    }

    /// Update shadow system state for the current frame
    pub fn update(&mut self, camera_pos: glam::Vec3, frame_index: u32) -> Result<()> {
        self.vsm_feature.begin_frame(frame_index, camera_pos);
        Ok(())
    }

    /// Render shadows using the provided context
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn render(&self, command_buffer: vk::CommandBuffer, ctx: &ShadowRenderContext) {
        // Skip if shadow pipeline is not ready
        if self.vsm_feature.shadow_pipeline_layout().is_none() {
            log::warn!("Shadow pipeline not ready, skipping shadow pass.");
            return;
        }

        // Validate BDA pointers
        if ctx.vertex_ptr == 0 || ctx.index_ptr == 0 {
            log::warn!(
                "Shadow pass: Invalid BDA pointers (V: {}, I: {}). Skipping.",
                ctx.vertex_ptr,
                ctx.index_ptr
            );
            return;
        }

        // Render shadows via VSM feature
        self.vsm_feature.render_shadows(
            command_buffer,
            ctx.light_direction,
            ctx.object_count,
            ctx.frame_index,
            ctx.frame_descriptor_set,
            ctx.bindless_descriptor_set,
            ctx.frame_ptr,
            ctx.vertex_ptr,
            ctx.instance_ptr,
            ctx.material_ptr,
            ctx.index_ptr,
            ctx.light_ptr,
            ctx.tile_ptr,
            ctx.transform_ptr,
            ctx.transform_index,
        );
    }

    /// Insert barrier to transition shadow map for reading in main pass
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn barrier_transition_for_read(&self, command_buffer: vk::CommandBuffer) {
        let vsm_barrier = vk::ImageMemoryBarrier::default()
            .image(self.vsm_feature.resources.physical_cache)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        self.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vsm_barrier],
        );
    }

    /// Get shadow map view for binding
    pub fn get_shadow_map_view(&self) -> vk::ImageView {
        self.vsm_feature.physical_cache_view()
    }

    /// Get VSM page table index
    pub fn vsm_page_index(&self) -> u32 {
        self.vsm_page_index
    }

    /// Get VSM cache index
    pub fn vsm_cache_index(&self) -> u32 {
        self.vsm_cache_index
    }

    /// Check if shadow system is enabled
    pub fn is_enabled(&self) -> bool {
        self.vsm_feature.is_enabled()
    }

    /// Get mutable reference to internal VSM feature for advanced operations
    pub fn vsm_feature_mut(&mut self) -> &mut VsmFeature {
        &mut self.vsm_feature
    }

    /// Get reference to internal VSM feature
    pub fn vsm_feature(&self) -> &VsmFeature {
        &self.vsm_feature
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying Shadow System");
        self.vsm_feature.destroy();
        // Textures will be dropped automatically via their Drop impl
    }
}

impl Drop for ShadowSystem {
    fn drop(&mut self) {
        // SAFETY: The ShadowSystem owns its resources and the user is expected
        // to ensure the GPU is idle before dropping the system (Standard RAII).
        unsafe {
            self.destroy();
        }
    }
}
