//! VSM Feature - High-level integration of Virtual Shadow Maps

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::Result;

use super::clipmap_manager::ClipmapManager;
use super::compute_pipelines::VsmComputePipelines;
use super::page_manager::{PageManager, PageManagerStats};
use super::resources::{VsmConfig, VsmResources};
use super::shadow_pass::VsmShadowPass;
use crate::renderer::passes::ShadowCullPass;

/// VSM Feature - Complete virtual shadow map system
pub struct VsmFeature {
    /// GPU resources
    pub resources: VsmResources,

    /// CPU-side page manager
    page_manager: PageManager,

    /// Clipmap manager for directional lights
    clipmap_manager: Option<ClipmapManager>,

    /// Compute pipelines
    compute_pipelines: VsmComputePipelines,

    /// Shadow rendering pass
    pub shadow_pass: VsmShadowPass,

    /// Shadow culling pass
    pub shadow_cull_pass: ShadowCullPass,

    /// Current frame index
    current_frame: u32,

    /// VSM device handle
    device: Arc<ash::Device>,

    /// Allocator for resource cleanup
    allocator: Arc<Allocator>,

    /// Enabled state
    enabled: bool,

    /// Whether the feature has been destroyed
    destroyed: bool,
}

impl VsmFeature {
    /// Create a new VSM feature
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this feature.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        config: VsmConfig,
        frame_count: u32,
    ) -> Result<Self> {
        log::info!("Creating VSM feature with {} frames", frame_count);

        // Create resources
        let resources =
            VsmResources::new(Arc::clone(&device), Arc::clone(&allocator), config.clone())?;

        // Create page manager
        let page_manager = PageManager::new(
            config.virtual_resolution,
            config.physical_resolution,
            config.page_size,
            config.clipmap_levels,
        );

        // Create compute pipelines
        let compute_pipelines =
            VsmComputePipelines::new(Arc::clone(&device), Arc::clone(&allocator))?;

        // Create shadow pass
        let shadow_pass =
            VsmShadowPass::new(Arc::clone(&device), Arc::clone(&allocator), &resources)?;

        // Create shadow cull pass
        let shadow_cull_pass = ShadowCullPass::new(
            Arc::clone(&device),
            &allocator,
            frame_count,
            2048, // max_objects
            config.clipmap_levels,
        )?;

        // Create clipmap manager if clipmaps are enabled
        let clipmap_manager = if config.clipmap_levels > 0 {
            Some(ClipmapManager::new(config.clone()))
        } else {
            None
        };

        log::info!("VSM feature created successfully");

        Ok(Self {
            resources,
            page_manager,
            clipmap_manager,
            compute_pipelines,
            shadow_pass,
            shadow_cull_pass,
            current_frame: 0,
            device,
            allocator,
            enabled: true,
            destroyed: false,
        })
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self, frame_index: u32, camera_pos: glam::Vec3) {
        self.current_frame = frame_index;
        self.page_manager.begin_frame(frame_index);

        // Update clipmap centers if enabled
        if let Some(clipmap) = &mut self.clipmap_manager {
            clipmap.update(camera_pos);
        }

        // Update metadata buffer
        if let Err(e) = self.resources.update_metadata(frame_index) {
            log::error!("Failed to update VSM metadata: {e:?}");
        }
    }

    /// Get page manager statistics
    pub fn stats(&self) -> PageManagerStats {
        self.page_manager.stats()
    }

    /// Check if VSM is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable VSM
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get physical cache image view for binding
    pub fn physical_cache_view(&self) -> vk::ImageView {
        self.resources.physical_cache_view
    }

    /// Get physical cache sampler
    pub fn physical_cache_sampler(&self) -> vk::Sampler {
        self.resources.physical_cache_sampler
    }

    /// Get page table image view
    pub fn page_table_view(&self) -> vk::ImageView {
        self.resources.page_table_view
    }

    /// Get page table sampler
    pub fn page_table_sampler(&self) -> vk::Sampler {
        self.resources.page_table_sampler
    }

    /// Get shadow render pass
    pub fn shadow_render_pass(&self) -> vk::RenderPass {
        self.shadow_pass.render_pass
    }

    /// Get shadow framebuffer
    pub fn shadow_framebuffer(&self) -> vk::Framebuffer {
        self.shadow_pass.framebuffer
    }

    /// Get configuration
    pub fn config(&self) -> &VsmConfig {
        self.resources.config()
    }

    /// Get clipmap manager (if enabled)
    pub fn clipmap_manager(&self) -> Option<&ClipmapManager> {
        self.clipmap_manager.as_ref()
    }

    /// Get mutable clipmap manager (if enabled)
    pub fn clipmap_manager_mut(&mut self) -> Option<&mut ClipmapManager> {
        self.clipmap_manager.as_mut()
    }

    /// Check if clipmaps are enabled
    pub fn has_clipmaps(&self) -> bool {
        self.clipmap_manager.is_some()
    }

    /// Render shadows using GPU-driven culling and indirect drawing
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn render_shadows(
        &self,
        cmd: vk::CommandBuffer,
        light_dir: glam::Vec3,
        object_count: u32,
        frame_index: usize,
        frame_set: vk::DescriptorSet,
        bindless_set: vk::DescriptorSet,
        frame_ptr: u64,
        vertex_ptr: u64,
        instance_ptr: u64,
        material_ptr: u64,
        index_ptr: u64,
        light_ptr: u64,
        tile_ptr: u64,
    ) {
        let allocations = self.page_manager.get_allocated_pages();
        if allocations.is_empty() {
            return;
        }

        let page_size = self.resources.config().page_size;
        let page_table_res = self.resources.config().page_table_resolution();

        // 1. Clear count buffers for this frame
        let count_buffer = self.shadow_cull_pass.count_buffers[frame_index];
        self.device
            .cmd_fill_buffer(cmd, count_buffer, 0, vk::WHOLE_SIZE, 0);

        // 2. Memory barrier to ensure clear is finished
        let clear_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(count_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[clear_barrier],
            &[],
        );

        // 3. Dispatch culling for each clipmap level and collect matrices
        let mut level_matrices = Vec::new();
        if let Some(ref clipmap_manager) = self.clipmap_manager {
            for level_idx in 0..clipmap_manager.level_count() {
                if let Some(level) = clipmap_manager.level(level_idx) {
                    let view_proj = level.view_projection_matrix(light_dir, page_table_res);
                    level_matrices.push(view_proj);

                    self.shadow_cull_pass.cull_shadows(
                        cmd,
                        frame_index,
                        view_proj,
                        object_count,
                        0, // base_index
                        level_idx as u32,
                        instance_ptr,
                    );

                    // Memory barrier between levels for atomic counter visibility (Suggested)
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(count_buffer)
                            .offset((level_idx as vk::DeviceSize) * 4)
                            .size(4)],
                        &[],
                    );
                }
            }
        }

        // 4. Pipeline barrier: Compute -> Indirect/Draw
        let cull_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(self.shadow_cull_pass.indirect_buffers[frame_index])
            .offset(0)
            .size(vk::WHOLE_SIZE);

        let count_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(count_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::DRAW_INDIRECT,
            vk::DependencyFlags::empty(),
            &[],
            &[cull_barrier, count_barrier],
            &[],
        );

        // 5. Render using indirect commands
        self.shadow_pass.render_shadows_indirect(
            cmd,
            &allocations,
            page_size,
            frame_set,
            bindless_set,
            self.shadow_cull_pass.indirect_buffers[frame_index],
            count_buffer,
            2048, // max_commands_per_level
            &level_matrices,
            material_ptr,
            index_ptr,
            light_ptr,
            tile_ptr,
            frame_ptr,
            vertex_ptr,
            instance_ptr,
        );
    }

    /// Get shadow pipeline (if created)
    pub fn shadow_pipeline(&self) -> Option<vk::Pipeline> {
        self.shadow_pass.pipeline()
    }

    /// Get shadow pipeline layout (if created)
    pub fn shadow_pipeline_layout(&self) -> Option<vk::PipelineLayout> {
        self.shadow_pass.pipeline_layout()
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        log::debug!("Destroying VSM feature");

        self.shadow_pass.destroy(&self.allocator);
        self.shadow_cull_pass.destroy(&self.allocator);
        self.compute_pipelines.destroy();

        self.resources.destroy();

        log::debug!("VSM feature destroyed");
    }
}

impl Drop for VsmFeature {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}

/// Helper to create a default VSM configuration
pub fn default_vsm_config() -> VsmConfig {
    VsmConfig::default()
}

/// Helper to create a high-quality VSM configuration
pub fn high_quality_vsm_config() -> VsmConfig {
    VsmConfig {
        virtual_resolution: 16384,
        physical_resolution: 8192, // Larger cache
        page_size: 128,
        max_requests_per_frame: 2048,
        debug_mode: false,
        clipmap_levels: 8,
        clipmap_base_extent: 100.0,
    }
}

/// Helper to create a performance-focused VSM configuration
pub fn performance_vsm_config() -> VsmConfig {
    VsmConfig {
        virtual_resolution: 8192,
        physical_resolution: 2048, // Smaller cache
        page_size: 128,
        max_requests_per_frame: 512,
        debug_mode: false,
        clipmap_levels: 6, // Fewer levels for performance
        clipmap_base_extent: 80.0,
    }
}
