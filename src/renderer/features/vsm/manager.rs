//! VSM Feature - High-level integration of Virtual Shadow Maps

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::clipmap::calculate_clipmap_view_proj;
use super::clipmap_manager::ClipmapManager;
use super::compute_pipelines::VsmComputePipelines;
use super::data::VsmGlobalInfo;
use super::page_manager::{PageManager, PageManagerStats};
use super::resources::{VsmConfig, VsmResources};
use super::shadow_pass::VsmShadowPass;
use crate::renderer::passes::ShadowCullPass;

/// VSM GPU Resources and Pipelines
struct VsmInner {
    /// GPU resources (buffers, images)
    resources: VsmResources,
    /// Compute pipelines
    compute_pipelines: VsmComputePipelines,
    /// Shadow rendering pass
    shadow_pass: VsmShadowPass,
    /// Shadow culling pass
    shadow_cull_pass: ShadowCullPass,
}

/// VSM Manager - Complete virtual shadow map system
pub struct VsmManager {
    /// Internal GPU resources and pipelines
    inner: Option<VsmInner>,

    /// Global configuration and state for GPU
    pub global_info: VsmGlobalInfo,

    /// CPU-side page manager
    page_manager: PageManager,

    /// Clipmap manager for directional lights
    clipmap_manager: Option<ClipmapManager>,

    /// Bindless indices for VSM resources
    pub page_table_bindless_index: u32,
    pub physical_memory_bindless_index: u32,

    /// Current frame index
    current_frame: u32,

    /// VSM device handle
    device: Arc<ash::Device>,

    /// Allocator for resource cleanup
    allocator: Arc<Allocator>,

    /// Enabled state
    enabled: bool,
}

/// Arguments for VSM shadow rendering to satisfy Clippy's argument count limit.
pub struct VsmShadowArgs {
    pub cmd: vk::CommandBuffer,
    pub light_view_proj: glam::Mat4,
    pub bindless_set: vk::DescriptorSet,
    pub vertex_addr: u64,
    pub index_addr: u64,
    pub object_addr: u64,
    pub object_count: u32,
}

impl VsmManager {
    /// Create a new VSM manager
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this manager.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        bindless_manager: &mut crate::vulkan::BindlessManager,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        config: VsmConfig,
        frame_count: u32,
    ) -> Result<Self> {
        log::info!("Creating VSM manager with {frame_count} frames");

        // Create resources
        let resources = VsmResources::new(
            Arc::clone(&device),
            Arc::clone(&allocator),
            command_pool,
            queue,
            config.clone(),
            frame_count,
        )?;

        // Create page manager
        let page_manager = PageManager::new(
            config.virtual_resolution,
            config.physical_resolution,
            config.page_size,
            config.clipmap_levels,
        );

        // Create shadow pass
        let shadow_pass =
            VsmShadowPass::new(Arc::clone(&device), Arc::clone(&allocator), &resources)?;

        // Create compute pipelines
        let compute_pipelines = VsmComputePipelines::new(
            Arc::clone(&device),
            resources.compute_layout,
            config.max_requests_per_frame,
        )?;

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

        // Create initial global info
        let global_info = VsmGlobalInfo {
            light_view_projections: [glam::Mat4::IDENTITY; 16],
            view_proj: glam::Mat4::IDENTITY,
            inv_view_proj: glam::Mat4::IDENTITY,
            camera_position: glam::Vec4::ZERO,
            light_dir: glam::Vec4::ZERO,
            page_table_size: config.page_table_resolution(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        // Register with bindless manager
        let page_table_bindless_index = bindless_manager
            .add_page_table(resources.page_table_view, resources.page_table_sampler)?;

        let physical_memory_bindless_index = bindless_manager.add_sampled_image(
            resources.physical_cache_view,
            resources.physical_cache_sampler,
        )?;

        log::info!("VSM manager created successfully");

        Ok(Self {
            inner: Some(VsmInner {
                resources,
                compute_pipelines,
                shadow_pass,
                shadow_cull_pass,
            }),
            global_info,
            page_manager,
            clipmap_manager,
            page_table_bindless_index,
            physical_memory_bindless_index,
            current_frame: 0,
            device,
            allocator,
            enabled: true,
        })
    }

    /// Access inner GPU resources
    #[inline]
    fn inner(&self) -> Option<&VsmInner> {
        self.inner.as_ref()
    }

    /// Prepare VSM state for the current frame (CPU side)
    pub fn prepare(
        &mut self,
        _scene: &crate::renderer::Scene,
        camera_pos: glam::Vec3,
        camera_view_proj: glam::Mat4,
        light_dir: glam::Vec3,
        frame_index: u32,
    ) -> Result<()> {
        // self.current_frame is updated at the END of prepare to ensure
        // update() reads the correct buffered frame state.
        self.page_manager.begin_frame(frame_index);

        // Update clipmap centers if enabled
        if let Some(clipmap) = &mut self.clipmap_manager {
            clipmap.update(camera_pos);

            // Calculate clipmap matrix for level 0
            let radius = self
                .inner()
                .map(|i| i.resources.config().clipmap_radius)
                .unwrap_or(50.0);

            let clipmap_matrix = calculate_clipmap_view_proj(
                light_dir,
                camera_pos,
                radius,
                self.inner()
                    .map(|i| i.resources.config().virtual_resolution)
                    .unwrap_or(2048),
            );

            self.global_info.light_view_projections[0] = clipmap_matrix;
        }

        // Update global info state
        self.global_info.view_proj = camera_view_proj;
        self.global_info.inv_view_proj = camera_view_proj.inverse();
        self.global_info.camera_position = camera_pos.extend(1.0);
        self.global_info.light_dir = light_dir.extend(0.0);

        // Update VSM resources (upload to GPU)
        if let Some(inner) = &mut self.inner {
            inner.resources.update_global_info(&self.global_info)?;
        }

        self.current_frame = frame_index;
        Ok(())
    }

    /// Get VSM resources
    pub fn get_resources(&self) -> Result<&VsmResources> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(&inner.resources)
    }

    /// Get VSM shadow pass
    pub fn get_shadow_pass(&self) -> Result<&VsmShadowPass> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(&inner.shadow_pass)
    }

    /// Get VSM shadow cull pass
    pub fn get_shadow_cull_pass(&self) -> Result<&ShadowCullPass> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(&inner.shadow_cull_pass)
    }

    /// Get shadow cull indirect buffer for the current frame
    pub fn get_shadow_cull_indirect_buffer(&self, frame_index: usize) -> Result<vk::Buffer> {
        if self.current_frame as usize != frame_index {
            return Err(AshError::VulkanError(format!(
                "VSM Sync Error: Expected frame {}, got {}",
                self.current_frame, frame_index
            )));
        }
        Ok(self
            .inner
            .as_ref()
            .map(|i| {
                i.shadow_cull_pass.indirect_buffers[frame_index % i.resources.request_buffers.len()]
            })
            .unwrap_or(vk::Buffer::null()))
    }

    /// Get shadow cull count buffer for the current frame
    pub fn get_shadow_cull_count_buffer(&self, frame_index: usize) -> Result<vk::Buffer> {
        if self.current_frame as usize != frame_index {
            return Err(AshError::VulkanError(format!(
                "VSM Sync Error: Expected frame {}, got {}",
                self.current_frame, frame_index
            )));
        }
        Ok(self
            .inner
            .as_ref()
            .map(|i| {
                i.shadow_cull_pass.count_buffers[frame_index % i.resources.request_buffers.len()]
            })
            .unwrap_or(vk::Buffer::null()))
    }

    /// Dispatch VSM compute work and read back results (GPU side)
    /// Dispatch VSM compute work and read back results (GPU side)
    pub fn update(
        &mut self,
        cmd: vk::CommandBuffer,
        _frame_index: u32,
        depth_view: vk::ImageView,
        screen_width: u32,
        screen_height: u32,
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let inner = match self.inner.as_mut() {
            Some(i) => i,
            None => {
                log::error!("VSM: Attempted update after destruction");
                return Ok(());
            }
        };
        // Safety Contract: VSM request readback relies on the FrameManager's fence synchronization.
        // 1. `FrameManager::next_frame(i)` waits for the fence of frame `i - max_frames_in_flight`.
        // 2. Therefore, when we are recording frame `i`, the request buffer slot for
        //    `i % max_frames_in_flight` is guaranteed to have its GPU writes completed.
        // 3. This enables a stall-free readback of the "Analysis" results from exactly
        //    `max_frames_in_flight` frames ago.
        //
        // Any change to the frame latency or fence signaling in `FrameManager` will
        // invalidate this contract.
        let buffer_index = self.current_frame as usize % inner.resources.request_buffers.len();

        #[cfg(debug_assertions)]
        {
            // Verify that we are indeed reading from the correct slot for the current frame
            // and that the buffer index calculation remains consistent.
            let expected_index =
                self.current_frame as usize % inner.resources.request_buffers.len();
            debug_assert_eq!(buffer_index, expected_index, "VSM buffer index mismatch");
        }

        let (requests, overflow_count) = inner.resources.request_buffers[buffer_index]
            .read_requests(&self.allocator)
            .map_err(|e| {
                crate::AshError::LockPoisoned(format!(
                    "VSM Request readback failed or lock poisoned: {e:?}"
                ))
            })?;

        if overflow_count > 0 {
            log::warn!("VSM: Request overflow! {overflow_count} requests dropped.");
        }

        // 1. Process Requests via PageManager (CPU side)
        let new_allocations = self.page_manager.process_requests(&requests);

        // 2. Clear the buffer for the current frame's Analysis Pass
        unsafe {
            self.device.cmd_fill_buffer(
                cmd,
                inner.resources.request_buffers[buffer_index].buffer,
                0,
                8, // Clear both count and overflow_count
                0,
            );

            let reset_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(inner.resources.request_buffers[buffer_index].buffer)
                .offset(0)
                .size(8);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[reset_barrier],
                &[],
            );
        }

        // 3. Update Analysis Descriptors (Bind Scene Depth and the buffer we just cleared)
        inner
            .resources
            .update_analysis_descriptors(depth_view, self.current_frame)?;

        unsafe {
            // 2. Bind Descriptor Set (Standardized Layout)
            let descriptor_sets = [inner.resources.descriptor_set];
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                inner.compute_pipelines.analyze.layout(),
                0,
                &descriptor_sets,
                &[],
            );

            // 3. Dispatch Analysis Pass (analyze.glsl)
            // This samples scene depth and generates page requests for the NEXT frame.
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                inner.compute_pipelines.analyze.handle(),
            );
            let groups_x = screen_width.div_ceil(8);
            let groups_y = screen_height.div_ceil(8);
            self.device.cmd_dispatch(cmd, groups_x, groups_y, 1);

            // 4. Barrier: Ensure Request Buffer updates are visible to CPU or next dispatch
            let buffer_index = self.current_frame as usize % inner.resources.request_buffers.len();
            let request_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ)
                .buffer(inner.resources.request_buffers[buffer_index].buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[request_barrier],
                &[],
            );
        }

        if !new_allocations.is_empty() {
            // 4. Upload to AllocationBuffer
            if let Err(e) = inner.resources.upload_allocations(&new_allocations) {
                log::error!("Failed to upload VSM allocations: {e:?}");
            }

            let alloc_count = new_allocations.len() as u32;
            log::info!("VSM: Allocating {alloc_count} pages");

            unsafe {
                // 5. Dispatch Allocator (Update Page Table)
                let group_count = alloc_count.div_ceil(64);
                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    inner.compute_pipelines.allocate.handle(),
                );
                self.device.cmd_dispatch(cmd, group_count, 1, 1);

                // 6. Barrier: Ensure Page Table update finishes
                let table_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(inner.resources.page_table)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: inner.resources.config().clipmap_levels.max(1),
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[table_barrier],
                );

                // 7. Dispatch Clear (Clear Physical Memory)
                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    inner.compute_pipelines.clear.handle(),
                );
                // 16x16 local size -> 8x8 groups per 128x128 page
                self.device.cmd_dispatch(cmd, 8, 8, alloc_count);

                // 8. Barrier: Physical cache clear finishes
                let cache_barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(inner.resources.physical_cache)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[cache_barrier],
                );
            }
        }

        Ok(())
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
    pub fn physical_cache_view(&self) -> Result<vk::ImageView> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.resources.physical_cache_view)
    }

    /// Get physical cache sampler
    pub fn physical_cache_sampler(&self) -> Result<vk::Sampler> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.resources.physical_cache_sampler)
    }

    /// Get page table image view
    pub fn page_table_view(&self) -> Result<vk::ImageView> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.resources.page_table_view)
    }

    /// Get page table sampler
    pub fn page_table_sampler(&self) -> Result<vk::Sampler> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.resources.page_table_sampler)
    }

    /// Get shadow render pass
    pub fn shadow_render_pass(&self) -> Result<vk::RenderPass> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.shadow_pass.render_pass)
    }

    /// Get shadow framebuffer
    pub fn shadow_framebuffer(&self) -> Result<vk::Framebuffer> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.shadow_pass.framebuffer)
    }

    /// Get configuration
    pub fn config(&self) -> Result<&VsmConfig> {
        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;
        Ok(inner.resources.config())
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

    /// Render shadows using the new page rendering path
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    /// # Callback Contract
    /// The `scene_draw_fn` is expected to record draw commands. The viewport and scissor
    /// are PRE-CONFIGURED to match the target physical atlas page. Do not override them inside the closure.
    pub unsafe fn render_shadows<F>(
        &self,
        frame_index: usize,
        args: VsmShadowArgs,
        scene_draw_fn: F,
    ) -> Result<()>
    where
        F: FnMut(vk::CommandBuffer),
    {
        if self.current_frame as usize != frame_index {
            return Err(AshError::VulkanError(format!(
                "VSM Sync Error: Expected frame {}, got {}",
                self.current_frame, frame_index
            )));
        }
        let pages = self.page_manager.get_pages_to_render(args.light_view_proj);
        if pages.is_empty() {
            return Ok(());
        }

        let inner = self.inner().ok_or_else(|| {
            crate::AshError::VulkanError("VSM is disabled or uninitialized".into())
        })?;

        // 1. Dispatch Shadow Culling for the light view
        // We use clipmap level 0 for the generic shadow culling pass for now.
        inner.shadow_cull_pass.cull_shadows(
            args.cmd,
            crate::renderer::passes::ShadowCullInfo {
                frame_index: self.current_frame as usize % inner.resources.request_buffers.len(),
                view_proj: args.light_view_proj,
                object_count: args.object_count,
                base_index: 0,
                clipmap_level: 0,
                object_buffer_ptr: args.object_addr,
            },
        );

        // Barrier to ensure culling results are visible to indirect draw
        let cull_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
            .buffer(
                inner.shadow_cull_pass.indirect_buffers
                    [self.current_frame as usize % inner.resources.request_buffers.len()],
            )
            .offset(0)
            .size(vk::WHOLE_SIZE);

        self.device.cmd_pipeline_barrier(
            args.cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::DRAW_INDIRECT,
            vk::DependencyFlags::empty(),
            &[],
            &[cull_barrier],
            &[],
        );

        // 2. Render pages
        inner.shadow_pass.render_pages(
            args.cmd,
            super::shadow_pass::ShadowPageRenderInfo {
                resources: &inner.resources,
                bindless_descriptor_set: args.bindless_set,
                vertex_addr: args.vertex_addr,
                index_addr: args.index_addr,
                pages: &pages,
            },
            scene_draw_fn,
        );

        Ok(())
    }

    /// Get shadow pipeline layout
    pub fn shadow_pipeline_layout(&self) -> Option<vk::PipelineLayout> {
        self.inner
            .as_ref()
            .and_then(|i| i.shadow_pass.pipeline_layout())
    }

    /// Get shadow pipeline (if created)
    pub fn shadow_pipeline(&self) -> Option<vk::Pipeline> {
        self.inner.as_ref().and_then(|i| i.shadow_pass.pipeline())
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            log::debug!("Destroying VSM manager");

            inner.shadow_pass.destroy(&self.allocator);
            inner.shadow_cull_pass.destroy(&self.allocator);
            inner.compute_pipelines.destroy();

            inner.resources.destroy();

            log::debug!("VSM manager destroyed");
        }
    }
}

impl Drop for VsmManager {
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
        clipmap_radius: 50.0,
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
        clipmap_radius: 40.0,
    }
}
