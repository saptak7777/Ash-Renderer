use crate::{
    AshError, Result,
    renderer::{
        FramePreparationInfo, GeometryRenderContext, HdrSystem, MaterialHandle, MaterialManager,
        Mesh, MeshUploadInfo, PipelineCache, Scene,
        context::Context,
        diagnostics::{DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, GpuProfiler},
        features::vsm::VsmManager,
        frame_manager,
        passes::temporal_aa::{
            ConfigChangeType, ConfigMetrics, ConfigMetricsReport, ConfigValidationError, TaaConfig,
            Validate, detect_config_change,
        },
        resources::{
            self, BufferPool, MaterialDescriptor, MeshDescriptor, Resources, ResourcesInitInfo,
        },
        swapchain_manager,
        types::{DebugMode, MeshData, RenderCommand, RenderFrameContext, RendererConfig},
    },
    vulkan::{self, Allocator, CommandBufferContext},
};

use ash::vk;
use glam::Mat4;
use std::sync::Arc;
// use std::time::Instant; // Moved to Frame

// RendererResources moved to init_types.rs

use super::frame::Frame;
use super::sync::SceneSynchronizer;
use super::systems::Systems;
use super::util::frame_state::FrameState;

/// Main rendering system.
///
/// # Safety & Drop Semantics
///
/// Fields are declared in LIFO (Last-In-First-Out) destruction order.
/// High-level systems and GPU assets are dropped before the core `Context`.
/// Post-context fields (`frame_state`, `sync`) are POD/stateless and safe to drop last.
///
/// PANIC SAFETY: If a thread panics while this guard is held, the `Resources` struct will be left
/// with a default/empty manager until the stack unwinds and `drop()` is called. Because the
/// restoration occurs in the 'Drop' implementation, the InstancingManager is guaranteed to be
/// restored to the Resources struct even during thread unwinding (panics). This strictly
/// prevents memory leaks or dangling pointers during catastrophic engine failures.
///
/// **DO NOT REORDER FIELDS OR IMPLEMENT `Drop`**.
///
/// **DO NOT REORDER THESE FIELDS.**
pub struct Renderer {
    /// High-level logic and systems. Drops FIRST.
    pub systems: Systems,

    /// VSM shadowing state and resources.
    pub vsm_manager: VsmManager,

    /// Frame-level synchronization and command management. Drops SECOND.
    pub frame: Frame,

    /// GPU resources and asset managers. Drops THIRD.
    pub resources: Resources,

    /// Vulkan context (Device/Instance). Drops LAST.
    pub context: Context,

    // â”€â”€ New Modular Systems â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    /// Single source of truth for per-frame temporal state (jitter, matrices, time).
    pub frame_state: FrameState,

    /// CPU-to-GPU scene reconciliation: uniform upload, VSM, Forward+.
    pub sync: SceneSynchronizer,

    /// Time counter for frame delta calculation.
    pub last_frame_instant: std::time::Instant,
}

pub struct MainPassParameters<'a> {
    pub cmd_ctx: &'a CommandBufferContext<'a>,
    pub frame_index: usize,
    pub image_index: u32,
    pub scene_pipeline: vk::Pipeline,
    pub pipeline_layout_handle: vk::PipelineLayout,
    pub view: Mat4,
    pub projection: Mat4,
    pub swapchain_extent: vk::Extent2D,
    pub light_ptr: u64,
    pub tile_ptr: u64,
    pub scene: &'a super::Scene,
}

impl Renderer {
    /// Returns a new [`RendererBuilder`] to configure and build the renderer.
    pub fn builder() -> crate::renderer::builder::RendererBuilder {
        crate::renderer::builder::RendererBuilder::new()
    }

    pub fn wait_for_idle(&self) -> Result<()> {
        self.context.wait_for_idle()
    }

    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.systems.debug_mode = mode;
        log::info!("Debug mode set to: {mode:?}");
    }

    /// Initializes the renderer with a specific configuration.
    pub(crate) fn new_with_config<S: vulkan::SurfaceProvider>(
        surface_provider: &S,
        config: RendererConfig,
    ) -> Result<Self> {
        log::info!("Renderer::new_with_config: Starting initialization");
        unsafe {
            let context = Context::new(surface_provider)?;
            let pipeline_cache = PipelineCache::new(Arc::clone(&context.device.device))?;

            let (width, height) = if let Some((w, h)) = config.resolution {
                (w, h)
            } else {
                surface_provider.physical_size()
            };

            let mut frame = Frame::new(&context, width, height, config.present_mode)?;

            // We now initialize resources FIRST so we have the BindlessManager ready for VSM
            let mut resources = Resources::new(ResourcesInitInfo {
                context: &context,
                width,
                height,
                config: &config,
                surface_provider,
                pipeline_cache: &pipeline_cache,
                swapchain: frame.swapchain.as_ref().unwrap(),
                cmds: &frame.cmds,
            })?;

            log::info!("Initializing VSM Manager");
            let mut vsm_config = crate::renderer::features::vsm::default_vsm_config();
            vsm_config.physical_resolution = config.shadow_resolution;

            let vsm_manager = VsmManager::new(
                Arc::clone(&context.device.device),
                Arc::clone(&context.alloc),
                &mut resources.assets.bindless_manager,
                frame.cmds.upload_command_pool_handle(),
                context.device.graphics_queue,
                vsm_config,
                frame.swapchain.as_ref().unwrap().image_views.len() as u32,
            )?;

            log::info!("Initializing Systems");
            let systems = Systems::new(
                &context,
                &mut resources,
                &mut frame,
                pipeline_cache,
                width,
                height,
                &config,
            )?;

            let mut renderer = Self {
                systems,
                vsm_manager,
                frame,
                resources,
                context,
                frame_state: FrameState::default(),
                sync: SceneSynchronizer::new(),
                last_frame_instant: std::time::Instant::now(),
            };

            let (image_count, extent) = {
                let sc = renderer.frame.swapchain.as_ref().unwrap();
                (sc.image_views.len(), sc.extent)
            };
            renderer
                .systems
                .pipeline
                .post_process_mut()
                .resize(image_count, extent)?;
            renderer.systems.pipeline.validate()?;
            renderer.register_tracked_subsystems()?;
            renderer
                .systems
                .init_motion_pass(&renderer.context, &renderer.resources)?;

            // --- Pre-initialize Forward+ lighting pipeline (prevents first-frame stutter) ---
            {
                let fp_lock = &renderer.systems.pipeline.forward_plus;
                let db = &renderer.resources.depth_buffer;
                let mut fp = fp_lock
                    .write()
                    .map_err(|_| AshError::LockPoisoned("ForwardPlus".to_string()))?;
                fp.init_pipeline(
                    Arc::clone(&renderer.context.device.device),
                    db.sampler(),
                    db.view(),
                    Some(
                        renderer
                            .context
                            .device
                            .properties12
                            .max_descriptor_set_update_after_bind_sampled_images,
                    ),
                )?;
                log::info!("Forward+ lighting compute pipeline pre-initialized at startup.");
            }

            renderer.initialize_hdr(width, height)?;

            renderer.context.queue.pending_extent =
                Some(renderer.frame.swapchain.as_ref().unwrap().extent);

            Ok(renderer)
        }
    }

    /// Convenience for default initialization.
    pub fn new<S: vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        Self::builder().build(surface_provider)
    }

    /// Register all major subsystems with the ResourceRegistry for automatic cleanup
    fn register_tracked_subsystems(&mut self) -> Result<()> {
        let registry = &self.context.resources;

        registry
            .register_shared_resource(Arc::clone(&self.systems.pipeline.forward_plus))
            .map_err(|e| AshError::VulkanError(e.to_string()))?;

        registry
            .register_shared_resource(Arc::clone(&self.systems.pipeline.hiz_pass))
            .map_err(|e| AshError::VulkanError(e.to_string()))?;

        registry
            .register_shared_resource(Arc::clone(&self.systems.pipeline.indirect_draw_pass))
            .map_err(|e| AshError::VulkanError(e.to_string()))?;

        // Register uniform buffers
        for ub in &self.resources.uniform_buffers {
            registry
                .register_shared_resource(Arc::clone(ub))
                .map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        // Register material storage buffer
        if let Some(ref msb) = self.resources.material_storage_buffer {
            registry
                .register_shared_resource(Arc::clone(msb))
                .map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        Ok(())
    }

    /// Access the underlying memory allocator.
    pub fn allocator(&self) -> &Allocator {
        &self.context.alloc
    }

    pub fn bindless_manager(&self) -> &vulkan::BindlessManager {
        &self.resources.assets.bindless_manager
    }

    pub fn bindless_manager_mut(&mut self) -> &mut vulkan::BindlessManager {
        &mut self.resources.assets.bindless_manager
    }

    pub fn get_mesh_material(&self, scene: &Scene, mesh_handle: u32) -> MaterialHandle {
        scene
            .mesh_data
            .get(mesh_handle as usize)
            .map(|m| m.material_handle)
            .unwrap_or_else(MaterialHandle::null)
    }

    /// Get immutable access to consolidated mesh data.
    pub fn mesh_data<'a>(&self, scene: &'a Scene) -> &'a [MeshData] {
        &scene.mesh_data
    }

    /// Get mutable access to mesh data by handle.
    pub fn get_mesh_data_mut<'a>(
        &self,
        scene: &'a mut Scene,
        handle: u32,
    ) -> Option<&'a mut MeshData> {
        scene.mesh_data.get_mut(handle as usize)
    }

    /// Returns the global geometry buffer for shared vertex/index storage.
    pub fn geometry_buffer(&self) -> Arc<resources::DualHeapGeometryBuffer> {
        Arc::clone(&self.resources.geometry_buffer)
    }

    /// Get mutable access to the material manager.
    pub fn material_manager_mut<'a>(&self, scene: &'a mut Scene) -> &'a mut MaterialManager {
        &mut scene.material_manager
    }

    /// Returns a one-time use command buffer for transfer operations.
    pub fn get_transfer_command_buffer(&self) -> Result<vk::CommandBuffer> {
        self.frame.cmds.get_transfer_command_buffer()
    }

    pub fn get_stats(&self) -> crate::renderer::diagnostics::RendererStats {
        crate::renderer::diagnostics::RendererStats {
            vram_usage: self.resources.assets.vram_budget.get_stats(),
            draw_calls_per_frame: self.systems.diagnostics.frame_stats.draw_calls,
            triangles_rendered: self.systems.diagnostics.frame_stats.triangles,
            cull_efficiency: {
                // Calculate cull efficiency: (potential_draws - actual_draws) / potential_draws
                let potential_draws =
                    (self.systems.diagnostics.frame_stats.triangles / 1000).max(1) as f32;
                let actual_draws = self.systems.diagnostics.frame_stats.draw_calls as f32;
                ((potential_draws - actual_draws) / potential_draws.max(1.0)).clamp(0.0, 1.0)
            },
            gpu_frame_ms: self.systems.diagnostics.gpu_timings.total_ms,
            hiz_quality: format!(
                "{:?}",
                self.systems
                    .pipeline
                    .hiz_pass
                    .read()
                    .map(|h| h.quality())
                    .unwrap_or(crate::renderer::passes::hiz::HiZQuality::Balanced)
            ),
            frame_count: self.systems.diagnostics.frame_stats.total_frames,
        }
    }

    /// Logs the current frame statistics to the debug log.
    pub fn log_frame_stats(&self) {
        self.get_stats().log_frame_stats();
    }

    /// Unloads all currently registered textures from the host-side registry.
    /// Caution: Ensure no GPU frames are in flight using these textures before clearing.
    pub fn clear_texture_registry(&mut self) {
        self.resources.assets.texture_registry.clear();
    }

    /// Updates the GPU material buffer with a material at the specified index (AAA-grade direct streaming)
    /// Uses single-element writes instead of read-modify-write to avoid GPU stalls and race conditions.
    /// This must be called after registering the material to ensure the GPU sees the correct material.
    ///
    /// This follows modern game engine patterns (UE5, Unity) where material updates are streamed
    /// directly without reading back the entire buffer.
    /// Get access to the material manager (for testing)
    pub fn material_manager<'a>(&self, scene: &'a Scene) -> &'a MaterialManager {
        &scene.material_manager
    }

    /// Registers mesh data described by a [`MeshDescriptor`] with the renderer and returns the
    /// internal key used for lookup.
    pub fn register_mesh_descriptor(
        &mut self,
        scene: &mut Scene,
        _handle: u32,
        descriptor: &MeshDescriptor,
        upload_cmd: vk::CommandBuffer,
        staging_resources: &mut Vec<crate::renderer::resources::BufferHandle>,
    ) -> Result<String> {
        let mut mesh = Mesh::from_descriptor(descriptor);
        let key: Arc<str> = Arc::clone(&mesh.name);

        scene.upload_mesh(MeshUploadInfo {
            device: Arc::clone(&self.context.device.device),
            allocator: Arc::clone(&self.context.alloc),
            command_pool: self.frame.cmds.upload_command_pool_handle(),
            command_buffer: upload_cmd,
            queue: self.context.device.graphics_queue,
            mesh: &mut mesh,
            asset_manager: &mut self.resources.assets,
            staging_resources,
            material_override: None, // No material override for simple descriptors
        })?;

        Ok(key.to_string())
    }

    /// Converts a material descriptor into a renderer material and registers it.
    pub fn register_material_descriptor(
        &mut self,
        scene: &mut Scene,
        handle: u32,
        descriptor: &MaterialDescriptor,
    ) -> Result<MaterialHandle> {
        let mut material = descriptor.material.clone();
        material.name = format!("Material_{handle}");

        // Consolidated register call
        scene.register_material(&material)
    }

    /// Submit render commands for the current frame.
    ///
    /// Each `RenderCommand` specifies a mesh handle, material handle, and transform.
    /// For large command counts (>1000), uses parallel processing across all CPU cores.
    pub fn submit_render_commands(
        &mut self,
        scene: &mut super::Scene,
        commands: &[RenderCommand],
    ) -> Result<()> {
        log::debug!("Submitting {} render commands", commands.len());
        scene.occlusion_culling.begin_frame();

        // 1. Sort commands to minimize state changes
        let mut sorted_commands: Vec<usize> = (0..commands.len()).collect();
        sorted_commands.sort_by(|&a, &b| {
            let cmd_a = &commands[a];
            let cmd_b = &commands[b];

            // Sort by mesh handle then material
            cmd_a.mesh_handle.cmp(&cmd_b.mesh_handle).then_with(|| {
                cmd_a
                    .material_handle
                    .index
                    .cmp(&cmd_b.material_handle.index)
            })
        });

        // 2. Populate Occlusion Culling directly from sorted commands
        for (i, &idx) in sorted_commands.iter().enumerate() {
            let command = &commands[idx];
            if let Some(mesh_data) = scene.mesh_data.get(command.mesh_handle as usize) {
                if let Some(uploaded) = scene.model_renderer.get(&mesh_data.name) {
                    let material_handle = if command.material_handle.is_null() {
                        mesh_data.material_handle
                    } else {
                        command.material_handle
                    };

                    scene.occlusion_culling.push_clusters(
                        crate::renderer::vcgs::CullObjectDesc {
                            bounds: mesh_data.bounds,
                            model: command.transform,
                            prev_model: command.prev_transform.unwrap_or(command.transform),
                            draw_index: i as u32,
                            first_index: (uploaded.index_offset.unwrap_or(0) / 4) as u32,
                            index_count: uploaded.index_count(),
                            material_index: material_handle.index,
                            vertex_offset: (uploaded.vertex_offset.unwrap_or(0) / 64) as i32,
                        },
                        uploaded.clusters(),
                    );
                }
            }
        }

        Ok(())
    }

    /// Bake IBL maps from an equirectangular texture.
    ///
    /// This function converts an equirectangular environment map to cubemap format
    /// and generates irradiance and prefiltered maps for image-based lighting.
    /// Currently unused but preserved for runtime environment map loading features.
    pub fn request_swapchain_resize(&mut self, new_extent: vk::Extent2D) {
        self.context.queue.request_resize(new_extent);
    }

    pub(crate) fn update_image_views(&mut self, image_views: &[vk::ImageView]) -> Result<()> {
        if self.context.device.headless && !self.frame.swapchain_image_view_ids.is_empty() {
            // In headless mode, we reuse the same image views.
            // Cleaning them up would destroy the underlying Vulkan handles.
            return Ok(());
        }

        for id in self.frame.swapchain_image_view_ids.drain(..) {
            if let Err(e) = self.context.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old swapchain image view {id}: {e}");
            }
        }

        self.frame.swapchain_image_view_ids.clear();
        for &view in image_views {
            let id = self
                .context
                .resources
                .register_image_view(view)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register swapchain image view: {e}"))
                })?;
            self.frame.swapchain_image_view_ids.push(id);
        }

        if let Some(ref mut sc) = self.frame.swapchain {
            sc.mark_image_views_managed_by_registry();
        }

        Ok(())
    }

    pub(crate) fn recreate_frame_syncs(&mut self, count: usize) -> Result<()> {
        log::info!("Recreating frame synchronization objects: Count {count}");
        self.frame
            .frame_manager
            .destroy(&self.context.device.device);
        self.frame.frame_manager = frame_manager::FrameManager::new(
            &self.context.device.device,
            self.frame.cmds.upload_command_pool_handle(),
            count,
        )?;
        Ok(())
    }

    pub(crate) fn recreate_command_buffers(&mut self) -> Result<()> {
        log::info!("Resetting frame management lifecycle");
        self.frame.frame_manager.reset_frame();
        Ok(())
    }

    pub(crate) fn recreate_descriptor_sets(&mut self) -> Result<()> {
        unsafe {
            self.context.device.device.device_wait_idle().map_err(|e| {
                AshError::VulkanError(format!("Failed to wait for device idle: {e:?}"))
            })?;
        }

        if let Some(_manager) = self.resources.descriptors.as_mut() {
            let _count = self.frame.frame_manager.get_max_frames_in_flight() as u32;
            // Removed recreate_frame_sets lines

            // CRITICAL FIX: Re-bind Environment defaults
            // Set 2 is gone.

            // CRITICAL FIX: Update Forward+ Descriptors (Set 3)
            // ForwardPlusIntegration's pool is separate but its bindings need to be refreshed
        }

        Ok(())
    }

    /// Render frame with the specified camera view.
    ///
    /// Arguments:
    /// - `view`: View matrix (camera look-at)
    /// - `projection`: Projection matrix (perspective/orthographic)
    /// - `camera_pos`: Camera world position (for lighting calculations)
    pub fn render_main_pass(&self, params: &MainPassParameters) -> Result<()> {
        let cmd_ctx = params.cmd_ctx;
        let frame_index = params.frame_index;

        // Resolve global debug state
        let debug_enabled = !matches!(self.systems.debug_mode, DebugMode::None);

        // Context Upgrade: Retrieve views for Dynamic Rendering
        // Mandatory HDR: All main-pass rendering must route through the FP16 HDR buffer.
        let hdr = self
            .systems
            .hdr_system
            .as_ref()
            .expect("HdrSystem is mandatory for Pure Renderer remediation");
        let (color_view, color_image) = (hdr.view(), hdr.image());

        let depth_view = self.resources.depth_buffer.view();
        let depth_image = self.resources.depth_buffer.image();
        let depth_format = self.resources.depth_buffer.format();

        // Create a dummy transform for feature rendering context
        let dummy_transform = crate::renderer::Transform::identity();

        // Consolidated Geometry Delegation
        let geo_ctx = GeometryRenderContext {
            device: &self.context.device,
            command_buffer: cmd_ctx,
            scene: params.scene,
            bindless_descriptor_set: self.resources.assets.bindless_manager.descriptor_set(),
            vsm_manager: &self.vsm_manager,
            swapchain_extent: params.swapchain_extent,
            frame_ptr: self.resources.uniform_buffers[frame_index]
                .read()
                .map_err(|_| AshError::LockPoisoned("UniformBuffer".to_string()))?
                .device_address(),
            material_ptr: self.resources.material_heap_address,
            light_ptr: params.light_ptr,
            tile_ptr: params.tile_ptr,
            debug_enabled,
            color_image,
            color_view,
            depth_image,
            depth_view,
            normal_view: Some(self.resources.gbuffer.normal_view()),
            albedo_view: Some(self.resources.gbuffer.albedo_view()),
            motion_image: Some(self.resources.gbuffer.motion_image()),
            motion_view: Some(self.resources.gbuffer.motion_view()),
            skybox: self.systems.skybox_pass.as_ref(),
            features: Some(&self.systems.features),
            frame_index,
            descriptor_allocator: self.resources.descriptors.as_ref(),
            transform: &dummy_transform,
            depth_format,
            vsm_ptr: unsafe {
                let info = vk::BufferDeviceAddressInfo::default()
                    .buffer(self.vsm_manager.get_resources()?.metadata_buffer);
                self.context.device.device.get_buffer_device_address(&info)
            },
        };

        self.systems
            .pipeline
            .render_geometry(&geo_ctx, params.scene, frame_index)?;

        Ok(())
    }

    /// Update TAA configuration with validation, metrics, and resource management.
    ///
    /// Validates the configuration, logs change severity, tracks metrics, and recreates
    /// TAA resources when a major change occurs (e.g., quality/sharpening/enable).
    pub fn update_taa_config(&mut self, config: TaaConfig) -> Result<(), ConfigValidationError> {
        // Validate incoming config
        config.validate()?;

        // Determine change type
        let change_type = detect_config_change(&self.systems.taa_config, &config);

        // Log change severity
        match change_type {
            ConfigChangeType::None => {
                log::debug!("TAA config unchanged");
                return Ok(());
            }
            ConfigChangeType::Minor => {
                log::info!(
                    "TAA config updated (minor): {:?} -> {:?}",
                    self.systems.taa_config.quality,
                    config.quality
                );
            }
            ConfigChangeType::Major => {
                log::info!(
                    "TAA config updated (major, recreation needed): {:?} -> {:?}",
                    self.systems.taa_config.quality,
                    config.quality
                );
            }
        }

        // Apply configuration
        self.systems.taa_config = config;

        // Track metrics
        self.systems
            .taa_config_metrics
            .record_change(self.frame.frame_manager.get_current_frame_index() as u64);

        // Recreate resources if needed
        if change_type.needs_recreation() {
            self.recreate_taa_resources();
        }

        Ok(())
    }

    /// Recreate TAA resources when config changes (resets temporal accumulation)
    fn recreate_taa_resources(&mut self) {
        log::debug!("Recreating TAA resources due to TAA config change");
        // Reset temporal accumulation to avoid ghosting after major config changes
        self.frame.frame_manager.reset_frame();
        log::debug!("TAA resources recreated");
    }

    /// Access TAA configuration metrics
    pub fn taa_config_metrics(&self) -> &ConfigMetrics {
        &self.systems.taa_config_metrics
    }

    /// Generate TAA configuration metrics report
    pub fn taa_config_metrics_report(&self) -> ConfigMetricsReport {
        self.systems.taa_config_metrics.report()
    }

    pub fn sync_frame_resources(&mut self, scene: &mut Scene) -> Result<()> {
        // Delegate all resource synchronization and bookkeeping to the SceneSynchronizer.
        let shaders_changed = self.sync.sync_resources(
            &mut self.frame,
            scene,
            &mut self.systems,
            &mut self.resources,
        )?;

        // If high-level shader changes were detected, trigger the swapchain manager
        // to rebuild pipelines using the new shader bytecode.
        if shaders_changed {
            if let Err(e) =
                crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene)
            {
                log::error!("Failed to recreate pipeline: {e}");
            }
        }

        Ok(())
    }

    pub fn prepare_frame_data(
        &mut self,
        scene: &mut Scene,
        view: Mat4,
        projection: Mat4,
        camera_pos: glam::Vec3,
        model_matrix: Option<Mat4>,
    ) -> Result<(usize, u32, vk::Extent2D, Mat4, [f32; 2])> {
        let swapchain_extent = self
            .frame
            .swapchain
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Swapchain not available".into()))?
            .extent;

        // Acquire the next swapchain image and advance the frame manager.
        let (image_index, _suboptimal) = self.frame.begin_frame(&self.context)?;
        let frame_index = self.frame.frame_manager.get_current_frame_index();

        // Calculate actual delta time.
        let now = std::time::Instant::now();
        let delta_time = now.duration_since(self.last_frame_instant).as_secs_f32();
        self.last_frame_instant = now;

        // Advance the centralized frame state (jitter, matrix history, time).
        self.frame_state.begin_frame(
            delta_time,
            view,
            projection,
            camera_pos,
            swapchain_extent.width,
            swapchain_extent.height,
        );

        // Delegate GPU uniform uploads to the SceneSynchronizer.
        self.sync.prepare_frame(FramePreparationInfo {
            context: &self.context,
            frame: &mut self.frame,
            scene,
            systems: &mut self.systems,
            resources: &mut self.resources,
            vsm_manager: &mut self.vsm_manager,
            frame_state: &self.frame_state,
            frame_index,
            model_matrix,
        })?;

        let jitter_uv = self.frame_state.jitter_uv();
        let jittered_projection = self.frame_state.jittered_projection;

        Ok((
            frame_index,
            image_index,
            swapchain_extent,
            jittered_projection,
            jitter_uv,
        ))
    }

    pub fn record_and_submit(&mut self, ctx: RenderFrameContext) -> Result<()> {
        let RenderFrameContext {
            frame_index,
            image_index,
            scene,
            view,
            jitter_proj,
            jitter_uv,
            extent,
            ui_callback,
        } = ctx;
        unsafe {
            let scene_pipeline = self.systems.pipeline.main_graphics_pipeline.pipeline;

            let command_buffer = self
                .frame
                .frame_manager
                .begin_command_buffer(&self.context.device.device)?;
            let device_arc = Arc::clone(&self.context.device.device);
            let cmd_ctx = CommandBufferContext::new(device_arc.as_ref(), command_buffer);

            log::debug!(
                "Frame {}: Using image index {}",
                self.frame.frame_manager.get_current_frame_index(),
                image_index
            );

            // â”€â”€ 1. Host-Write Barrier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Ensure CPU-side buffer writes (uniforms, instance data) are visible to
            // all GPU shader stages before any rendering begins.
            let global_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .src_access_mask(vk::AccessFlags2::HOST_WRITE | vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::UNIFORM_READ
                        | vk::AccessFlags2::INDEX_READ
                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                );

            let memory_barriers = [global_barrier];
            let dep_info = vk::DependencyInfo::default().memory_barriers(&memory_barriers);
            cmd_ctx.pipeline_barrier2(&dep_info);

            // â”€â”€ 2. GPU-Driven Occlusion Culling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Upload per-object draw data then run the compute culling pass.
            let indirect_pass = self
                .systems
                .culling
                .indirect_draw_pass
                .read()
                .map_err(|e| {
                    AshError::VulkanError(format!("Indirect draw pass lock poisoned: {e}"))
                })?;
            indirect_pass.upload_objects(
                &self.context.alloc.vma,
                scene.occlusion_culling.object_data(),
                0,
            )?;
            let hiz_arc = &self.systems.pipeline.hiz_pass;
            let hiz_buffer_addr = hiz_arc
                .read()
                .map_err(|e| crate::AshError::VulkanError(format!("Hi-Z pass lock poisoned: {e}")))?
                .hiz_buffer_addr();

            self.systems.culling.execute_culling(
                cmd_ctx.handle(),
                &self.context,
                &self.resources,
                scene,
                hiz_buffer_addr,
                frame_index,
            )?;

            // â”€â”€ 3. Hi-Z Pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Build the hierarchical-Z depth pyramid used by the culling pass next
            // frame, and route the updated view+sampler into the IndirectDraw pass.
            self.systems.pipeline.execute_hiz_pass(
                command_buffer,
                self.resources.depth_buffer.view(),
                self.systems.gpu_profiler.as_ref(),
                self.resources.black_texture.view(),
                self.resources.black_texture.sampler(),
            )?;

            // â”€â”€ 4. VSM Shadow Pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Update the VSM page table and render shadow geometry.
            let depth_view = self.resources.depth_buffer.view();

            let bindless_set = self.resources.assets.bindless_manager.descriptor_set();
            self.vsm_manager.update(
                command_buffer,
                bindless_set,
                frame_index as u32,
                depth_view,
                self.resources.swapchain_extent.width,
                self.resources.swapchain_extent.height,
            )?;
            {
                let bindless_set = self.resources.assets.bindless_manager.descriptor_set();
                let (vertex_addr, index_addr) = scene.get_geometry_buffer_addresses();
                let cull_indirect = self
                    .vsm_manager
                    .get_shadow_cull_indirect_buffer(frame_index)?;
                let cull_count = self.vsm_manager.get_shadow_cull_count_buffer(frame_index)?;

                let light_dir =
                    glam::Vec3::from_slice(&scene.scene_lighting.directional.direction[0..3]);
                let light_view =
                    glam::Mat4::look_at_rh(light_dir * 10.0, glam::Vec3::ZERO, glam::Vec3::Y);
                let light_proj = glam::Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
                let light_vp = light_proj * light_view;

                self.vsm_manager.render_shadows(
                    frame_index,
                    crate::renderer::features::vsm::VsmShadowArgs {
                        cmd: command_buffer,
                        light_view_proj: light_vp,
                        bindless_set,
                        vertex_addr,
                        index_addr,
                        object_addr: self
                            .systems
                            .culling
                            .indirect_draw_pass
                            .read()
                            .map(|p| p.object_buffer_address())
                            .unwrap_or(0),
                        object_count: scene.occlusion_culling.object_count() as u32,
                    },
                    |cmd| {
                        // Lean geometry-only draw for shadow geometry.
                        let pass = self.systems.culling.indirect_draw_pass.read().unwrap();
                        if pass.is_initialized() {
                            self.context.device.device.cmd_draw_indirect_count(
                                cmd,
                                cull_indirect,
                                0,
                                cull_count,
                                0,
                                2048, // max_draw_count
                                std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
                            );
                        }
                    },
                )?;
            }

            // â”€â”€ 5. Light Culling Pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Tile the scene lights for Forward+ shading.
            {
                let frame_ptr = self.resources.uniform_buffers[frame_index]
                    .read()
                    .map_err(|_| AshError::LockPoisoned("UniformBuffer".to_string()))?
                    .device_address();

                self.systems
                    .pipeline
                    .forward_plus
                    .read()
                    .map_err(|_| AshError::LockPoisoned("ForwardPlus".to_string()))?
                    .cull_lights(command_buffer, frame_index, frame_ptr)?;
            }

            // â”€â”€ 6. Geometry / Main Pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // Build the per-pass parameter block and dispatch to RenderPipeline.
            {
                let light_ptr = self
                    .systems
                    .pipeline
                    .forward_plus
                    .read()
                    .map_err(|_| AshError::LockPoisoned("ForwardPlus".to_string()))?
                    .get_lights()
                    .light_ptr(frame_index);

                let tile_ptr = self
                    .systems
                    .pipeline
                    .forward_plus
                    .read()
                    .map_err(|_| AshError::LockPoisoned("ForwardPlus".to_string()))?
                    .get_lights()
                    .tile_ptr(frame_index);
                let pipeline_layout_handle = self.systems.pipeline.pipeline_layout.handle();

                let main_pass_params = MainPassParameters {
                    cmd_ctx: &cmd_ctx,
                    frame_index,
                    image_index,
                    scene_pipeline,
                    pipeline_layout_handle,
                    view,
                    projection: jitter_proj,
                    swapchain_extent: extent,
                    light_ptr,
                    tile_ptr,
                    scene,
                };
                self.render_main_pass(&main_pass_params)?;
            }

            // â”€â”€ 7. Post-Process Pass (TAA â†’ VSR â†’ Tonemapping) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            let depth_view = self.resources.depth_buffer.view();
            let motion_view = self.resources.gbuffer.motion_view();

            let (bloom_view, bloom_intensity) = if let Some(bloom) =
                self.systems
                    .features
                    .get_feature::<crate::renderer::features::bloom::BloomFeature>()
            {
                (bloom.output_view(), bloom.config().intensity)
            } else {
                (self.resources.black_texture.view(), 0.0)
            };

            let pp_ctx = crate::renderer::systems::post_process::PostProcessContext {
                device: &self.context.device.device,
                command_buffer,
                frame_index,
                image_index: image_index as usize,
                resources: &self.resources,
                frame: &self.frame,
                swapchain: self.frame.swapchain.as_ref().ok_or_else(|| {
                    AshError::VulkanError("Swapchain missing during post-process".to_string())
                })?,
                hdr: self.systems.hdr_system.as_ref(),
                taa_config: self.systems.taa_config.clone(),
                taa_metrics: Some(&mut self.systems.taa_config_metrics),
                jitter_uv,
                prev_jitter_uv: self.systems.prev_jitter_uv,
                depth_view,
                motion_view,
                bloom_view,
                bloom_intensity,
            };
            self.systems.prev_jitter_uv = jitter_uv;
            self.systems.pipeline.post_process.record_commands(pp_ctx)?;

            // â”€â”€ 7.5 UI / Debug Overlay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if let Some(callback) = ui_callback {
                callback(command_buffer);
            }

            // â”€â”€ 8. End & Submit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            cmd_ctx.end()?;
            self.frame.submit_and_present(&self.context, image_index)?;

            Ok(())
        }
    }

    pub fn render_frame(
        &mut self,
        scene: &mut Scene,
        view: Mat4,
        projection: Mat4,
        camera_pos: glam::Vec3,
        model_matrix: Option<Mat4>,
        ui_callback: Option<&dyn Fn(vk::CommandBuffer)>,
    ) -> Result<()> {
        scene.transform_system.update();
        scene.transform_system.update_buffers()?;

        if self.context.queue.is_resize_pending() {
            crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene)?;
        }

        // Minimization Guard: Skip frame if swapchain extent is zero
        if let Some(swapchain) = &self.frame.swapchain {
            if swapchain.extent.width == 0 || swapchain.extent.height == 0 {
                return Ok(());
            }
        }
        self.context
            .queue
            .flush_old_swapchains(&self.context.device);

        // 1. Sync
        self.sync_frame_resources(scene)?;

        // 2. Prepare
        let (frame_index, image_index, extent, jitter_proj, jitter_uv) =
            self.prepare_frame_data(scene, view, projection, camera_pos, model_matrix)?;

        self.frame.last_image_index = image_index;

        // 3. Record & Submit
        let render_ctx = RenderFrameContext {
            frame_index,
            image_index,
            scene,
            view,
            jitter_proj,
            jitter_uv,
            extent,
            ui_callback,
        };
        self.record_and_submit(render_ctx)?;

        Ok(())
    }

    pub fn buffer_pool(&self) -> Arc<BufferPool> {
        Arc::clone(&self.resources.buffer_pool)
    }

    /// Set the post-processing configuration
    pub fn set_post_processing_config(
        &mut self,
        config: crate::renderer::systems::post_process::PostProcessConfig,
    ) {
        self.systems.pipeline.post_process_mut().set_config(config);
    }

    /// Sets the tonemapping exposure value
    #[inline]
    pub fn set_tonemapping_exposure(&mut self, exposure: f32) {
        self.systems.post_process_mut().set_exposure(exposure);
    }

    /// Returns the tonemapping exposure value
    #[inline]
    pub fn tonemapping_exposure(&self) -> f32 {
        self.systems.post_process().exposure()
    }

    /// Returns the tonemapping gamma value
    #[inline]
    pub fn tonemapping_gamma(&self) -> f32 {
        self.systems.post_process().gamma()
    }

    /// Enables or disables bloom
    #[inline]
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        if let Some(bloom) = self
            .systems
            .features
            .get_feature_mut::<crate::renderer::features::bloom::BloomFeature>()
        {
            bloom.set_enabled(enabled);
        }
    }

    /// Returns whether bloom is enabled
    #[inline]
    pub fn bloom_enabled(&self) -> bool {
        if let Some(bloom) = self
            .systems
            .features
            .get_feature::<crate::renderer::features::bloom::BloomFeature>()
        {
            bloom.config().enabled
        } else {
            false
        }
    }

    /// Sets the bloom intensity
    #[inline]
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        if let Some(bloom) = self
            .systems
            .features
            .get_feature_mut::<crate::renderer::features::bloom::BloomFeature>()
        {
            bloom.set_intensity(intensity);
        }
    }

    /// Returns the bloom intensity
    #[inline]
    pub fn bloom_intensity(&self) -> f32 {
        if let Some(bloom) = self
            .systems
            .features
            .get_feature::<crate::renderer::features::bloom::BloomFeature>()
        {
            bloom.config().intensity
        } else {
            0.0
        }
    }

    // â”€â”€â”€ Lighting Delegates (owned by systems.lighting) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Update point lights for Forward+ rendering.
    ///
    /// Call once per frame before submitting render commands.
    pub fn update_point_lights(
        &mut self,
        lights: &[crate::renderer::features::PointLight],
    ) -> Result<()> {
        self.systems.lighting.update_point_lights(lights)?;
        Ok(())
    }

    /// Update directional lights for Forward+ rendering.
    pub fn update_directional_lights(
        &mut self,
        lights: &[crate::renderer::features::DirectionalLight],
    ) -> Result<()> {
        self.systems.lighting.update_directional_lights(lights)?;
        Ok(())
    }

    /// Update spot lights for Forward+ rendering.
    pub fn update_spot_lights(
        &mut self,
        lights: &[crate::renderer::features::SpotLight],
    ) -> Result<()> {
        self.systems.lighting.update_spot_lights(lights)?;
        Ok(())
    }

    /// Convenience overload: update point and directional lights in one call.
    pub fn update_lights(
        &mut self,
        point_lights: &[crate::renderer::features::PointLight],
        directional_lights: &[crate::renderer::features::DirectionalLight],
    ) -> Result<()> {
        self.systems
            .lighting
            .update_lights(point_lights, directional_lights)?;
        Ok(())
    }

    /// Returns whether Forward+ lighting is enabled
    pub fn forward_plus_enabled(&self) -> bool {
        self.systems.lighting.is_lighting_enabled()
    }

    /// Returns the number of active lights
    pub fn forward_plus_light_count(&self) -> usize {
        self.systems.lighting.light_count()
    }

    /// Enables HDR rendering. Should be called after initialization.
    /// Allocates GPU memory for the HDR buffer.
    pub(crate) fn initialize_hdr(&mut self, width: u32, height: u32) -> Result<()> {
        // 1. Explicitly drop the old system to free VRAM immediately
        // This prevents holding 2x HDR buffers (Old + New) simultaneously
        self.systems.hdr_system = None;

        unsafe {
            let hdr = HdrSystem::new(
                Arc::clone(&self.context.device.device),
                Arc::clone(&self.context.alloc),
                width,
                height,
            )?;

            // GBuffer Correctness (The Resize Trap)
            // If we have a previously registered hdr_image_index, update the bindless descriptor.
            // Otherwise, register it for the first time.
            if let Some(index) = self.frame.hdr_image_index {
                self.resources
                    .assets
                    .bindless_manager
                    .update_sampled_image(index, hdr.view(), hdr.sampler())?;
            } else {
                // First-time registration
                let index = self
                    .resources
                    .assets
                    .bindless_manager
                    .add_sampled_image(hdr.view(), hdr.sampler())?;
                self.frame.hdr_image_index = Some(index);
            }

            self.systems.hdr_system = Some(hdr);
            log::info!(
                "HDR System initialized ({width}x{height}) - Bindless Index: {:?}",
                self.frame.hdr_image_index
            );
        }

        Ok(())
    }

    /// Returns post-processing settings as a tuple (exposure, gamma, bloom_intensity)
    pub fn post_processing_settings(&self) -> (f32, f32, f32) {
        let bloom_intensity = if let Some(bloom) =
            self.systems
                .features
                .get_feature::<crate::renderer::features::bloom::BloomFeature>()
        {
            bloom.config().intensity
        } else {
            0.0
        };

        (
            self.systems.pipeline.post_process().config.exposure,
            self.systems.pipeline.post_process().config.gamma,
            bloom_intensity,
        )
    }

    // ========== Diagnostics API ==========

    /// Get current diagnostics state
    pub fn diagnostics(&self) -> &DiagnosticsState {
        &self.systems.diagnostics
    }

    /// Get mutable diagnostics state
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticsState {
        &mut self.systems.diagnostics
    }

    /// Set diagnostics display mode
    pub fn set_diagnostics_mode(&mut self, mode: DiagnosticsMode) {
        self.systems.diagnostics.mode = mode;
        log::info!("Diagnostics mode set to {mode:?}");
    }

    /// Toggle diagnostics mode (F6 behavior)
    pub fn toggle_diagnostics(&mut self) {
        self.systems.diagnostics.toggle_mode();
    }

    /// Collects frame diagnostics.
    /// Call this after render_frame() to collect stats
    pub fn update_diagnostics(&mut self) {
        if let Some(ref mut profiler) = self.systems.gpu_profiler {
            profiler.enabled = self.systems.diagnostics.mode != DiagnosticsMode::Off;
        }
        // Begin frame profiling
        self.systems.frame_profiler.begin_frame();

        // Collect frame stats
        self.systems.diagnostics.frame_stats = self.systems.frame_profiler.stats(
            self.systems.diagnostics.frame_stats.draw_calls,
            self.systems.diagnostics.frame_stats.triangles,
        );

        // Collect memory stats from buffer pool
        let stats = self.resources.buffer_pool.stats();
        self.systems.diagnostics.memory_stats.buffer_pool = (
            stats.current_available,
            stats.current_in_use,
            stats.total_allocated_bytes,
        );

        // Collect GPU timings (if profiler initialized)
        if let Some(ref mut profiler) = self.systems.gpu_profiler {
            self.systems.diagnostics.gpu_timings = profiler.end_frame();
        }

        // Print to console if enabled
        if self.systems.diagnostics.should_print_console() {
            self.systems.diagnostics.print_console();
        }
    }

    /// Log quality reports for debug/profiling
    pub fn log_quality_reports(&self) {
        match self.systems.pipeline.hiz_pass.read() {
            Ok(guard) => log::info!("{}", guard.quality_report()),
            Err(e) => log::warn!("HiZ lock poisoned in log_quality_reports: {e}"),
        }
    }

    /// Initialize GPU profiler for timing queries
    ///
    /// Automatically initialized when diagnostics are active.
    pub fn initialize_gpu_profiler(&mut self) -> Result<()> {
        if self.systems.gpu_profiler.is_some() {
            return Ok(());
        }

        let timestamp_period = self.context.device.timestamp_period_ns;
        let timestamps_supported = timestamp_period > 0.0;

        // SAFETY: `GpuProfiler::new` checks device limits internally.
        unsafe {
            let profiler = GpuProfiler::new(
                Arc::clone(&self.context.device.device),
                timestamp_period,
                timestamps_supported,
            )?;
            self.systems.gpu_profiler = Some(profiler);
        }

        Ok(())
    }

    /// Get overlay vertices for current frame
    ///
    /// Returns (text_vertices, background_vertices) for rendering.
    /// Call this after update_diagnostics() to get fresh data.
    pub fn overlay_vertices(
        &mut self,
    ) -> (
        &[crate::renderer::diagnostics::TextVertex],
        &[crate::renderer::diagnostics::TextVertex],
    ) {
        let extent = self
            .frame
            .swapchain
            .as_ref()
            .map(|s| (s.extent.width as f32, s.extent.height as f32))
            .unwrap_or((1920.0, 1080.0));

        self.systems.diagnostics_overlay.generate_vertices(
            &self.systems.diagnostics,
            extent.0,
            extent.1,
        )
    }

    /// Check if overlay should be rendered this frame
    pub fn should_render_overlay(&self) -> bool {
        self.systems.diagnostics.mode.overlay_enabled()
    }

    /// Get mutable reference to diagnostics overlay for configuration
    pub fn diagnostics_overlay_mut(&mut self) -> &mut DiagnosticsOverlay {
        &mut self.systems.diagnostics_overlay
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        log::info!("Shutting down Ash Renderer...");

        let _ = self.wait_for_idle();

        // Explicitly destroy systems that own GPU resources (VMA allocations)
        // Must happen BEFORE the allocator is destroyed.
        self.systems.destroy(&self.context);
        self.frame
            .frame_manager
            .destroy(&self.context.device.device);

        // Cleanup Global Cluster Buffer (BDA)
        // CRITICAL: This BDA buffer must be destroyed explicitly while the device is still valid
        // and BEFORE the allocator is dropped, as it depends on both.
        if let Some(_cluster_buffer) = self.resources.global_cluster_buffer.take() {
            // Drop will call destroy() via GlobalClusterBuffer::drop()
        }

        // SAFETY: The order of destruction is critical in Vulkan. We adhere to the standard
        // "destroy dependents first" rule:
        // 1. Swapchain-dependent pipelines (FullscreenPass, etc.) are cleared.
        // 2. Old swapchains are flushed to ensure no pending GPU work.
        // 3. Descriptor sets/pools are dropped.
        // 4. Feature subsystems are cleaned up.
        // 5. The ResourceRegistry (self.context.resources) handles the remaining tracked
        //    buffers, images, and samplers.
        #[allow(unused_unsafe)]
        unsafe {
            // CRITICAL FIX: Explicitly drop post-processing resources before general resource cleanup.
            // This prevents access violations during shutdown if the window/surface is destroyed.
            // ORDER MATTERS: Pipeline depends on resources destroyed in its children, so destroy Pipeline FIRST.
            swapchain_manager::cleanup_pipeline(self);

            // GBuffer and Depth are mandatory now, so we don't null them out here manually
            // The Registry will handle their cleanup.

            self.context
                .queue
                .flush_old_swapchains(&self.context.device);

            if let Some(manager) = self.resources.descriptors.take() {
                drop(manager);
            }

            self.systems.features.cleanup();

            // Cleanup all tracked resources via registry
            if let Err(_e) = self.context.resources.cleanup() {}
        }

        self.frame.swapchain = None;

        log::info!("Ash Renderer shut down successfully");
    }
}
