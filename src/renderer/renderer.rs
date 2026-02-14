use crate::{
    renderer::{
        diagnostics::{
            DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler,
        },
        features::{
            AutoRotateFeature, FeatureFrameContext, FeatureManager,
        },
        types::{
            DebugMode, DrawItem, RenderCommand,
            SampleShadingQuality, GBufferIndices, MeshData, RendererConfig,
        },
        passes::{
            vsr as vsr_pass,
            hiz::{AdaptiveHiZManager, HiZPass},
            motion::MotionVectorPass,
            temporal_aa::{
                detect_config_change, ConfigChangeType, ConfigMetrics, ConfigMetricsReport,
                ConfigValidationError, SharpeningMode, TaaConfig, Validate,
            },
            vsr::{SharpenConfig, VsrConfig, VsrInputs, VsrPass, VsrQuality, VsrUpscaleConfig},
        },
        vcgs::{CullBoundingBox, IndirectDrawPass},
        instancing::{BatchKey, InstanceData, InstancingManager},
        passes,
        resource_registry::{ResourceId, ResourceRegistry},
        resources,
        resources::{
            uniform::{StorageBuffer, UniformBuffer},
        },
        vram_budget, DepthBuffer, GBuffer, HdrSystem, MaterialHandle, MaterialManager, Mesh,
        PipelineCache, Texture, Scene,
        GeometryRenderContext,
        initialization,
        assets::AssetManager,
        frame_manager,
    },
    vulkan::{self, Allocator, CommandBufferContext},
    AshError, Result,
};

use ash::vk;
use glam::{Mat4, Vec3};
use rayon::prelude::*;
use resources::BufferPool;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::renderer::queue::RenderQueue;
use super::swapchain_manager;
use crate::renderer::resources::buffer::BufferHandle;
use crate::renderer::resources::mesh::{MaterialDescriptor, MeshDescriptor};
use crate::renderer::resources::GlobalClusterBuffer;




// RendererResources moved to init_types.rs



/// Main rendering system.
///
/// # Safety Contract
///
/// The `Renderer` manages complex GPU resource lifecycles. To ensure memory safety:
///
/// 1. **Drop Order**: Rust's drop order (top-to-bottom for fields) is critical. Buffers,
///    textures, and pipelines must be destroyed before the `Allocator` and `Device`.
/// 2. **Map Safety**: All `MapGuard` instances must be dropped before the underlying
///    buffer is destroyed or the allocator is dropped.
/// 3. **Validation**: Use Vulkan validation layers (`VK_LAYER_KHRONOS_validation`) in
///    development to verify that no resources leak or are accessed after destruction.
pub struct Renderer {
    // Resources dependent on allocator/device - dropped in reverse order.
    buffer_pool: Arc<BufferPool>,
    features: FeatureManager,
    _pipeline_cache: PipelineCache,
    prev_view_proj: Mat4,
    _default_texture: Texture,
    _black_texture: Texture,
    _white_texture: Texture,
    _default_skybox: Texture, // Procedural skybox fallback
    _default_cube_black: Texture, // Keep alive
    draw_items: Vec<DrawItem>,
    pub(crate) swapchain: Option<vulkan::SwapchainWrapper>,
    pub(crate) pipeline_id: Option<ResourceId>,
    
    // Skybox Rendering (Modularized)
    skybox_pass: Option<passes::SkyboxPass>,
    
    pub(crate) depth_buffer: Option<DepthBuffer>,
    pub(crate) uniform_buffers: Vec<Arc<RwLock<UniformBuffer>>>,
    pub material_storage_buffer: Option<Arc<RwLock<StorageBuffer<resources::uniform::MaterialUniform>>>>,
    pub(crate) pipeline_layout_id: Option<ResourceId>,
    pub(crate) descriptors: Option<vulkan::DescriptorAllocator>,
    start_time: Instant,
    pub(crate) swapchain_image_view_ids: Vec<ResourceId>,
    pub(crate) depth_buffer_id: Option<ResourceId>,
    pub frame_manager: frame_manager::FrameManager,
    pub cmds: Arc<vulkan::CommandBufferManager>,
    
    // Post-processing support
    sample_shading: crate::renderer::types::SampleShadingQuality,
    pub(crate) hdr_system: Option<crate::renderer::HdrSystem>,

    // Diagnostics
    diagnostics: DiagnosticsState,
    frame_profiler: FrameProfiler,
    gpu_profiler: Option<GpuProfiler>,
    diagnostics_overlay: DiagnosticsOverlay,

    // Dedicated Render Pipeline (Chunk 3.1)
    pub pipeline: crate::renderer::render_pipeline::RenderPipeline,

    // Bindless textures
    pub assets: AssetManager,
    // Forward+ lighting
    // Temporal Super-Resolution
    pub(crate) vsr_pass: Option<VsrPass>,
    // Motion Vector Pass for VSR/TAA
    pub(crate) motion_pass: Option<MotionVectorPass>,
    // G-Buffer for Normals and Motion Vectors
    pub(crate) gbuffer: Option<GBuffer>,
    // Pipeline optimization
    pub debug_mode: DebugMode,
    pub global_cluster_buffer: Option<Arc<GlobalClusterBuffer>>,
    
    pub(crate) gbuffer_indices: GBufferIndices,
    pub(crate) hdr_image_index: Option<u32>,

    instancing_manager: InstancingManager,
    instance_buffers: Vec<resources::InstanceBuffer>,
    // Image-Based Lighting
    strict_mode: bool,
    // Headless support
    readback_buffer: Option<BufferHandle>,
    last_image_index: u32,

    // TAA Configuration
    pub taa_config: TaaConfig,
    /// TAA configuration metrics (tracking changes/validation)
    taa_config_metrics: ConfigMetrics,

    pub vsr_config: VsrConfig,

    pub instance_buffer_addresses: Vec<u64>,
    pub material_heap_address: u64,

    pub(crate) geometry_buffer: Arc<resources::DualHeapGeometryBuffer>,
    pub(crate) resources: Arc<ResourceRegistry>,
    pub alloc: Arc<vulkan::Allocator>,
    pub queue: RenderQueue,
    pub device: vulkan::VulkanDevice,
}


pub struct MainPassParameters<'a> {
    pub cmd_ctx: &'a CommandBufferContext<'a>,
    pub frame_index: usize,
    pub image_index: u32,
    pub scene_pipeline: vk::Pipeline,
    pub pipeline_layout_handle: vk::PipelineLayout,
    pub batch_offsets: &'a HashMap<BatchKey, u32>,
    pub view: Mat4,
    pub projection: Mat4,
    pub swapchain_extent: vk::Extent2D,
    pub light_ptr: u64,
    pub tile_ptr: u64,
    pub scene: &'a super::Scene,
}

impl Renderer {
    fn sample_shading_config(&self) -> vulkan::MultisampleConfig {
        vulkan::MultisampleConfig {
            sample_count: vk::SampleCountFlags::TYPE_1,
            enable_sample_shading: self.sample_shading.enabled(),
            min_sample_shading: self.sample_shading.min_sample_shading(),
        }
    }

    pub fn wait_for_idle(&self) -> Result<()> {
        unsafe {
            self.device.device.device_wait_idle().map_err(|e| AshError::VulkanError(e.to_string()))
        }
    }

    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.debug_mode = mode;
        log::info!("Debug mode set to: {mode:?}");
    }

    /// Initializes the renderer.
    pub fn new<S: vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        log::info!("Renderer::new: Starting initialization");
        unsafe {
            log::info!("Creating Vulkan Foundation (Instance, Device, Allocator)");
            let instance = Arc::new(vulkan::VulkanInstance::new(
                surface_provider,
                cfg!(debug_assertions),
            )?);
            let device =
                vulkan::VulkanDevice::new(Arc::clone(&instance), surface_provider.is_headless())?;
            let alloc = Arc::new(vulkan::Allocator::new(&device)?);
            let resources = Arc::new(ResourceRegistry::new(Arc::clone(&device.device)));
            let dev_mem_props = device.memory_properties;
            let vram_budget = vram_budget::VramBudget::new(&dev_mem_props);

            log::info!("Initializing Features & Global Cache");
            let mut features = FeatureManager::new();
            features.set_device(Arc::clone(&device.device));
            features.add_feature(AutoRotateFeature::new());

            let pipeline_cache = PipelineCache::new(Arc::clone(&device.device))?;
            let renderer_config = RendererConfig::default();
            let mut pipeline_cfg = renderer_config.pipeline.clone();
            
            if !device.sample_rate_shading_supported && pipeline_cfg.sample_shading.enabled() {
                log::warn!("Sample rate shading requested but not supported by hardware. Falling back to disabled.");
                pipeline_cfg.sample_shading = SampleShadingQuality::Disabled;
            }

            let swapchain_with_ids = initialization::init_swapchain(&device, &alloc, &resources, surface_provider)?;
            let swapchain = swapchain_with_ids.data.swapchain;
            let depth_buffer = swapchain_with_ids.data.depth_buffer;
            let swapchain_image_view_ids = swapchain_with_ids.swapchain_image_view_ids;
            let depth_buffer_id = swapchain_with_ids.depth_buffer_id;

            let queue_data = initialization::init_render_queue(&device)?;
            let queue = queue_data.queue;

            let worker_count = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let cmds = Arc::new(vulkan::CommandBufferManager::new(
                Arc::clone(&device.device),
                device.graphics_queue_family,
                worker_count,
            )?);

            let frame_manager = frame_manager::FrameManager::new(
                &device.device,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len(),
            )?;

            let aspect = swapchain.extent.width as f32 / swapchain.extent.height as f32;

            // --- Modular Phase 26 Initialization ---
            let mut core = initialization::init_core_infrastructure(
                &device, &alloc, &instance, &resources,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len(),
                aspect
            )?;

            let set_layouts = [core.bindless_manager.descriptor_set_layout()];
            
            // Define color formats for Dynamic Rendering (Main Pass + GBuffer)
            // Note: Since GBuffer isn't created yet, we check for its intent or future presence.
            // For now, we follow the logic in recreate_pipeline.
            let color_formats = vec![
                swapchain.format,                   // Index 0: Main Color
            ];

            let pipelines = initialization::init_pipelines(
                &device, &resources, swapchain.extent,
                &color_formats,
                &set_layouts, &pipeline_cfg,
                depth_buffer.format(), pipeline_cache.handle(),
            )?;

            let passes = initialization::init_rendering_passes(
                &device, &alloc, &resources, &mut core.bindless_manager, &core.renderer_resources,
                swapchain.format,
                swapchain.extent, depth_buffer.format(), depth_buffer.view(),
                pipeline_cache.handle(),
                pipeline_cfg.multisample_config(),
                &set_layouts, &core.model_renderer, cmds.upload_command_pool_handle()
            )?;

            let lighting = initialization::init_lighting_system(
                &device, &alloc, &mut core.bindless_manager,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len() as u32, swapchain.extent
            )?;

            let post = initialization::init_post_processing(
                &device.device, swapchain.image_views.len(), swapchain.extent, swapchain.format
            )?;

            // Finalizing Metadata
            let material_heap_address = core.renderer_resources.material_storage_buffer.device_address();
            let mut instance_buffer_addresses = Vec::with_capacity(core.renderer_resources.instance_buffers.len());
            for buffer in &core.renderer_resources.instance_buffers {
                instance_buffer_addresses.push(buffer.device_address());
            }


            let readback_buffer = if device.headless {
                let buffer_size = (swapchain.extent.width * swapchain.extent.height * 4) as u64; 
                Some(BufferHandle::new_with_flags(
                    Arc::clone(&alloc), buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::Auto,
                    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    Some("Headless Readback Buffer".to_string()),
                )?)
            } else {
                None
            };

            log::info!("Renderer initialization complete. Constructing struct.");

            let mut renderer = Self {
                queue,
                buffer_pool: core.buffer_pool,
                resources,
                features,
                _pipeline_cache: pipeline_cache,
                prev_view_proj: Mat4::IDENTITY,
                _default_texture: core.renderer_resources.default_texture,
                _black_texture: core.renderer_resources.black_texture,
                _white_texture: core.renderer_resources.white_texture,
                _default_skybox: core.renderer_resources.default_skybox,
                _default_cube_black: core.renderer_resources.default_cube_black,
                draw_items: Vec::new(),
                swapchain: Some(swapchain),
                pipeline_id: Some(pipelines.pipeline_id),
                
                skybox_pass: Some(passes.skybox_pass),
                
                depth_buffer: Some(depth_buffer),
                uniform_buffers: core.renderer_resources.uniform_buffers.into_iter().map(|ub| Arc::new(RwLock::new(ub))).collect(),
                material_storage_buffer: Some(Arc::new(RwLock::new(core.renderer_resources.material_storage_buffer))),
                instance_buffer_addresses,
                material_heap_address,
                pipeline_layout_id: Some(pipelines.layout_id),
                descriptors: Some(core.descriptor_allocator),
                geometry_buffer: core.geometry_buffer,
                start_time: Instant::now(),
                alloc,
                device,
                swapchain_image_view_ids,
                depth_buffer_id: Some(depth_buffer_id),
                frame_manager,
                cmds,

                sample_shading: pipeline_cfg.sample_shading,
                hdr_system: None,

                diagnostics: DiagnosticsState::default(),
                frame_profiler: FrameProfiler::new(),
                gpu_profiler: None,
                diagnostics_overlay: DiagnosticsOverlay::new(),

                pipeline: crate::renderer::render_pipeline::RenderPipeline::new(
                    lighting.shadow_system,
                    post.post_process,
                    Some(Arc::new(RwLock::new(passes.hiz_pass))),
                    AdaptiveHiZManager::new(3.0),
                    Some(Arc::new(RwLock::new(lighting.forward_plus))),
                    Some(Arc::new(RwLock::new(passes.indirect_draw_pass))),
                    Some(pipelines.pipeline),
                    Some(pipelines.layout),
                ),

                assets: {
                    let mut assets = AssetManager::new(core.bindless_manager, vram_budget);
                    assets.texture_compression = renderer_config.texture_compression;
                    assets
                },
                vsr_pass: None,
                motion_pass: None,
                gbuffer: Some(passes.gbuffer),
                debug_mode: DebugMode::default(),
                global_cluster_buffer: Some(Arc::new(lighting.global_cluster_buffer)),
                gbuffer_indices: passes.gbuffer_indices,
                hdr_image_index: None,

                instancing_manager: InstancingManager::new(),
                instance_buffers: core.renderer_resources.instance_buffers,

                strict_mode: renderer_config.strict_mode,
                readback_buffer,
                last_image_index: 0,
                taa_config: TaaConfig::default(),
                taa_config_metrics: ConfigMetrics::default(),
                vsr_config: VsrConfig::default(),
            };

            let (image_count, extent) = {
                let sc = renderer.swapchain.as_ref().unwrap();
                (sc.image_views.len(), sc.extent)
            };
            renderer.pipeline.post_process_mut().resize(image_count, extent)?;
            renderer.pipeline.validate()?;
            renderer.register_tracked_subsystems()?;
            renderer.init_motion_pass()?;

            // --- Phase 2.7: Pre-initialize Forward+ lighting pipeline (prevents first-frame stutter) ---
            if let Some(ref fp_lock) = renderer.pipeline.forward_plus {
                if let (Some(db), Some(_db_ptr)) = (&renderer.depth_buffer, renderer.depth_buffer_id) {
                    let mut fp = fp_lock.write().unwrap();
                    fp.init_pipeline(
                        Arc::clone(&renderer.device.device),
                        db.sampler(),
                        db.view(),
                    )?;
                    log::info!("Forward+ lighting compute pipeline pre-initialized at startup.");
                }
            }

            renderer.queue.pending_extent = Some(renderer.swapchain.as_ref().unwrap().extent);

            Ok(renderer)
        }
    }

    /// Register all major subsystems with the ResourceRegistry for automatic cleanup
    fn register_tracked_subsystems(&mut self) -> Result<()> {
        let registry = &self.resources;

        if let Some(ref fp) = self.pipeline.forward_plus {
            registry.register_shared_resource(Arc::clone(fp)).map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        if let Some(ref hiz) = self.pipeline.hiz_pass {
            registry.register_shared_resource(Arc::clone(hiz)).map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        if let Some(ref indirect) = self.pipeline.indirect_draw_pass {
            registry.register_shared_resource(Arc::clone(indirect)).map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        // Register uniform buffers
        for ub in &self.uniform_buffers {
            registry.register_shared_resource(Arc::clone(ub)).map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        // Register material storage buffer
        if let Some(ref msb) = self.material_storage_buffer {
            registry.register_shared_resource(Arc::clone(msb)).map_err(|e| AshError::VulkanError(e.to_string()))?;
        }

        Ok(())
    }

    pub fn read_headless_image(&mut self) -> Result<Vec<u8>> {
        if !self.device.headless {
            return Err(AshError::VulkanError("Not in headless mode".to_string()));
        }

        let swapchain = self.swapchain.as_ref().ok_or(AshError::VulkanError(
            "Swapchain not initialized".to_string(),
        ))?;

        let readback_buffer = self.readback_buffer.as_mut().ok_or(AshError::VulkanError(
            "Readback buffer not initialized".to_string(),
        ))?;

        // Wait for the device to be idle to ensure rendering is complete
        unsafe {
            self.device.device.device_wait_idle().map_err(|e| {
                AshError::VulkanError(format!("Failed to wait for device idle: {e}"))
            })?;
        }

        let image_index = self.last_image_index;
        if image_index >= swapchain.images.len() as u32 {
            return Err(AshError::VulkanError("Invalid image index".to_string()));
        }
        let src_image = swapchain.images[image_index as usize];

        // Create a command buffer for the copy
        let cmd = self.cmds.get_transfer_command_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
                })?;

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    depth: 1,
                });

            self.device.device.cmd_copy_image_to_buffer(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buffer.handle(),
                &[copy_region],
            );

            // Add barrier to ensure write is visible to host
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(readback_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.device.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            self.device
                .device
                .end_command_buffer(cmd)
                .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

            let command_buffers = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            self.device
                .device
                .queue_submit(
                    self.device.graphics_queue,
                    &[submit_info],
                    vk::Fence::null(),
                )
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to submit copy command: {e}"))
                })?;

            self.device
                .device
                .queue_wait_idle(self.device.graphics_queue)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to wait for queue idle: {e}"))
                })?;

            self.device
                .device
                .free_command_buffers(self.cmds.upload_command_pool_handle(), &[cmd]);
        }

        // Map memory and read
        let size = (swapchain.extent.width * swapchain.extent.height * 4) as usize;
        let mut data = vec![0u8; size];

        unsafe {
            let ptr = self
                .alloc
                .vma
                .map_memory(readback_buffer.allocation_mut())
                .map_err(|e| AshError::VulkanError(format!("Failed to map memory: {e}")))?;

            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size);

            self.alloc
                .vma
                .unmap_memory(readback_buffer.allocation_mut());
        }

        Ok(data)
    }

    /// Initialize motion vector pass for VSR/TAA
    ///
    /// # Safety
    /// Must be called after GBuffer is initialized
    pub unsafe fn init_motion_pass(&mut self) -> Result<()> {
        if self.motion_pass.is_some() {
            return Ok(()); // Already initialized
        }

        let _gbuffer = self.gbuffer.as_ref().ok_or(AshError::VulkanError(
            "GBuffer must be initialized before motion pass".to_string(),
        ))?;

        let mut motion_pass = MotionVectorPass::new(Arc::clone(&self.device.device));
        
        // Initialize with G-Buffer motion format
        let motion_format = vk::Format::R16G16_SFLOAT;
        motion_pass.init(&self.device, motion_format)?;

        self.motion_pass = Some(motion_pass);
        
        log::info!("Motion vector pass initialized");

        Ok(())
    }









    /// Access the underlying memory allocator.
    pub fn allocator(&self) -> &Allocator {
        &self.alloc
    }

    pub fn bindless_manager(&self) -> &vulkan::BindlessManager {
        &self.assets.bindless_manager
    }

    pub fn bindless_manager_mut(&mut self) -> &mut vulkan::BindlessManager {
        &mut self.assets.bindless_manager
    }

    pub fn get_mesh_material(&self, scene: &Scene, mesh_handle: u32) -> MaterialHandle {
        scene.mesh_data
            .get(mesh_handle as usize)
            .map(|m| m.material_handle)
            .unwrap_or_else(|| MaterialHandle::null())
    }

    // Legacy lighting methods removed for modern RAGE pipeline



    /// Convenience for setting view and projection at once
    pub fn set_view(&mut self, _eye: Vec3, _center: Vec3, _up: Vec3) {
        // We calculate the view matrix here.
        // The projection is usually handled by the camera, but we can store it or calculate it.
        // For simplicity, let's just make this a no-op that logs for now, or actually store it.
        // Wait, renderer doesn't have a view matrix field yet.
    }


    /// Get immutable access to consolidated mesh data.
    pub fn mesh_data<'a>(&self, scene: &'a Scene) -> &'a [MeshData] {
        &scene.mesh_data
    }

    /// Get mutable access to mesh data by handle.
    pub fn get_mesh_data_mut<'a>(&self, scene: &'a mut Scene, handle: u32) -> Option<&'a mut MeshData> {
        scene.mesh_data.get_mut(handle as usize)
    }

    /// Returns the global geometry buffer for shared vertex/index storage.
    pub fn geometry_buffer(&self) -> Arc<resources::DualHeapGeometryBuffer> {
        Arc::clone(&self.geometry_buffer)
    }

    /// Get mutable access to the material manager.
    pub fn material_manager_mut<'a>(&self, scene: &'a mut Scene) -> &'a mut MaterialManager {
        &mut scene.material_manager
    }

    /// Returns a one-time use command buffer for transfer operations.
    pub fn get_transfer_command_buffer(&self) -> Result<vk::CommandBuffer> {
        self.cmds.get_transfer_command_buffer()
    }


    pub fn get_stats(&self) -> crate::renderer::diagnostics::RendererStats {
        crate::renderer::diagnostics::RendererStats {
            vram_usage: self.assets.vram_budget.get_stats(),
            draw_calls_per_frame: self.diagnostics.frame_stats.draw_calls,
            triangles_rendered: self.diagnostics.frame_stats.triangles,
            cull_efficiency: {
                // Calculate cull efficiency: (potential_draws - actual_draws) / potential_draws
                let potential_draws = (self.diagnostics.frame_stats.triangles / 1000).max(1) as f32;
                let actual_draws = self.diagnostics.frame_stats.draw_calls as f32;
                ((potential_draws - actual_draws) / potential_draws.max(1.0)).clamp(0.0, 1.0)
            },
            gpu_frame_ms: self.diagnostics.gpu_timings.total_ms,
            hiz_quality: format!("{:?}", self.pipeline.hiz_pass.as_ref().map(|h| h.read().unwrap().quality()).unwrap_or(crate::renderer::passes::hiz::HiZQuality::Balanced)),
            frame_count: self.diagnostics.frame_stats.total_frames,
        }
    }

    /// Logs the current frame statistics to the debug log.
    pub fn log_frame_stats(&self) {
        self.get_stats().log_frame_stats();
    }

    /// Unloads all currently registered textures from the host-side registry.
    /// Caution: Ensure no GPU frames are in flight using these textures before clearing.
    pub fn clear_texture_registry(&mut self) {
        self.assets.texture_registry.clear();
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
        let key = Arc::clone(&mesh.name);

        scene.upload_mesh(
            Arc::clone(&self.device.device),
            Arc::clone(&self.alloc),
            self.cmds.upload_command_pool_handle(),
            upload_cmd,
            &self.device.graphics_queue,
            &mut mesh,
            &mut self.assets,
            staging_resources,
            None, // No material override for simple descriptors
        )?;

        Ok(key.to_string())
    }

    /// AAA-grade transient transform upload (Phase 19).

    /// Converts a material descriptor into a renderer material and registers it.
    pub fn register_material_descriptor(
        &mut self,
        scene: &mut Scene,
        handle: u32,
        descriptor: &MaterialDescriptor,
    ) -> Result<MaterialHandle> {
        let mut material = descriptor.material.clone();
        material.name = format!("Material_{handle}");

        // Consolidated Phase 2.4 register call
        scene.register_material(&material)
    }

    /// Registers a generic storage buffer with the bindless manager.
    ///
    /// This is the "pro" way to handle bindless storage buffers, providing type safety
    /// and automatic memory management.
    pub fn register_bindless_storage_buffer<T: Copy>(
        &mut self,
        data: &[T],
        name: &str,
    ) -> Result<(
        Arc<parking_lot::Mutex<resources::uniform::StorageBuffer<T>>>,
        u32,
    )> {
        let bindless_manager = &mut self.assets.bindless_manager;

        unsafe {
            let mut buffer = resources::uniform::StorageBuffer::new(
                Arc::clone(&self.alloc),
                Arc::clone(&self.device.device),
                data.len(),
                name,
            )?;

            // Initial upload
            buffer.update(data)?;

            // Register with bindless manager (Binding 2)
            let index = bindless_manager.add_storage_buffer(
                buffer.buffer,
                0,
                std::mem::size_of_val(data) as vk::DeviceSize,
            )?;

            Ok((Arc::new(parking_lot::Mutex::new(buffer)), index))
        }
    }



    /// Submit render commands for the current frame.
    ///
    /// Each `RenderCommand` specifies a mesh handle, material handle, and transform.
    /// For large command counts (>1000), uses parallel processing across all CPU cores.
    pub fn submit_render_commands(&mut self, scene: &mut super::Scene, commands: &[RenderCommand]) -> Result<()> {
        log::debug!("Submitting {} render commands", commands.len());
        self.draw_items.clear();
        self.instancing_manager.begin_frame();

        const PARALLEL_THRESHOLD: usize = 1000;

        if commands.len() > PARALLEL_THRESHOLD {
            // Parallel extraction for large command counts
            use std::collections::HashMap;

            // Capture only thread-safe fields
            let mesh_data = &scene.mesh_data;
            let material_manager = &scene.material_manager;
            let strict_mode = self.strict_mode;

            let (draw_items, instance_batches) = commands
                .par_iter()
                .fold(
                    || (Vec::new(), HashMap::<BatchKey, Vec<InstanceData>>::new()),
                    |(items, mut batches), command| {
                        if let Some(mesh_data_entry) = mesh_data.get(command.mesh_handle as usize) {
                            let mesh_key = &mesh_data_entry.name;

                            let material_handle = if command.material_handle.is_null() {
                                mesh_data_entry.material_handle
                            } else {
                                command.material_handle
                            };
                            
                            // Unreal-Style validation: get material or fallback to default
                            let material = material_manager.get_material(material_handle);
                            
                            // Safety check: log if version mismatch (rare but possible)
                            if !material_manager.is_handle_valid(material_handle) {
                                let msg = format!("Invalid material handle {material_handle:?} detected for mesh handle {}, using default", command.mesh_handle);
                                if strict_mode {
                                    log::error!("{msg}");
                                } else {
                                    log::warn!("{msg}");
                                }
                            }


                            let key = BatchKey::new(command.mesh_handle, material_handle);
                            let mut instance = InstanceData::from_matrix(command.transform)
                                .with_bounds(mesh_data_entry.bounds)
                                .with_material_index(material_handle.index as u32);
                            if command.cast_shadows {
                                instance.set_flag(crate::renderer::vcgs::CULL_FLAG_CAST_SHADOWS, true);
                            }
                            if command.is_hidden {
                                instance.set_flag(crate::renderer::vcgs::CULL_FLAG_HIDDEN, true);
                            }
                            let item = DrawItem {
                                key: mesh_key.clone(),
                                mesh_id: command.mesh_handle,
                                transform: command.transform,
                                material: material.clone(),
                                material_handle,
                            };
                            
                            let mut items = items;
                            items.push(item);

                            batches
                                .entry(key)
                                .or_default()
                                .extend(vec![instance]);
                            (items, batches)
                        } else if strict_mode {
                            log::error!("Mesh handle {} not found in registry", command.mesh_handle);
                            (items, batches)
                        } else {
                            (items, batches)
                        }
                    },
                )
                .reduce(
                    || (Vec::new(), HashMap::<BatchKey, Vec<InstanceData>>::new()),
                    |(mut a_items, mut a_batches), (b_items, b_batches)| {
                        a_items.extend(b_items);
                        for (key, instances) in b_batches {
                            a_batches
                                .entry(key)
                                .or_default()
                                .extend(instances);
                        }
                        (a_items, a_batches)
                    },
                );

            // Merge results
            self.draw_items = draw_items;
            for (key, instances) in instance_batches {
                self.instancing_manager.add_instances(key, instances);
            }
        } else {
            // Sequential processing for small command counts (avoids rayon overhead)
            for command in commands {
                if let Some(mesh_data) = scene.mesh_data.get(command.mesh_handle as usize) {
                    let mesh_key = &mesh_data.name;

                    let material_handle = if command.material_handle.is_null() {
                        mesh_data.material_handle
                    } else {
                        command.material_handle
                    };

                    let material = scene.material_manager.get_material(material_handle);
                    
                    // Safety check: log if version mismatch (rare but possible)
                    if !scene.material_manager.is_handle_valid(material_handle) {
                        let msg = format!("Invalid material handle {material_handle:?} detected for mesh handle {}, using default", command.mesh_handle);
                        if self.strict_mode {
                            log::error!("{msg}");
                            return Err(AshError::VulkanError(msg));
                        } else {
                            log::warn!("{msg}");
                        }
                    }

                    // We must fetch the uploaded mesh to get the actual buffer offsets
                    if let Some(uploaded) = scene.model_renderer.get(&mesh_data.name) {

                        let key = BatchKey::new(command.mesh_handle, material_handle);
                        let item = DrawItem {
                            key: mesh_key.clone(),
                            mesh_id: command.mesh_handle,
                            transform: command.transform,
                            material: material.clone(),
                            material_handle,
                        };
                        self.draw_items.push(item);

                        let instance = InstanceData::from_matrix(command.transform)
                            .with_bounds(mesh_data.bounds)
                            .with_cast_shadows(command.cast_shadows)
                            .with_receive_shadows(command.receive_shadows)
                            .with_hidden(command.is_hidden)
                            .with_index_count(uploaded.index_count())
                            .with_first_index((uploaded.index_offset.unwrap_or(0) / 4) as u32)
                            .with_vertex_offset((uploaded.vertex_offset.unwrap_or(0) / 64) as i32)
                            .with_material_index(material_handle.index as u32);
                        self.instancing_manager.add_instance(key, instance);
                    } else {
                        log::error!("Mesh '{}' found in registry but not in model renderer cache!", mesh_data.name);
                    }
                } else {
                    let msg = format!("Mesh handle {} not found in registry", command.mesh_handle);
                    if self.strict_mode {
                        log::error!("{msg}");
                        return Err(AshError::MeshNotFound(command.mesh_handle));
                    } else {
                        log::warn!("{msg}");
                    }
                }
            }
        }



        self.instancing_manager.finalize();

        // Sort draw items to minimize pipeline and material changes
        self.draw_items.sort_by(|a, b| {
            a.material.name
                .cmp(&b.material.name)
                .then_with(|| a.key.cmp(&b.key))
        });

        Ok(())
    }

    /// Bake IBL maps from an equirectangular texture.
    ///
    /// This function converts an equirectangular environment map to cubemap format
    /// and generates irradiance and prefiltered maps for image-based lighting.
    /// Currently unused but preserved for runtime environment map loading features.


    pub fn request_swapchain_resize(&mut self, new_extent: vk::Extent2D) {
        self.queue.request_resize(new_extent);
    }

    // Simplified/Removed resize_if_needed and flush_old_swapchains as they are now handled by RenderQueue and SwapchainManager


    pub(crate) fn recreate_pipeline(&mut self) -> Result<()> {
        log::info!("Recompiling pipeline due to shader change...");
        let layout = self
            .pipeline
            .pipeline_layout
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Pipeline layout missing".to_string()))?
            .handle();
        
        // Determine color format (HDR or swapchain)
        let color_format = if let Some(hdr) = &self.hdr_system {
            hdr.format()
        } else {
            self.swapchain
                .as_ref()
                .ok_or(AshError::VulkanError("Swapchain missing".into()))?
                .format
        };
        
        let extent = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain missing".into()))?
            .extent;
        let cache = self._pipeline_cache.handle();
        let depth_format = self
            .depth_buffer
            .as_ref()
            .ok_or(AshError::VulkanError("Depth buffer missing".into()))?
            .format();

        let multisample_config = self.sample_shading_config();

        // Build color attachment formats based on GBuffer configuration
        let color_formats = if self.gbuffer.is_some() {
            vec![
                color_format,                       // Index 0: Main color (HDR or swapchain)
                vk::Format::R16G16B16A16_SFLOAT,   // Index 1: Normals
                vk::Format::R8G8B8A8_UNORM,        // Index 2: Albedo
                vk::Format::R16G16_SFLOAT,         // Index 3: Motion Vectors
            ]
        } else {
            vec![color_format] // ONLY Main Color (for Example 07)
        };

        let mut builder = vulkan::Pipeline::builder(Arc::clone(&self.device.device))
            .with_layout(layout)
            .with_dynamic_rendering(&color_formats, Some(depth_format), None)
            .with_extent(extent)
            .with_pipeline_cache(cache)
            .with_depth_format(depth_format)
            // CRITICAL FIX: Ensure recreated pipeline matches Reverse-Z (Greater/Equal)
            .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, true)
            .with_cull_mode(vk::CullModeFlags::NONE)
            .with_front_face(vk::FrontFace::CLOCKWISE)
            .with_multisampling(multisample_config);

        if self.gbuffer.is_some() {
            let blend_attachments = vec![
                // Index 0: Swapchain Color (with blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::TRUE,
                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                    alpha_blend_op: vk::BlendOp::ADD,
                },
                // Index 1: Normals (no blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                // Index 2: Albedo (no blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                // Index 3: Motion Vectors (no blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
            ];
            builder = builder.with_color_blend_attachments(blend_attachments);
        }

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/vert.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;
        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

        let mut new_pipeline = builder.build()?;
        let pipeline_layout_id = self.pipeline_layout_id.ok_or_else(|| {
            AshError::VulkanError("Pipeline layout ID missing during recreation".into())
        })?;

        // Register pipeline with only pipeline_layout dependency (no render_pass)
        let pipeline_id = self
            .resources
            .register_pipeline(new_pipeline.pipeline, &[pipeline_layout_id])
            .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;

        new_pipeline.mark_managed_by_registry();
        self.pipeline.main_graphics_pipeline = Some(new_pipeline);
        self.pipeline_id = Some(pipeline_id);

        log::info!("Pipeline recreation complete (Dynamic Rendering mode)");
        Ok(())
    }

    pub(crate) fn recreate_skybox_pipeline(&mut self) -> Result<()> {
        if self.skybox_pass.is_none() {
            log::info!("Skybox pass not initialized; skipping pipeline status check.");
            return Ok(());
        }

        log::info!("Checking skybox pipeline status...");

        // Skybox pass uses dynamic viewport and scissor states, so it adapts to 
        // swapchain extent changes automatically at draw time. The pipeline only needs
        // full recreation if the render pass format or descriptor layouts change,
        // which is already handled during root swapchain recreation.
        log::debug!("Skybox pass adapts via dynamic state; no explicit recreation needed.");
        
        Ok(())
    }

    pub(crate) fn update_image_views(&mut self, image_views: &[vk::ImageView]) -> Result<()> {
        if self.device.headless && !self.swapchain_image_view_ids.is_empty() {
             // In headless mode, we reuse the same image views. 
             // Cleaning them up would destroy the underlying Vulkan handles.
             return Ok(());
        }

        for id in self.swapchain_image_view_ids.drain(..) {
            if let Err(e) = self.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old swapchain image view {id}: {e}");
            }
        }

        self.swapchain_image_view_ids.clear();
        for &view in image_views {
            let id = self.resources.register_image_view(view).map_err(|e| {
                AshError::VulkanError(format!("Failed to register swapchain image view: {e}"))
            })?;
            self.swapchain_image_view_ids.push(id);
        }

        if let Some(ref mut sc) = self.swapchain {
            sc.mark_image_views_managed_by_registry();
        }

        Ok(())
    }

    pub(crate) fn recreate_depth_buffer(&mut self, extent: vk::Extent2D) -> Result<()> {
        if let Some(id) = self.depth_buffer_id.take() {
            if let Err(e) = self.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old depth buffer: {e}");
            }
        }

        let mut depth_buffer = unsafe {
            DepthBuffer::new(
                Arc::clone(&self.device.device),
                Arc::clone(&self.alloc),
                extent.width,
                extent.height,
            )?
        };

        let depth_buffer_id = depth_buffer
            .register_with_registry(&self.resources)
            .map_err(|e| AshError::VulkanError(format!("Failed to register depth buffer: {e}")))?;

        self.depth_buffer = Some(depth_buffer);
        self.depth_buffer_id = Some(depth_buffer_id);
        
        // Register Depth Buffer (The Resize Trap & Order-Independence)
        if self.gbuffer_indices.depth_index == u32::MAX {
            self.gbuffer_indices.depth_index = self.assets.bindless_manager.add_sampled_image(
                self.depth_buffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Depth buffer not initialized".into()))?
                    .view(),
                self._default_texture.sampler(),
            )?;
        } else {
            self.assets.bindless_manager.update_sampled_image(
                self.gbuffer_indices.depth_index,
                self.depth_buffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Depth buffer not initialized".into()))?
                    .view(),
                self._default_texture.sampler(),
            )?;
        }

        Ok(())
    }

    pub(crate) fn recreate_gbuffer(&mut self, extent: vk::Extent2D) -> Result<()> {
        let gbuffer = unsafe {
            GBuffer::new(
                Arc::clone(&self.device.device),
                Arc::clone(&self.alloc),
                extent.width,
                extent.height,
            )?
        };
        
        // Register Motion Vector for VSR (The Resize Trap & Order-Independence)
        if self.gbuffer_indices.motion_index == u32::MAX {
            self.gbuffer_indices.motion_index = self.assets.bindless_manager.add_sampled_image(
                gbuffer.motion_view(),
                self._default_texture.sampler(),
            )?;
        } else {
            self.assets.bindless_manager.update_sampled_image(
                self.gbuffer_indices.motion_index,
                gbuffer.motion_view(),
                self._default_texture.sampler(),
            )?;
        }
        
        self.gbuffer = Some(gbuffer);
        Ok(())
    }


    pub(crate) fn recreate_vsr_pass(&mut self, extent: vk::Extent2D) -> Result<()> {
        if let Some(ref mut vsr) = self.vsr_pass {
            unsafe {
                vsr.destroy(&self.alloc.vma);
                vsr.init(
                    &self.alloc.vma,
                    &self.device,
                    &mut self.assets.bindless_manager,
                    extent.width,
                    extent.height,
                    self.vsr_config.quality,
                )
                .map_err(|e| AshError::VulkanError(format!("VSR init failed: {e}")))?;
            }
        }
        Ok(())
    }

    pub(crate) fn recreate_frame_syncs(&mut self, count: usize) -> Result<()> {
        log::info!("Recreating frame synchronization objects: Count {}", count);
        self.frame_manager.destroy(&self.device.device);
        self.frame_manager = frame_manager::FrameManager::new(
            &self.device.device,
            self.cmds.upload_command_pool_handle(),
            count,
        )?;
        Ok(())
    }

    pub(crate) fn recreate_command_buffers(&mut self) -> Result<()> {
        log::info!("Resetting frame management lifecycle");
        self.frame_manager.reset_frame();
        Ok(())
    }

    pub(crate) fn recreate_uniform_buffers(&mut self, count: usize) -> Result<()> {
        for ub in &self.uniform_buffers {
            let _ = ub.write().unwrap().cleanup();
        }
        self.uniform_buffers.clear();

        unsafe {
            for _ in 0..count {
                let mut buffer =
                    UniformBuffer::new(Arc::clone(&self.alloc), Arc::clone(&self.device.device))?;

                {
                    // Initialize with identity matrices. Values are updated during render_frame.
                    let matrices = buffer.matrices_mut();
                    matrices.model = Mat4::IDENTITY;
                    matrices.view = Mat4::IDENTITY;
                    matrices.projection = Mat4::IDENTITY;
                    matrices.view_proj = Mat4::IDENTITY;
                    matrices.camera_pos = glam::Vec4::W; // (0,0,0,1)
                }
                buffer.update()?;
                self.uniform_buffers.push(Arc::new(RwLock::new(buffer)));
            }
        }
        Ok(())
    }

    pub(crate) fn recreate_descriptor_sets(&mut self) -> Result<()> {
        unsafe {
            self.device.device.device_wait_idle().map_err(|e| {
                AshError::VulkanError(format!("Failed to wait for device idle: {e:?}"))
            })?;
        }

        if let Some(_manager) = self.descriptors.as_mut() {
            let _count = self.frame_manager.get_max_frames_in_flight() as u32;
            // Removed recreate_frame_sets lines

            // CRITICAL FIX: Re-bind Environment defaults removed in Phase 3
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


    /// Executes the GPU-driven culling pass (Compute).
    /// Must be called OUTSIDE of a render pass.
    pub fn cull_main_pass(
        &mut self,
        params: &MainPassParameters,
    ) -> Result<()> {
        let cmd_ctx = params.cmd_ctx;
        let frame_index = params.frame_index;
        let view = params.view;
        let projection = params.projection;

        // Modern Phase 4: Full GPU-Driven Indirect Draw (Opaque Path)
            if let Some(ref indirect_arc) = self.pipeline.indirect_draw_pass {
                let indirect_pass = indirect_arc.write().unwrap();
            // No longer populating objects here; they were already populated in render_frame
            // with high-fidelity clusters.

            // 2. Upload and Execute Culling
            log::debug!("Occlusion culling object count: {}", params.scene.occlusion_culling.object_count());
            if params.scene.occlusion_culling.object_count() > 0 {
                let _frame_address = self.uniform_buffers[frame_index].read().unwrap().device_address();
                
                unsafe {
                    indirect_pass.upload_objects(&self.alloc.vma, params.scene.occlusion_culling.object_data(), 0)?;
                    
                    // Reset count buffer to 0 before compute pass
                    self.device.device.cmd_fill_buffer(cmd_ctx.handle(), indirect_pass.count_buffer(), 0, 4, 0);

                    // Compute barrier: ensure fill is done
                    let fill_barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                    
                    cmd_ctx.pipeline_barrier(
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[fill_barrier],
                        &[],
                        &[].as_ref(),
                    );

                    let extent = self.swapchain.as_ref().map_or(vk::Extent2D { width: 1, height: 1 }, |sw| sw.extent);

                    indirect_pass.execute_culling(
                        cmd_ctx.handle(),
                        &params.scene.occlusion_culling,
                        projection * view,
                        extent.width,
                        extent.height,
                        0,
                        params.scene.occlusion_culling.object_count() as u32,
                        0,
                        self.global_cluster_buffer.as_ref().map(|b| b.device_address()).unwrap_or(0),
                    )?;

                    // CRITICAL BARRIER: Compute-to-Graphics for Indirect Buffers
                    let indirect_barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ);
                    
                    cmd_ctx.pipeline_barrier(
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::VERTEX_SHADER,
                        vk::DependencyFlags::empty(),
                        &[indirect_barrier],
                        &[],
                        &[].as_ref(),
                    );
                }
            }
        }
        Ok(())
    }

    pub fn render_main_pass(
        &mut self,
        params: &MainPassParameters,
    ) -> Result<()> {
        let cmd_ctx = params.cmd_ctx;
        let frame_index = params.frame_index;
        let _scene_pipeline = params.scene_pipeline;
        let _pipeline_layout_handle = params.pipeline_layout_handle;
        let _batch_offsets = params.batch_offsets;
        let _view = params.view;
        let _projection = params.projection;
        let _swapchain_extent = params.swapchain_extent;

        // Resolve global debug state
        let debug_enabled = match self.debug_mode {
            DebugMode::None => false,
            _ => true,
        };

        // Context Upgrade: Retrieve views for Dynamic Rendering
        let (color_view, color_image) = if let Some(hdr) = &self.hdr_system {
            (hdr.view(), hdr.image())
        } else {
            let swapchain = self.swapchain.as_ref().ok_or_else(|| AshError::VulkanError("Swapchain missing".into()))?;
            let view = swapchain.image_views[params.image_index as usize];
            let image = swapchain.images[params.image_index as usize];
            (view, image)
        };

        let (depth_view, depth_image, depth_format) = self.depth_buffer.as_ref()
            .map(|d| (d.view(), d.image(), d.format()))
            .ok_or_else(|| AshError::VulkanError("Depth buffer missing".into()))?;

        // Create a dummy transform for feature rendering context
        let dummy_transform = crate::renderer::Transform::identity();

        // Phase 3.2, Part 2: Relocated Geometry Rendering
        let geo_ctx = GeometryRenderContext {
            device: &self.device,
            command_buffer: cmd_ctx,
            scene: params.scene,
            bindless_descriptor_set: self.assets.bindless_manager.descriptor_set(),
            swapchain_extent: params.swapchain_extent,
            frame_ptr: self.uniform_buffers[frame_index].read().unwrap().device_address(),
            material_ptr: self.material_heap_address,
            light_ptr: params.light_ptr,
            tile_ptr: params.tile_ptr,
            debug_enabled,
            color_image,
            color_view,
            depth_image,
            depth_view,
            normal_view: self.gbuffer.as_ref().map(|g| g.normal_view()),
            albedo_view: self.gbuffer.as_ref().map(|g| g.albedo_view()),
            motion_view: self.gbuffer.as_ref().map(|g| g.motion_view()),
            skybox: self.skybox_pass.as_ref(),
            features: Some(&self.features),
            frame_index,
            descriptor_allocator: self.descriptors.as_ref(),
            transform: &dummy_transform,
            is_swapchain_image: self.hdr_system.is_none(),
            depth_format,
        };

        self.pipeline.render_geometry(&geo_ctx)?;

        // --- PHASE 4: STRICT MODERN - Legacy paths removed ---
        // (Only skinned meshes would remain here if we hadn't moved them, 
        // but for now we focus on opaque stability)

        Ok(())
    }



    pub fn render_skybox(
        &mut self,
        cmd_ctx: &CommandBufferContext,
        frame_index: usize,
        _view: Mat4,
        _projection: Mat4,
    ) -> Result<()> {
        if let Some(ref skybox_pass) = self.skybox_pass {
            let bindless_set = self.assets.bindless_manager.descriptor_set();
            let frame_ptr = self.uniform_buffers[frame_index].read().unwrap().device_address();
            
            unsafe {
                skybox_pass.render(&self.device, cmd_ctx, bindless_set, frame_ptr)?;
            }
        }
        Ok(())
    }

    /// Set scene lighting configuration (RAGE)
    /// Deprecated: Use scene.set_lighting() instead.
    pub fn set_lighting(&mut self, _lighting: &crate::renderer::features::SceneLighting) {
        log::warn!("Renderer::set_lighting is deprecated. Use scene.set_lighting() instead.");
    }

    pub fn set_light_direction(&mut self, _direction: Vec3) {
        // Legacy shadow feature removed. VSM handles light direction internally or via scene updates.
        // self.shadow_feature.set_light_direction(direction);
    }

    /// Update TAA configuration with validation, metrics, and resource management.
    ///
    /// Validates the configuration, logs change severity, tracks metrics, and recreates
    /// TAA resources when a major change occurs (e.g., quality/sharpening/enable).
    pub fn update_taa_config(&mut self, config: TaaConfig) -> Result<(), ConfigValidationError> {
        // Validate incoming config
        config.validate()?;

        // Determine change type
        let change_type = detect_config_change(&self.taa_config, &config);

        // Log change severity
        match change_type {
            ConfigChangeType::None => {
                log::debug!("TAA config unchanged");
                return Ok(());
            }
            ConfigChangeType::Minor => {
                log::info!(
                    "TAA config updated (minor): {:?} -> {:?}",
                    self.taa_config.quality,
                    config.quality
                );
            }
            ConfigChangeType::Major => {
                log::info!(
                    "TAA config updated (major, recreation needed): {:?} -> {:?}",
                    self.taa_config.quality,
                    config.quality
                );
            }
        }

        // Apply configuration
        self.taa_config = config;

        // Track metrics
        self.taa_config_metrics
            .record_change(self.frame_manager.get_current_frame_index() as u64);

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
        self.frame_manager.reset_frame();
        log::debug!("TAA resources recreated");
    }

    /// Access TAA configuration metrics
    pub fn taa_config_metrics(&self) -> &ConfigMetrics {
        &self.taa_config_metrics
    }

    /// Generate TAA configuration metrics report
    pub fn taa_config_metrics_report(&self) -> ConfigMetricsReport {
        self.taa_config_metrics.report()
    }

    pub fn sync_frame_resources(&mut self, scene: &mut Scene) -> Result<()> {
        // Phase 20: Sync Materials to GPU (Modularity Fix)
        let sync_list: Vec<(u32, crate::renderer::resources::Material)> = {
            scene.material_manager.iter_unsynced(&scene.uploaded_material_indices)
                .map(|(id, material)| (id, material.clone()))
                .collect()
        };

        for (handle_index, material) in sync_list {
            if !scene.uploaded_material_indices.contains(&handle_index) {
                if let Err(e) = scene.register_material(&material) {
                    log::error!("Failed to sync material {handle_index} to GPU: {e}");
                }
            }
        }

        // Recycle per-frame descriptor pools (static pools are unaffected)
        if let Some(dm) = self.descriptors.as_mut() {
            dm.next_frame();
        }

        // Hot-reload shaders if changed (throttled to every ~1 second)
        const SHADER_CHECK_INTERVAL: usize = 60;

        // Ensure mutable borrow of pipeline scope ends prior to recreation call.
        let shaders_changed = if self.frame_manager.get_current_frame_index() % SHADER_CHECK_INTERVAL == 0 {
            if let Some(pipeline) = &mut self.pipeline.main_graphics_pipeline {
                match pipeline.detect_shader_changes() {
                    Ok(changed) => changed,
                    Err(e) => {
                        log::warn!("Failed to check shader changes: {e}");
                        false
                    }
                }
            } else {
                false
            }
        } else {
            false
        };

        if shaders_changed {
            if let Err(e) = crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene) {
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
        unsafe {
            let swapchain_extent = self
                .swapchain
                .as_ref()
                .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
                .extent;

            // Phase 1: Use FrameManager to acquire next frame and synchronization objects
            self.frame_manager.next_frame(&self.device.device)?;
            let swapchain_loader = ash::khr::swapchain::Device::new(self.device.instance.instance(), &self.device.device);
            let swapchain_khr = self.swapchain.as_ref().unwrap().swapchain;
            let (image_index, _suboptimal) = self.frame_manager.acquire_next_image(&swapchain_loader, swapchain_khr)?;
            
            let frame_index = self.frame_manager.get_current_frame_index();

            // Prepare culling data for this frame
            scene.occlusion_culling.begin_frame();
            for (i, item) in self.draw_items.iter().enumerate() {
                if let Some(uploaded) = scene.model_renderer.get(&item.key) {
                    let bounds = scene
                        .mesh_data
                        .get(item.mesh_id as usize)
                        .map(|m| m.bounds)
                        .unwrap_or_else(|| CullBoundingBox::new(Vec3::ZERO, Vec3::ONE * 100.0));
                    let material_index = item.material_handle.index as u32;
                    let vertex_offset = uploaded.vertex_offset.unwrap_or(0) as i32 / 64; 
                    
                    scene.occlusion_culling.push_clusters(
                        bounds,
                        item.transform,
                        i as u32,
                        uploaded.index_offset.unwrap_or(0) as u32 / 4,
                        uploaded.index_count(),
                        material_index,
                        vertex_offset,
                        uploaded.clusters(),
                    );
                } else {
                    log::warn!("Mesh key '{}' not found in model renderer cache!", item.key);
                }
            }

            // Apply sub-pixel jitter for VSR if enabled
            let mut jittered_projection = projection;
            let mut jitter_uv = [0.0f32; 2];
            if let Some(ref mut vsr) = self.vsr_pass {
                let (jx, jy) = vsr.next_jitter();
                jitter_uv = [jx, jy];
                let extent = self
                    .swapchain
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Swapchain missing".to_string()))?
                    .extent;
                jittered_projection.col_mut(2).x += jx / extent.width as f32;
                jittered_projection.col_mut(2).y += jy / extent.height as f32;
            }

            // GPU synchronization confirmed; safe to update uniform buffer.
            {
                let uniform_buffer = self.uniform_buffers.get_unchecked_mut(frame_index);
                let mut dummy_transform = resources::Transform::identity();
                let elapsed = self.start_time.elapsed().as_secs_f32();
                let mut feature_ctx = FeatureFrameContext {
                    device: self.device.device.as_ref(),
                    descriptor_allocator: self.descriptors.as_ref(),
                    transform: &mut dummy_transform,
                    auto_rotate: false,
                    elapsed_seconds: elapsed,
                };
                self.features.before_frame(&mut feature_ctx);

                if let Some(shadow_system) = self.pipeline.shadow_system_mut() {
                    shadow_system.vsm_feature_mut().begin_frame(frame_index as u32, camera_pos);
                }

                let mut ub = uniform_buffer.write().unwrap();
                let matrices = ub.matrices_mut();
                let model = model_matrix.unwrap_or(Mat4::IDENTITY);
                let mut transform = resources::Transform::identity();
                transform.set_model(model);
                
                matrices.model = model;
                matrices.normal_matrix = Mat4::from_mat3(transform.normal_matrix());
                matrices.view = view;
                matrices.projection = jittered_projection;
                matrices.view_proj = jittered_projection * view;
                matrices.prev_view_proj = self.prev_view_proj;
                matrices.camera_pos = camera_pos.extend(1.0);

                scene.scene_lighting.point_light_count = scene.point_lights.len() as u32;
                if let Some(fp) = &self.pipeline.forward_plus {
                    let info = fp.read().unwrap().get_lights().get_forward_plus_info();
                    scene.scene_lighting.num_tiles_x = info.num_tiles[0];
                    scene.scene_lighting.num_tiles_y = info.num_tiles[1];
                    scene.scene_lighting.tile_size = info.tile_size;
                }
                matrices.set_lighting(&scene.scene_lighting);
                matrices.set_light_space_matrix(glam::Mat4::IDENTITY);

                // Phase 2: Host-side Forward+ updates (Lights and Camera)
                // We perform these updates during preparation to match the Uniform Buffer lifecycle
                if let Some(ref fp_integration) = self.pipeline.forward_plus {
                    let mut fp = fp_integration.write().unwrap();
                    // Update light data from scene before uploading
                    fp.update_lights(&scene.point_lights, &scene.directional_lights, &scene.spot_lights);

                    // Update GPU buffers and descriptors for the current frame
                    fp.upload_to_gpu(&self.alloc, &self.device.device, frame_index as usize)?;

                    // Update camera buffer with current view/projection matrices
                    // We use the NON-JITTERED projection for culling to match frustum
                    fp.update_camera(
                        &self.alloc,
                        frame_index as usize,
                        &view.to_cols_array_2d(),
                        &projection.to_cols_array_2d(),
                        &camera_pos.extend(1.0).to_array(),
                    )?;
                }

                let view_proj = matrices.view_proj;
                ub.update()?;
                self.prev_view_proj = view_proj;
            }

            Ok((frame_index, image_index, swapchain_extent, jittered_projection, jitter_uv))
        }
    }

    fn execute_hiz_pass(&mut self, command_buffer: vk::CommandBuffer) -> Result<()> {
        if let Some(ref db) = self.depth_buffer {
            self.pipeline.execute_hiz_pass(
                command_buffer,
                db.image(),
                self.gpu_profiler.as_ref(),
                self._black_texture.view(),
                self._black_texture.sampler(),
            )?;
        }
        Ok(())
    }

    fn execute_shadow_pass(
        &mut self,
        command_buffer: vk::CommandBuffer,
        scene: &mut Scene,
        frame_index: usize,
    ) -> Result<(Vec<InstanceData>, HashMap<BatchKey, u32>)> {
        let mut all_instances = Vec::new();
        let mut batch_offsets = HashMap::new();
        {
            for batch in self.instancing_manager.batches() {
                batch_offsets.insert(batch.key.clone(), all_instances.len() as u32);
                all_instances.extend_from_slice(&batch.instances);
            }
        }
        if !all_instances.is_empty() {
            unsafe {
                self.instance_buffers[frame_index].update(&all_instances)?;
            }
        }

        let light_ptr = self.pipeline.forward_plus.as_ref().map(|fp| fp.read().unwrap().get_lights().light_ptr(frame_index)).unwrap_or(0);
        let tile_ptr = self.pipeline.forward_plus.as_ref().map(|fp| fp.read().unwrap().get_lights().tile_ptr(frame_index)).unwrap_or(0);

        let bindless_descriptor_set = self.assets.bindless_manager.descriptor_set();
        let uniform_buffer_address = self.uniform_buffers[frame_index].read().unwrap().device_address();

        self.pipeline.render_shadows(
            command_buffer,
            scene,
            frame_index,
            &self.device.device,
            bindless_descriptor_set,
            uniform_buffer_address,
            self.instance_buffer_addresses[frame_index],
            self.material_heap_address,
            light_ptr,
            tile_ptr,
            all_instances.len(),
        )?;
        
        Ok((all_instances, batch_offsets))
    }

    fn dispatch_light_culling(&self, command_buffer: vk::CommandBuffer, frame_index: usize) -> Result<()> {
        if let Some(ref fp_integration) = self.pipeline.forward_plus {
            let fp = fp_integration.read().unwrap();
            unsafe {
                fp.dispatch(command_buffer, &self.device.device, frame_index as usize);
            }

            let light_barrier = vk::BufferMemoryBarrier::default()
                .buffer(fp.lights().get_light_buffer(frame_index as usize).unwrap_or(vk::Buffer::null()))
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[light_barrier],
                    &[],
                );
            }
        }
        Ok(())
    }

    fn execute_geometry_pass(
        &mut self,
        cmd_ctx: &CommandBufferContext,
        frame_index: usize,
        image_index: u32,
        scene: &Scene,
        view: Mat4,
        jittered_projection: Mat4,
        swapchain_extent: vk::Extent2D,
        batch_offsets: &HashMap<BatchKey, u32>,
        scene_pipeline: vk::Pipeline,
    ) -> Result<()> {
        let light_ptr = self.pipeline.forward_plus.as_ref().map(|fp| fp.read().unwrap().get_lights().light_ptr(frame_index)).unwrap_or(0);
        let tile_ptr = self.pipeline.forward_plus.as_ref().map(|fp| fp.read().unwrap().get_lights().tile_ptr(frame_index)).unwrap_or(0);

        let pipeline_layout = self.pipeline.pipeline_layout.as_ref().ok_or_else(|| {
            AshError::VulkanError("Pipeline layout not available".to_string())
        })?;
        let pipeline_layout_handle = pipeline_layout.handle();

        let main_pass_params = MainPassParameters {
            cmd_ctx,
            frame_index,
            image_index,
            scene_pipeline,
            pipeline_layout_handle,
            batch_offsets,
            view,
            projection: jittered_projection,
            swapchain_extent,
            light_ptr,
            tile_ptr,
            scene,
        };

        self.cull_main_pass(&main_pass_params)?;

        cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, scene_pipeline);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        };
        cmd_ctx.set_viewport(0, &[viewport]);
        cmd_ctx.set_scissor(0, &[scissor]);

        if self.descriptors.is_some() {
            cmd_ctx.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout_handle,
                0, 
                &[self.assets.bindless_manager.descriptor_set()],
                &[],
            );
        }

        self.render_main_pass(&main_pass_params)?;
        Ok(())
    }

    fn execute_post_process(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
        swapchain_extent: vk::Extent2D,
        jitter_uv: [f32; 2],
    ) -> Result<()> {
        if let (Some(ref mut vsr), Some(ref mut _gbuffer)) = (&mut self.vsr_pass, &mut self.gbuffer) {
            let _: std::result::Result<vsr_pass::VsrMetricsReadback, vsr_pass::VsrError> = vsr.readback_metrics(command_buffer, &self.alloc.vma);

            let sharpen_mode = self.taa_config.sharpening;
            let sharpen_config = if sharpen_mode != SharpeningMode::None {
                Some(SharpenConfig {
                    strength: sharpen_mode.strength(),
                    edge_threshold: 0.12,
                    adaptive: true,
                })
            } else {
                None
            };

            let vsr_config = VsrUpscaleConfig {
                velocity_threshold: self.taa_config.velocity_threshold,
                history_weight: self.taa_config.blend_factor,
                clamping_gamma: self.taa_config.quality.clamping_gamma(),
                anti_ghosting: true,
            };

            let vsr_inputs = VsrInputs {
                color_index: self.hdr_image_index.ok_or_else(|| {
                    AshError::VulkanError("HDR image index not initialized for VSR".to_string())
                })?,
                depth_index: if self.gbuffer_indices.depth_index == u32::MAX {
                    return Err(AshError::VulkanError("Depth index not initialized for VSR".to_string()));
                } else {
                    self.gbuffer_indices.depth_index
                },
                motion_index: if self.gbuffer_indices.motion_index == u32::MAX {
                    return Err(AshError::VulkanError("Motion index not initialized for VSR".to_string()));
                } else {
                    self.gbuffer_indices.motion_index
                },
                jitter: jitter_uv,
            };

            unsafe {
                vsr.upscale_with_sharpening(command_buffer, vsr_inputs, &vsr_config, sharpen_config.as_ref())
                    .map_err(|e| AshError::VulkanError(format!("VSR upscale failed: {e}")))?;
            }
            vsr.next_frame();
        }

        self.pipeline.render_post_process(
            command_buffer,
            image_index as usize,
            swapchain_extent,
            self.swapchain.as_ref().unwrap().images[image_index as usize],
            self.swapchain.as_ref().unwrap().image_views[image_index as usize],
            self.hdr_system.as_ref(),
            self.vsr_pass.as_ref(),
            self._black_texture.view(),
        )?;

        Ok(())
    }

    pub fn record_and_submit(
        &mut self,
        frame_index: usize,
        image_index: u32,
        scene: &mut Scene,
        view: Mat4,
        jittered_projection: Mat4,
        jitter_uv: [f32; 2],
        swapchain_extent: vk::Extent2D,
    ) -> Result<()> {
        unsafe {
            let scene_pipeline = self
                .pipeline
                .main_graphics_pipeline
                .as_ref()
                .map(|p| p.pipeline)
                .ok_or(AshError::VulkanError("Pipeline not available".to_string()))?;

            let command_buffer = self.frame_manager.begin_command_buffer(&self.device.device)?;
            let device_arc = Arc::clone(&self.device.device);
            let cmd_ctx = CommandBufferContext::new(device_arc.as_ref(), command_buffer);

            log::debug!(
                "Frame {}: Using image index {}",
                self.frame_manager.get_current_frame_index(),
                image_index
            );

            // 1. Barriers: Ensure all host-written buffers are visible to GPU
            let global_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::UNIFORM_READ
                    | vk::AccessFlags::INDEX_READ
                    | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
                );

            self.device.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[global_barrier],
                &[],
                &[],
            );

            // 2. Hi-Z Pass
            self.execute_hiz_pass(command_buffer)?;

            // 3. Shadow Pass (Includes Instance Data Preparation)
            let (_all_instances, batch_offsets) = self.execute_shadow_pass(command_buffer, scene, frame_index)?;

            // 4. Light Culling Pass
            self.dispatch_light_culling(command_buffer, frame_index)?;

            // 5. Geometry Pass (Culling & Main Rendering)
            self.execute_geometry_pass(
                &cmd_ctx,
                frame_index,
                image_index,
                scene,
                view,
                jittered_projection,
                swapchain_extent,
                &batch_offsets,
                scene_pipeline,
            )?;

            // 6. Post-Process Pass (Includes VSR)
            self.execute_post_process(command_buffer, image_index, swapchain_extent, jitter_uv)?;

            // 7. End and Submit
            cmd_ctx.end()?;

            let swapchain_loader = ash::khr::swapchain::Device::new(self.device.instance.instance(), &self.device.device);
            let swapchain_khr = self.swapchain.as_ref().unwrap().swapchain;
            
            self.frame_manager.submit_and_present(
                &self.device.device,
                self.queue.graphics_queue,
                self.queue.present_queue,
                &swapchain_loader,
                swapchain_khr,
                image_index,
            )?;

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
    ) -> Result<()> {
        scene.transform_system.update();
        scene.transform_system.update_buffers()?;
        
        if self.queue.is_resize_pending() {
            crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene)?;
        }
        self.queue.flush_old_swapchains(&self.device);

        // 1. Sync
        self.sync_frame_resources(scene)?;

        // 2. Prepare
        let (frame_index, image_index, extent, jitter_proj, jitter_uv) = 
            self.prepare_frame_data(scene, view, projection, camera_pos, model_matrix)?;
        
        self.last_image_index = image_index;

        // 3. Record & Submit
        self.record_and_submit(frame_index, image_index, scene, view, jitter_proj, jitter_uv, extent)?;

        Ok(())
    }


    pub fn buffer_pool(&self) -> Arc<BufferPool> {
        Arc::clone(&self.buffer_pool)
    }







    /// Enables or disables tonemapping

    /// Set the post-processing configuration
    pub fn set_post_processing_config(
        &mut self,
        config: crate::renderer::systems::post_process::PostProcessConfig,
    ) {
        self.pipeline.post_process_mut().config = config;
    }

    /// Returns whether tonemapping is enabled
    pub fn tonemapping_enabled(&self) -> bool {
        self.pipeline.post_process().config.tonemapping_enabled
    }

    /// Sets the tonemapping exposure value
    pub fn set_tonemapping_exposure(&mut self, exposure: f32) {
        self.pipeline.post_process_mut().config.exposure = exposure.max(0.0);
    }

    /// Returns the tonemapping exposure value
    pub fn tonemapping_exposure(&self) -> f32 {
        self.pipeline.post_process().config.exposure
    }

    /// Sets the tonemapping gamma value
    pub fn set_tonemapping_gamma(&mut self, gamma: f32) {
        self.pipeline.post_process_mut().config.gamma = gamma.max(0.1);
    }

    /// Returns the tonemapping gamma value
    pub fn tonemapping_gamma(&self) -> f32 {
        self.pipeline.post_process().config.gamma
    }

    /// Enables or disables bloom
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.pipeline.post_process_mut().config.bloom_enabled = enabled;
    }

    /// Returns whether bloom is enabled
    pub fn bloom_enabled(&self) -> bool {
        self.pipeline.post_process().config.bloom_enabled
    }

    /// Sets the bloom intensity
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        self.pipeline.post_process_mut().config.bloom_intensity = intensity.clamp(0.0, 2.0);
    }

    /// Returns the bloom intensity
    pub fn bloom_intensity(&self) -> f32 {
        self.pipeline.post_process().config.bloom_intensity
    }

   
    /// Update point lights for Forward+ rendering
    ///
    /// Call this each frame to update light positions and properties.
    pub fn update_point_lights(&mut self, lights: &[crate::renderer::features::PointLight]) {
        if let Some(ref forward_plus) = self.pipeline.forward_plus {
            forward_plus.write().unwrap().update_lights(lights, &[], &[]);
        }
    }

    /// Update directional lights for Forward+ rendering
    pub fn update_directional_lights(
        &mut self,
        lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref forward_plus) = self.pipeline.forward_plus {
            forward_plus.write().unwrap().update_lights(&[], lights, &[]);
        }
    }

    /// Update spot lights for Forward+ rendering
    pub fn update_spot_lights(&mut self, lights: &[crate::renderer::features::SpotLight]) {
        if let Some(ref forward_plus) = self.pipeline.forward_plus {
            forward_plus.write().unwrap().update_lights(&[], &[], lights);
        }
    }

    /// Update all lights (point and directional) for Forward+ rendering
    pub fn update_lights(
        &mut self,
        point_lights: &[crate::renderer::features::PointLight],
        directional_lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref forward_plus) = self.pipeline.forward_plus {
            forward_plus.write().unwrap().update_lights(point_lights, directional_lights, &[]);
        }
    }

    /// Returns whether Forward+ lighting is enabled
    pub fn forward_plus_enabled(&self) -> bool {
        self.pipeline.forward_plus
            .as_ref()
            .map(|fp| fp.read().unwrap().is_enabled())
            .unwrap_or(false)
    }

    /// Returns the number of active lights
    pub fn forward_plus_light_count(&self) -> usize {
        self.pipeline.forward_plus
            .as_ref()
            .map(|fp| {
                let lights = fp.read().unwrap();
                lights.light_count()
            })
            .unwrap_or(0)
    }

    
    /// Enables GPU-driven occlusion culling using Hi-Z pyramid.
    ///
    /// This initializes the Hi-Z pass and indirect draw pass for GPU-based
    /// visibility culling. Objects are tested against a hierarchical depth
    /// buffer before rendering, reducing draw calls significantly.
    ///
    /// # Safety
    /// Should be called after the renderer is fully initialized.
    pub fn enable_occlusion_culling(&mut self) -> Result<()> {
        if self.pipeline.hiz_pass.is_some() {
            return Ok(()); // Already enabled
        }

        let extent = self
            .swapchain
            .as_ref()
            .map(|s| s.extent)
            .unwrap_or(vk::Extent2D {
                width: 1920,
                height: 1080,
            });

        // Create Hi-Z pass
        let mut hiz = HiZPass::new(Arc::clone(&self.device.device));
        unsafe {
            hiz.init(&self.alloc, &self.device, extent.width, extent.height)?;
        }

        // Create Indirect Draw pass
        let mut indirect = IndirectDrawPass::new(Arc::clone(&self.device.device));
        if self.descriptors.is_none() {
            return Err(AshError::VulkanError(
                "DescriptorManager not initialized".to_string(),
            ));
        }
        let bindless_manager = &mut self.assets.bindless_manager;

        unsafe {
            indirect.init(
                &self.alloc,
                &self.device,
                bindless_manager,
                crate::renderer::vcgs::MAX_INDIRECT_OBJECTS,
            )?;
            let hiz_view = if let Some(view) = hiz.hiz_view() {
                view
            } else {
                // REVERSE-Z FIX: 0.0 is Far Plane (no occlusion)
                self._black_texture.view()
            };

            let hiz_sampler = if hiz.is_initialized() { 
                hiz.hiz_sampler() 
            } else { 
                self._black_texture.sampler() 
            };
            indirect.update_hiz_descriptor(hiz_view, hiz_sampler);
        }

        self.pipeline.hiz_pass = Some(Arc::new(RwLock::new(hiz)));
        self.pipeline.indirect_draw_pass = Some(Arc::new(RwLock::new(indirect)));

        // Register new subsystems with the registry (Phase 31)
        self.register_tracked_subsystems()?;

        log::info!("Occlusion culling enabled (Hi-Z + Indirect Draw)");
        Ok(())
    }

    /// Returns whether GPU-driven occlusion culling is enabled
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.pipeline.hiz_pass.is_some() && self.pipeline.indirect_draw_pass.is_some()
    }


    /// Enables Temporal Super-Resolution (VSR)
    ///
    /// VSR renders at a lower internal resolution and uses temporal
    /// accumulation to reconstruct higher quality output. This improves
    /// performance while maintaining near-native image quality.
    ///
    /// # Arguments
    /// * `quality` - The VSR quality preset (affects internal render resolution)
    pub fn enable_vsr(&mut self, quality: VsrQuality) -> Result<()> {
        if self.vsr_pass.is_some() {
            return Ok(()); // Already enabled
        }

        let extent = self
            .swapchain
            .as_ref()
            .map(|s| s.extent)
            .unwrap_or(vk::Extent2D {
                width: 1920,
                height: 1080,
            });

        let mut vsr = VsrPass::new(Arc::clone(&self.device.device));
        unsafe {
            vsr.init(
                &self.alloc.vma,
                &self.device,
                &mut self.assets.bindless_manager,
                extent.width,
                extent.height,
                quality,
            )
            .map_err(|e| AshError::VulkanError(format!("VSR init failed: {e}")))?;
        }

        self.vsr_pass = Some(vsr);
        self.vsr_config.quality = quality;
        log::info!(
            "VSR enabled with {:?} quality ({}x upscale)",
            quality,
            quality.factor()
        );
        Ok(())
    }

    /// Returns whether VSR is enabled
    pub fn vsr_enabled(&self) -> bool {
        self.vsr_pass.is_some()
    }

    /// Returns the current VSR quality preset
    pub fn vsr_quality(&self) -> Option<VsrQuality> {
        self.vsr_pass.as_ref().map(|t| t.config.quality)
    }

    /// Get jittered projection matrix for TAA/VSR
    ///
    /// Call this each frame to get a projection matrix with sub-pixel jitter
    /// applied. This is essential for temporal accumulation quality.
    pub fn jitter_projection(&mut self, projection: glam::Mat4) -> glam::Mat4 {
        if let Some(ref mut vsr) = self.vsr_pass {
            vsr.jitter_projection(projection)
        } else {
            projection
        }
    }


    

    /// Enables HDR rendering. Should be called after initialization.
    /// Allocates GPU memory for the HDR buffer.
    pub(crate) fn initialize_hdr(&mut self, width: u32, height: u32) -> Result<()> {
        // 1. Explicitly drop the old system to free VRAM immediately
        // This prevents holding 2x HDR buffers (Old + New) simultaneously
        self.hdr_system = None;

        unsafe {
            let hdr = HdrSystem::new(
                Arc::clone(&self.device.device),
                Arc::clone(&self.alloc),
                width,
                height,
            )?;
            
            // Phase 2: GBuffer Correctness (The Resize Trap)
            // If we have a previously registered hdr_image_index, update the bindless descriptor.
            // Otherwise, register it for the first time.
            if let Some(index) = self.hdr_image_index {
                self.assets.bindless_manager.update_sampled_image(
                    index,
                    hdr.view(),
                    hdr.sampler(),
                )?;
            } else {
                // First-time registration
                let index = self.assets.bindless_manager.add_sampled_image(
                    hdr.view(),
                    hdr.sampler(),
                )?;
                self.hdr_image_index = Some(index);
            }

            self.hdr_system = Some(hdr);
            log::info!("HDR System initialized ({width}x{height}) - Bindless Index: {:?}", self.hdr_image_index);
        }

        Ok(())
    }

    /// Enables post-processing with default settings
    ///
    /// Initializes HDR, fullscreen pass, and enables tonemapping.
    pub fn enable_post_processing(&mut self, scene: &mut super::Scene) -> Result<()> {
        let extent = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain not available".into()))?
            .extent;

        self.initialize_hdr(extent.width, extent.height)?;
        
        // PostProcessSystem initializes its own FullscreenPass in new()
        // and descriptors are handled by resize() and update_descriptor_sets().
        // So we don't need manual initialization here.

        self.pipeline.post_process_mut().config.tonemapping_enabled = true;
        
        // CRITICAL: Recreate main pipeline and framebuffers to use the NEW HDR render pass format
        crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene)?;
        
        log::info!("Post-processing pipeline enabled (HDR + Tonemapping)");
        Ok(())
    }



    /// Returns post-processing settings as a tuple (exposure, gamma, bloom_intensity)
    pub fn post_processing_settings(&self) -> (f32, f32, f32) {
        (
            self.pipeline.post_process().config.exposure,
            self.pipeline.post_process().config.gamma,
            self.pipeline.post_process().config.bloom_intensity,
        )
    }

    // ========== Diagnostics API ==========

    /// Get current diagnostics state
    pub fn diagnostics(&self) -> &DiagnosticsState {
        &self.diagnostics
    }

    /// Get mutable diagnostics state
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticsState {
        &mut self.diagnostics
    }

    /// Set diagnostics display mode
    pub fn set_diagnostics_mode(&mut self, mode: DiagnosticsMode) {
        self.diagnostics.mode = mode;
        log::info!("Diagnostics mode set to {mode:?}");
    }

    /// Toggle diagnostics mode (F6 behavior)
    pub fn toggle_diagnostics(&mut self) {
        self.diagnostics.toggle_mode();
    }

    /// Collects frame diagnostics.
    /// Call this after render_frame() to collect stats
    pub fn update_diagnostics(&mut self) {
        if let Some(ref mut profiler) = self.gpu_profiler {
            profiler.enabled = self.diagnostics.mode != DiagnosticsMode::Off;
        }
        // Begin frame profiling
        self.frame_profiler.begin_frame();

        // Collect frame stats
        self.diagnostics.frame_stats = self.frame_profiler.stats(
            self.diagnostics.frame_stats.draw_calls,
            self.diagnostics.frame_stats.triangles,
        );

        // Collect memory stats from buffer pool
        let stats = self.buffer_pool.stats();
        self.diagnostics.memory_stats.buffer_pool = (
            stats.current_available,
            stats.current_in_use,
            stats.total_allocated_bytes,
        );

        // Collect GPU timings (if profiler initialized)
        if let Some(ref mut profiler) = self.gpu_profiler {
            self.diagnostics.gpu_timings = profiler.end_frame();
        }

        // Print to console if enabled
        if self.diagnostics.should_print_console() {
            self.diagnostics.print_console();
        }
    }

    /// Log quality reports for debug/profiling
    pub fn log_quality_reports(&self) {
        if let Some(hiz) = &self.pipeline.hiz_pass {
            log::info!("{}", hiz.read().unwrap().quality_report());
        }

        if let Some(vsr) = &self.vsr_pass {
            log::info!("{}", vsr.quality_report());
        }
    }

    /// Initialize GPU profiler for timing queries
    ///
    /// Automatically initialized when diagnostics are active.
    pub fn initialize_gpu_profiler(&mut self) -> Result<()> {
        if self.gpu_profiler.is_some() {
            return Ok(());
        }

        let timestamp_period = self.device.timestamp_period_ns;
        let timestamps_supported = timestamp_period > 0.0;

        // SAFETY: `GpuProfiler::new` checks device limits internally.
        unsafe {
            let profiler = GpuProfiler::new(
                Arc::clone(&self.device.device),
                timestamp_period,
                timestamps_supported,
            )?;
            self.gpu_profiler = Some(profiler);
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
            .swapchain
            .as_ref()
            .map(|s| (s.extent.width as f32, s.extent.height as f32))
            .unwrap_or((1920.0, 1080.0));

        self.diagnostics_overlay
            .generate_vertices(&self.diagnostics, extent.0, extent.1)
    }

    /// Check if overlay should be rendered this frame
    pub fn should_render_overlay(&self) -> bool {
        self.diagnostics.mode.overlay_enabled()
    }

    /// Get mutable reference to diagnostics overlay for configuration
    pub fn diagnostics_overlay_mut(&mut self) -> &mut DiagnosticsOverlay {
        &mut self.diagnostics_overlay
    }

}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            log::info!("Shutting down Ash Renderer...");

            let _ = self.wait_for_idle();
            
            self.frame_manager.destroy(&self.device.device);

            // Phase 7: Cleanup Global Cluster Buffer (BDA)
            // CRITICAL: This BDA buffer must be destroyed explicitly while the device is still valid
            // and BEFORE the allocator is dropped, as it depends on both.
            if let Some(_cluster_buffer) = self.global_cluster_buffer.take() {
                // Drop will call destroy() via GlobalClusterBuffer::drop()
            }

            // CRITICAL FIX: Explicitly drop post-processing resources before general resource cleanup.
            // This prevents access violations during shutdown if the window/surface is destroyed.
            // ORDER MATTERS: Pipeline depends on RenderPass (in FullscreenPass), so destroy Pipeline FIRST.

            swapchain_manager::cleanup_pipeline(self);

            // Cleanup VSM (Explicit)
            if let Some(mut shadow_system) = self.pipeline.take_shadow_system() {
                shadow_system.destroy();
            }

            self.queue.flush_old_swapchains(&self.device);

            if let Err(e) = self.resources.cleanup() {
                log::error!("Resource registry cleanup failed: {e}");
            }

            if let Some(manager) = self.descriptors.take() {
                drop(manager);
            }

            self.features.cleanup();

            // Cleanup all tracked resources via registry (Consolidated Phase 31)
            if let Err(e) = self.resources.cleanup() {
                log::error!("Resource cleanup failed: {e}");
            }

            self.draw_items.clear();
            
            self.depth_buffer = None;
            self.pipeline.main_graphics_pipeline = None;
            self.swapchain = None;


            log::info!("Ash Renderer shut down successfully");
        }
    }
}

