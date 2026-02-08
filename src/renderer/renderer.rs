use crate::{
    renderer::{
        diagnostics::{
            DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler,
        },
        features::{
            AutoRotateFeature, FeatureFrameContext, FeatureManager,
            FeatureRenderContext,
            default_vsm_config,
        },
        types::{
            DebugMode, DrawItem, GBufferIndices, MeshData, RenderCommand,
            RendererConfig, SampleShadingQuality, TexturePresenceFlags,
        },
        ForwardPlusIntegration,
        passes::{
            hiz as hiz_pass,
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
        model_renderer::{
            MaterialPushConstants, ModelRenderer,
        },
        passes,
        resource_registry::{ResourceId, ResourceRegistry},
        resources,
        resources::{
            uniform::{StorageBuffer, UniformBuffer},
        },
        vram_budget, DepthBuffer, GBuffer, HdrSystem, Material, MaterialHandle, MaterialManager, Mesh,
        PipelineCache, Texture, Transform, Scene,
        initialization,
        init_types::*,
        assets::AssetManager,
    },
    vulkan::{self, Allocator, CommandBufferContext},
    AshError, Result,
};

use ash::vk;
use glam::{Mat4, Vec3};
use rayon::prelude::*;
use resources::BufferPool;
use std::collections::{HashMap, HashSet};
use std::ptr;
use std::sync::Arc;
use std::time::Instant;
use std::thread;

use crate::renderer::queue::RenderQueue;
use super::swapchain_manager;
use crate::renderer::resources::buffer::BufferHandle;
use crate::renderer::resources::mesh::{MaterialDescriptor, MeshDescriptor};
use crate::renderer::resources::GlobalClusterBuffer;
use crate::renderer::vcgs::culling::CullObjectData;




// RendererResources moved to init_types.rs

fn compute_worker_index(worker_count: usize, frame_index: usize) -> usize {
    if worker_count == 0 {
        0
    } else {
        frame_index % worker_count
    }
}



#[cfg(test)]
mod tests {
    use super::compute_worker_index;

    #[test]
    fn worker_index_zero_workers() {
        assert_eq!(compute_worker_index(0, 0), 0);
        assert_eq!(compute_worker_index(0, 5), 0);
    }

    #[test]
    fn worker_index_wraps() {
        assert_eq!(compute_worker_index(4, 0), 0);
        assert_eq!(compute_worker_index(4, 3), 3);
        assert_eq!(compute_worker_index(4, 4), 0);
        assert_eq!(compute_worker_index(4, 7), 3);
    }
}







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
    pub(crate) render_pass: Option<vulkan::RenderPass>,
    pub(crate) render_pass_id: Option<ResourceId>,
    /// Dedicated render pass for HDR rendering (ensures format compatibility)
    pub(crate) hdr_render_pass: Option<vulkan::RenderPass>,
    pub(crate) hdr_render_pass_id: Option<ResourceId>,
    pub(crate) pipeline: Option<vulkan::Pipeline>,
    pub(crate) pipeline_id: Option<ResourceId>,
    
    // Skybox Rendering (Modularized)
    skybox_pass: Option<passes::SkyboxPass>,
    
    pub(crate) depth_buffer: Option<DepthBuffer>,
    pub(crate) uniform_buffers: Vec<UniformBuffer>,
    pub(crate) material_storage_buffer: Option<StorageBuffer<resources::uniform::MaterialUniform>>,
    pub(crate) pipeline_layout: Option<vulkan::PipelineLayout>,
    pub(crate) pipeline_layout_id: Option<ResourceId>,
    pub(crate) descriptors: Option<vulkan::DescriptorAllocator>,
    pub(crate) framebuffers: Vec<vulkan::Framebuffer>,
    pub(crate) framebuffer_ids: Vec<ResourceId>,
    start_time: Instant,
    mesh_data: Vec<MeshData>, // Indexed by mesh handle for O(1) access
    uploaded_material_indices: HashSet<u32>, // Track which materials are GPU-resident (UE5 pattern)
    pub(crate) swapchain_image_view_ids: Vec<ResourceId>,
    pub(crate) depth_buffer_id: Option<ResourceId>,
    pub(crate) frame_sync_ids: Vec<(ResourceId, ResourceId, ResourceId)>,
    // Post-processing support
    sample_shading: SampleShadingQuality,
    pub(crate) hdr_system: Option<HdrSystem>,

    // Diagnostics
    diagnostics: DiagnosticsState,
    frame_profiler: FrameProfiler,
    gpu_profiler: Option<GpuProfiler>,
    diagnostics_overlay: DiagnosticsOverlay,
    // Shadow System
    pub(crate) shadow_system: Option<crate::renderer::features::ShadowSystem>,
    // Bindless textures
    pub assets: AssetManager,
    // Forward+ lighting
    pub(crate) forward_plus: Option<ForwardPlusIntegration>,
    // GPU-driven occlusion culling (Hi-Z + Indirect Draw)
    pub(crate) hiz_pass: Option<HiZPass>,
    adaptive_hiz_manager: AdaptiveHiZManager,
    pub(crate) indirect_draw_pass: Option<IndirectDrawPass>,
    // Temporal Super-Resolution
    pub(crate) vsr_pass: Option<VsrPass>,
    // Motion Vector Pass for VSR/TAA
    pub(crate) motion_pass: Option<MotionVectorPass>,
    pub(crate) motion_framebuffer: Option<vk::Framebuffer>,
    // G-Buffer for Normals and Motion Vectors
    pub(crate) gbuffer: Option<GBuffer>,
    // Pipeline optimization
    // Lighting
    pub debug_mode: DebugMode,
    
    // Phase 7: Host-Side Cluster Integration
    global_cluster_buffer: Option<GlobalClusterBuffer>,
    
    // Phase 4: G-Buffer & HDR Indices
    pub(crate) gbuffer_indices: GBufferIndices,
    pub(crate) hdr_image_index: Option<u32>,

    // Post-processing descriptors

    post_process: crate::renderer::systems::post_process::PostProcessSystem,

    vram_budget: vram_budget::VramBudget,
    texture_compression: bool,
    instancing_manager: InstancingManager,
    instance_buffers: Vec<resources::InstanceBuffer>,
    transform_system: resources::TransformSystem,
    // Transient Transform Arena (Phase 19)
    transform_arena: vk::Buffer,
    transform_arena_alloc: vk_mem::Allocation,
    transform_arena_addr: u64,
    transform_arena_offset: u32,
    // Image-Based Lighting
    allow_auto_material: bool,
    strict_mode: bool,
    // Headless support
    readback_buffer: Option<BufferHandle>,
    last_image_index: u32,

    // TAA Configuration
    pub taa_config: TaaConfig,
    /// TAA configuration metrics (tracking changes/validation)
    taa_config_metrics: ConfigMetrics,

    pub vsr_config: VsrConfig,

    // BDA Addresses
    pub instance_buffer_addresses: Vec<u64>,
    pub material_heap_address: u64,

    // Core foundation - dropped in declaration order (Top to Bottom)
    // So these should be at the VERY END to be dropped LAST.
    // However, they are dropped Top to Bottom?
    // WAIT! In Rust, fields are dropped in the order they are DECLARED.
    // So the FIRST field is dropped FIRST.
    // This means foundation should be at the BOTTOM so they are dropped LAST.
    pub(crate) geometry_buffer: Arc<resources::DualHeapGeometryBuffer>,
    pub(crate) resources: Arc<ResourceRegistry>,
    pub alloc: Arc<vulkan::Allocator>,
    pub queue: RenderQueue,
    pub device: vulkan::VulkanDevice,
}


pub struct MainPassParameters<'a> {
    pub cmd_ctx: &'a CommandBufferContext<'a>,
    pub frame_index: usize,
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

    pub fn set_debug_mode(&mut self, mode: DebugMode) {
        self.debug_mode = mode;
        log::info!("Debug mode set to: {mode:?}");
    }

    /// Initializes the renderer.
    pub fn new<S: vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        log::info!("Renderer::new: Starting initialization");
        unsafe {
            log::info!("Creating VulkanInstance");
            let instance = Arc::new(vulkan::VulkanInstance::new(
                surface_provider,
                cfg!(debug_assertions),
            )?);
            log::info!("Creating VulkanDevice");
            let device =
                vulkan::VulkanDevice::new(Arc::clone(&instance), surface_provider.is_headless())?;
            log::info!("Creating Allocator");
            let alloc = Arc::new(vulkan::Allocator::new(&device)?);
            log::info!("Creating ResourceRegistry");
            let resources = Arc::new(ResourceRegistry::new(Arc::clone(&device.device)));
            let dev_mem_props = device.memory_properties;
            let vram_budget = vram_budget::VramBudget::new(&dev_mem_props);

            log::info!("Initializing Features");
            let mut features = FeatureManager::new();
            features.set_device(Arc::clone(&device.device));
            features.add_feature(AutoRotateFeature::new());


            // Initialize Pipeline Cache
            log::info!("Creating PipelineCache");
            let pipeline_cache = PipelineCache::new(Arc::clone(&device.device))?;
            let renderer_config = RendererConfig::default();
            let _texture_compression = renderer_config.texture_compression;
            let mut pipeline_cfg = renderer_config.pipeline.clone();
            
            // Hardware fallback for sample shading
            if !device.sample_rate_shading_supported && pipeline_cfg.sample_shading.enabled() {
                log::warn!("Sample rate shading requested but not supported by hardware. Falling back to disabled.");
                pipeline_cfg.sample_shading = SampleShadingQuality::Disabled;
            }

            log::info!("Creating BufferPool");
            let buffer_pool = Arc::new(BufferPool::new(Arc::clone(&alloc)));
            let (width, height) = surface_provider.physical_size();
            let extent = vk::Extent2D { width, height };
            log::info!("Creating swapchain and depth buffer for extent {}x{}", width, height);
            
            let swapchain_data = 
                initialization::create_swapchain_data(&device, &alloc, extent)?;
            let mut swapchain = swapchain_data.swapchain;
            let render_pass_handle = swapchain_data.render_pass;
            let mut framebuffers = swapchain_data.framebuffers;
            let mut depth_buffer = swapchain_data.depth_buffer;

            log::info!("Registering swapchain resources");
            let mut swapchain_image_view_ids = Vec::with_capacity(swapchain.image_views.len());
            for &view in &swapchain.image_views {
                let id = resources.register_image_view(view)?;
                swapchain_image_view_ids.push(id);
            }
            swapchain.mark_image_views_managed_by_registry();

            let depth_buffer_id = depth_buffer.register_with_registry(&resources)?;
            let render_pass_id = resources.register_render_pass(render_pass_handle)?;

            let mut framebuffer_ids = Vec::with_capacity(framebuffers.len());
            for (idx, fb) in framebuffers.iter_mut().enumerate() {
                 let id = resources.register_framebuffer(fb.handle(), &[
                     render_pass_id,
                     depth_buffer_id,
                     swapchain_image_view_ids[idx],
                 ])?;
                 fb.mark_managed_by_registry();
                 framebuffer_ids.push(id);
            }

            log::info!("Creating expanded RenderQueue and frame resources");
            let worker_count = thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            
            let mut queue = RenderQueue::new(
                Arc::clone(&device.device),
                device.graphics_queue,
                device.present_queue,
                device.graphics_queue_family,
                framebuffers.len(),
                worker_count,
            )?;

            let mut frame_sync_ids = Vec::with_capacity(queue.frame_syncs.len());
            for sync in &mut queue.frame_syncs {
                let image_available_id = resources.register_semaphore(sync.image_available)?;
                let render_finished_id = resources.register_semaphore(sync.render_finished)?;
                let fence_id = resources.register_fence(sync.in_flight)?;
                sync.mark_managed_by_registry();
                frame_sync_ids.push((image_available_id, render_finished_id, fence_id));
            }

            resources.register_command_pool(queue.cmds.upload_command_pool_handle())?;
            queue.cmds.mark_pool_managed_by_registry();
            log::info!("RenderQueue expanded and resources registered successfully");

            let _worker_count = queue.cmds.worker_count();

            log::info!("Initializing GeometryBuffer and ModelRenderer");
            let geometry_buffer = Arc::new(resources::DualHeapGeometryBuffer::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                256, // 256MB for vertices
                128, // 128MB for indices
            )?);
            
            // Phase 7: Global Cluster Buffer (Static)
            let global_cluster_buffer = GlobalClusterBuffer::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                64, // 64MB capacity
            )?;

            let model_renderer =
                ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device.device), Arc::clone(&geometry_buffer));

            log::info!("Initializing DescriptorAllocator and BindlessManager");
            let mut descriptor_allocator = vulkan::DescriptorAllocator::new(
                Arc::clone(&device.device),
                2048, // Equivalent to EXTRA_TEXTURE_SETS previously in DescriptorManager
                Some(Arc::clone(&resources)),
            )?;

            let aspect = swapchain.extent.width as f32 / swapchain.extent.height as f32;

            let mut bindless_manager = crate::vulkan::BindlessManager::new(
                instance.instance(),
                device.physical_device,
                Arc::clone(&device.device),
                &mut descriptor_allocator,
                crate::vulkan::BindlessManager::DEFAULT_MAX_TEXTURES,
                crate::vulkan::BindlessManager::DEFAULT_MAX_PAGE_TABLES,
                crate::vulkan::BindlessManager::DEFAULT_MAX_CUBEMAPS,
                crate::vulkan::BindlessManager::DEFAULT_MAX_STORAGE_IMAGES,
                crate::vulkan::BindlessManager::DEFAULT_MAX_BUFFERS,
            )?;

            log::info!("Initializing Forward+ Integration");
            let mut forward_plus = ForwardPlusIntegration::new(
                Arc::clone(&device.device),
                &*alloc,
                queue.frame_syncs.len() as u32,
            )?;
            forward_plus.init(&alloc);
            forward_plus.on_resize(swapchain.extent.width, swapchain.extent.height);
            
            log::info!("Initializing Renderer Resources (Uniforms, Textures, Materials)");
            let renderer_resources = initialization::init_resources(
                &alloc,
                &device,
                queue.cmds.upload_command_pool_handle(),
                framebuffers.len(),
                aspect,
            )?;
            let RendererResources {
                uniform_buffers,
                default_texture,
                black_texture,
                white_texture,
                default_skybox,
                default_cube_black,
                material_storage_buffer,
                instance_buffers,
                transform_arena,
                transform_arena_alloc,
                post_sampler: _post_sampler,
            } = renderer_resources;

            // CRITICAL: All frame data is now accessed via BDA (push.frame_ptr).
            // Descriptor Set 0 (Frame Data) has been removed.
            // Bindless set (formerly Set 1) is now Set 0.

            // BDA Migration: No longer need to manually register buffers with bindless manager
            let material_heap_address = material_storage_buffer.device_address();
            log::info!("Global material heap BDA: 0x{material_heap_address:X}");

            let mut instance_buffer_addresses = Vec::with_capacity(instance_buffers.len());
            for buffer in &instance_buffers {
                instance_buffer_addresses.push(buffer.device_address());
            }

            // Mandatory Slot 0: Register black texture as the absolute fallback (The Void).
            let black_tex_index = bindless_manager
                .add_sampled_image(black_texture.view(), black_texture.sampler())?;
            log::info!("Registered black texture at bindless index {black_tex_index}");
            if black_tex_index != 0 {
                return Err(AshError::VulkanError(format!("Black texture fallback MUST be at index 0, but got {black_tex_index}")));
            }

            // Register default texture.
            let default_tex_index = bindless_manager
                .add_sampled_image(default_texture.view(), default_texture.sampler())?;
            log::info!("Registered default white texture at bindless index {default_tex_index}");

            log::info!("Configuring Pipeline Set Layouts");
            let set_layouts = [
                bindless_manager.descriptor_set_layout(), // Set 0: Unified Bindless
            ];
            
            // DIAGNOSTIC: Log layout handles for verification
            log::debug!("Main Pipeline Set Layouts: Unified={:?}", 
                set_layouts[0]);

            let (pipeline_layout, pipeline_layout_id, pipeline, pipeline_id) =
                initialization::create_main_pipeline(
                    &device,
                    &resources,
                    swapchain.extent,
                    render_pass_handle,
                    render_pass_id,
                    &set_layouts,
                    &pipeline_cfg,
                    depth_buffer.format(),
                    pipeline_cache.handle(),
                )?;

            let mut render_pass_wrapper = vulkan::RenderPass::from_handle(Arc::clone(&device.device), render_pass_handle);
            render_pass_wrapper.mark_managed_by_registry();

            log::info!("Initializing Shadow System...");
            let shadow_system = match crate::renderer::features::ShadowSystem::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                &mut bindless_manager,
                queue.cmds.upload_command_pool_handle(),
                device.graphics_queue,
                default_vsm_config(),
                queue.frame_syncs.len() as u32,
            ) {
                Ok(system) => {
                    log::info!("Shadow System initialized successfully.");
                    Some(system)
                },
                Err(e) => {
                    log::error!("Failed to initialize Shadow System: {e}");
                    None
                }
            };


            // Mesh data already added to mesh_data Vec above
            let start_time = Instant::now();

            let swapchain_extent = swapchain.extent;

            log::info!("Initializing GBuffer");
            let gbuffer = GBuffer::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                swapchain_extent.width,
                swapchain_extent.height,
            )?;

            forward_plus.init_pipeline(
                Arc::clone(&device.device), 
                black_texture.sampler(),
                depth_buffer.view()
            )?;
            
            // Phase 4: G-Buffer Registration
            let mut gbuffer_indices = GBufferIndices::default();
            
            // First-time registration (G-Buffer Motion)
            gbuffer_indices.motion_index = bindless_manager.add_sampled_image(
                gbuffer.motion_view(),
                default_texture.sampler(),
            )?;
            
            // First-time registration (Depth Buffer)
            gbuffer_indices.depth_index = bindless_manager.add_sampled_image(
                depth_buffer.view(),
                default_texture.sampler(),
            )?;

            log::info!("Registered GBuffer indices: Motion={}, Depth={}", 
                gbuffer_indices.motion_index, gbuffer_indices.depth_index);
            
            // HDR Framebuffer is not yet created (happens in initialize_hdr called by enable_post_processing)
            // But we should initialize indices to 0 or safe defaults.

            log::info!("Renderer initialization complete. Constructing struct.");

            // Create readback buffer if headless
            let readback_buffer = if device.headless {
                let buffer_size = (swapchain.extent.width * swapchain.extent.height * 4) as u64; // R8G8B8A8
                Some(BufferHandle::new_with_flags(
                    Arc::clone(&alloc),
                    buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::Auto,
                    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    Some("Headless Readback Buffer".to_string()),
                )?)
            } else {
                None
            };

            // Initialize Skybox
            log::info!("Initializing Skybox...");
            // 1. Textures (Binding 0)
            bindless_manager.add_sampled_image(
                black_texture.view(),
                black_texture.sampler(),
            )?;

            // 3. Cubemaps (Binding 2)
            let skybox_index = bindless_manager.add_cubemap(
                default_skybox.view(),
                default_skybox.sampler(),
            )?;

            log::info!("Bindless Defaults registered (Skybox: {skybox_index})");

            // Create Skybox Mesh (Unit Cube)

            let (_width, _height) = surface_provider.physical_size();
            
            // Create Skybox Mesh (Unit Cube)
            let skybox_mesh = {
                let mut mesh = crate::renderer::Mesh::create_cube();
                // SCALE FIX: Unit cube is 2m (-1 to 1). Scale to 1000m to avoid clipping.
                for v in &mut mesh.vertices {
                    v.position[0] *= 500.0;
                    v.position[1] *= 500.0;
                    v.position[2] *= 500.0;
                }
                model_renderer.upload_mesh_data(
                    &mesh,
                    queue.cmds.upload_command_pool_handle(),
                    device.graphics_queue
                )?
            };
            log::info!("Skybox initialized.");
            let mesh_data: Vec<MeshData> = Vec::new();
            let instancing_manager = InstancingManager::new();
            let transform_system = resources::TransformSystem::new();
            let config = &renderer_config;
            let swapchain_extent = swapchain.extent;
            let swapchain_format = swapchain.format;
            let swapchain_image_count = framebuffers.len();
            let device_handle = Arc::clone(&device.device);

            // Missing initializations

            // Initialize GPU-driven pipeline components
            log::info!("Initializing Hi-Z Pass");
            let mut hiz_pass = HiZPass::new(Arc::clone(&device.device));
            hiz_pass.init(&alloc.vma, &device, swapchain_extent.width, swapchain_extent.height)?;

            log::info!("Initializing Indirect Draw Pass");
            let mut indirect_draw_pass = IndirectDrawPass::new(Arc::clone(&device.device));
            indirect_draw_pass.init(
                &alloc.vma,
                &device,
                &mut bindless_manager,
                crate::renderer::vcgs::MAX_INDIRECT_OBJECTS,
            )?;


            let transform_arena_addr = device.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(transform_arena));

            // Create SkyboxPass
            log::info!("Creating SkyboxPass...");
            let skybox_pass = passes::SkyboxPass::new(
                &device,
                &resources,
                render_pass_handle,
                swapchain.extent,
                pipeline_cache.handle(),
                depth_buffer.format(),
                pipeline_cfg.multisample_config(),
                &set_layouts,
                skybox_mesh,
                skybox_index,
            )?;

            let mut renderer = Self {
                queue,
                buffer_pool: buffer_pool,
                resources,
                features,
                _pipeline_cache: pipeline_cache,
                prev_view_proj: Mat4::IDENTITY,
                _default_texture: default_texture,
                _black_texture: black_texture,
                _white_texture: white_texture,
                _default_skybox: default_skybox,
                _default_cube_black: default_cube_black,
                draw_items: Vec::new(),
                swapchain: Some(swapchain),
                render_pass: Some(render_pass_wrapper),
                render_pass_id: Some(render_pass_id),
                hdr_render_pass: None,
                hdr_render_pass_id: None,
                pipeline: Some(pipeline),
                pipeline_id: Some(pipeline_id),
                
                skybox_pass: Some(skybox_pass),
                
                depth_buffer: Some(depth_buffer),
                uniform_buffers,
                material_storage_buffer: Some(material_storage_buffer),
                instance_buffer_addresses,
                material_heap_address,
                pipeline_layout: Some(pipeline_layout),
                pipeline_layout_id: Some(pipeline_layout_id),
                descriptors: Some(descriptor_allocator),
                framebuffers,
                framebuffer_ids,
                geometry_buffer,
                start_time,
                alloc,
                device,
                mesh_data,
                uploaded_material_indices: HashSet::new(),
                swapchain_image_view_ids,
                depth_buffer_id: Some(depth_buffer_id),
                frame_sync_ids,

                sample_shading: pipeline_cfg.sample_shading,
                hdr_system: None,

                diagnostics: DiagnosticsState::default(),
                frame_profiler: FrameProfiler::new(),
                gpu_profiler: None,
                diagnostics_overlay: DiagnosticsOverlay::new(),
                shadow_system,
                assets: AssetManager::new(bindless_manager),
                forward_plus: Some(forward_plus),
                hiz_pass: Some(hiz_pass),
                adaptive_hiz_manager: AdaptiveHiZManager::new(3.0), // Target 3ms for Hi-Z
                indirect_draw_pass: Some(indirect_draw_pass),
                vsr_pass: None,
                motion_pass: None,
                motion_framebuffer: None,
                gbuffer: Some(gbuffer),
                debug_mode: DebugMode::default(),
                global_cluster_buffer: Some(global_cluster_buffer),
                gbuffer_indices: GBufferIndices::default(),
                hdr_image_index: None,

                vram_budget,
                post_process: crate::renderer::systems::post_process::PostProcessSystem::new(
                    device_handle,
                    swapchain_image_count,
                    swapchain_extent,
                    swapchain_format,
                )?,
                texture_compression: renderer_config.texture_compression,
                instancing_manager,
                instance_buffers: instance_buffers,
                transform_system,
                // Phase 19 Transient Arena
                transform_arena,
                transform_arena_alloc,
                transform_arena_addr,
                transform_arena_offset: 0,

                allow_auto_material: renderer_config.allow_auto_material,
                strict_mode: config.strict_mode,
                readback_buffer,
                last_image_index: 0,
                taa_config: TaaConfig::default(),
                taa_config_metrics: ConfigMetrics::default(),
                vsr_config: VsrConfig::default(),
            };



            // Initialize motion vector pass
            renderer.init_motion_pass()?;
            
            // Initialize async readback manager


            renderer.queue.pending_extent = Some(swapchain_extent);

            Ok(renderer)
        }
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
        let cmd = self.queue.cmds.get_transfer_command_buffer()?;

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
                .free_command_buffers(self.queue.cmds.upload_command_pool_handle(), &[cmd]);
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
        
        // Create motion framebuffer
        let swapchain = self.swapchain.as_ref().ok_or(AshError::VulkanError(
            "Swapchain not initialized".to_string(),
        ))?;
        self.create_motion_framebuffer(swapchain.extent.width, swapchain.extent.height)?;
        
        log::info!("Motion vector pass initialized");

        Ok(())
    }

    /// Create framebuffer for motion vector pass
    ///
    /// # Safety
    /// GBuffer and motion pass must be initialized
    unsafe fn create_motion_framebuffer(&mut self, width: u32, height: u32) -> Result<()> {
        let gbuffer = self.gbuffer.as_ref().ok_or(AshError::VulkanError(
            "GBuffer not initialized".to_string(),
        ))?;

        let motion_pass = self.motion_pass.as_ref().ok_or(AshError::VulkanError(
            "Motion pass not initialized".to_string(),
        ))?;

        // Destroy old framebuffer if exists
        if let Some(old_fb) = self.motion_framebuffer.take() {
            self.device.device.destroy_framebuffer(old_fb, None);
        }

        let attachments = [gbuffer.motion_view()];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(motion_pass.render_pass())
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = self.device.device.create_framebuffer(&framebuffer_info, None)?;
        self.motion_framebuffer = Some(framebuffer);

        log::debug!("Motion framebuffer created ({width}x{height})");
        Ok(())
    }




    fn worker_index_for_frame(&self, frame_index: usize) -> usize {
        compute_worker_index(self.queue.cmds.worker_count(), frame_index)
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

    pub fn get_mesh_material(&self, mesh_handle: u32) -> MaterialHandle {
        if (mesh_handle as usize) < self.mesh_data.len() {
            self.mesh_data[mesh_handle as usize].material_handle
        } else {
            MaterialHandle::null()
        }
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
    pub fn mesh_data(&self) -> &[MeshData] {
        &self.mesh_data
    }

    /// Get mutable access to mesh data by handle.
    pub fn get_mesh_data_mut(&mut self, handle: u32) -> Option<&mut MeshData> {
        self.mesh_data.get_mut(handle as usize)
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
        self.queue.cmds.get_transfer_command_buffer()
    }

    /// Uploads a single mesh to the GPU with its own transient command buffer.
    /// This is a convenience wrapper for simple cases/examples.
    pub fn upload_mesh_single(&mut self, scene: &mut Scene, mesh: Mesh) -> Result<u32> {
        let upload_cmd = self.get_transfer_command_buffer()?;
        {
            let cmd_ctx = self.queue.cmds.context(upload_cmd);
            cmd_ctx.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        }

        let mut staging_resources = Vec::new();
        let handle = self.upload_mesh(scene, mesh, upload_cmd, &mut staging_resources)?;

        {
            let cmd_ctx = self.queue.cmds.context(upload_cmd);
            cmd_ctx.end()?;
        }
        let cmds = [upload_cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        self.queue.cmds.submit(self.device.graphics_queue, &[submit_info], vk::Fence::null())?;
        
        unsafe {
            self.device.device.queue_wait_idle(self.device.graphics_queue)
                .map_err(|e| AshError::VulkanError(format!("Queue wait: {e}")))?;
        }

        Ok(handle)
    }

    /// Uploads a mesh to the GPU and returns its handle.
    /// This is the modern replacement for `set_mesh`.
    pub fn upload_mesh(
        &mut self,
        scene: &mut Scene,
        mut mesh: Mesh,
        upload_cmd: vk::CommandBuffer,
        staging_resources: &mut Vec<crate::renderer::resources::BufferHandle>,
    ) -> Result<u32> {
        let handle = self.mesh_data.len() as u32;
        self.register_mesh_handle(scene, handle, &mut mesh, upload_cmd, staging_resources)?;
        Ok(handle)
    }

    /// Registers a mesh handle with its own transient command buffer.
    pub fn register_mesh_handle_single(&mut self, scene: &mut Scene, handle: u32, mesh: &mut Mesh) -> Result<()> {
        let upload_cmd = self.get_transfer_command_buffer()?;
        let mut staging_resources = Vec::new();
        
        {
            let ctx = self.queue.cmds.context(upload_cmd);
            ctx.begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
        }
        
        self.register_mesh_handle(scene, handle, mesh, upload_cmd, &mut staging_resources)?;
        
        {
            let ctx = self.queue.cmds.context(upload_cmd);
            ctx.end()?;
        }
        
        let cmds = [upload_cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        self.queue.cmds.submit(self.device.graphics_queue, &[submit_info], vk::Fence::null())?;
        
        unsafe {
            self.device.device.queue_wait_idle(self.device.graphics_queue)
                .map_err(|e| AshError::VulkanError(format!("Queue wait: {e}")))?;
        }
        
        Ok(())
    }

    pub fn register_mesh_handle(
        &mut self,
        scene: &mut Scene,
        handle: u32,
        mesh: &mut Mesh,
        upload_cmd: vk::CommandBuffer,
        staging_resources: &mut Vec<crate::renderer::resources::BufferHandle>,
    ) -> Result<()> {
        // VCGS Phase 2: Build Cluster DAG
        // This generates the hierarchical cluster structure needed for GPU selection.
        // crate::renderer::vcgs::build_mesh_dag(mesh); // DIAGNOSTIC: Disabled to test stability
        
        // Phase 7: Upload Clusters to Global Buffer
        // Phase 7: Upload Clusters to Global Buffer
        let mut cluster_start_index = 0;
        let mut cluster_count = 0;

        // Pre-clone shared resources to avoid self-borrow issues
        let alloc = Arc::clone(&self.alloc);

        // ---------------------------------------------------------
        // PART 1: Texture & Material Registration (MOVED UP)
        // ---------------------------------------------------------
        // We must register materials FIRST so that we get a valid handle
        // to pack into the cluster data.
        
        let key;
        let flags;
        let indices;
        let emissive_index;
        let bounds;
        let material_handle;

        unsafe {
            key = mesh.name.clone();
            let upload_pool = self.queue.cmds.upload_command_pool_handle();
            
            // 1. Ensure Textures
            mesh.ensure_texture(
                Arc::clone(&self.alloc),
                Arc::clone(&self.device.device),
                upload_pool,
                self.device.graphics_queue,
                &mut self.vram_budget,
                self.texture_compression,
            )?;

            // 2. Ensure Model Renderer
            scene.model_renderer
                .ensure_mesh(&key, mesh, upload_pool, self.device.graphics_queue)?;

            // 3. Register Bindless
            register_mesh_textures(mesh, &mut self.assets.bindless_manager, &mut self.assets.texture_registry)?;

            // 4. Register Material
            let mut handle_mat = scene.material_manager.default_material();
            if let Some(props) = &mesh.material_properties {
                if !self.allow_auto_material {
                    log::warn!(
                        "Mesh '{}' has material properties but auto-material creation is disabled.",
                        &*mesh.name
                    );
                } else {
                    let material = Material {
                        name: format!("{}_material_{handle}", &*mesh.name),
                        color: props.base_color_factor,
                        metallic: props.metallic_factor,
                        roughness: props.roughness_factor,
                        emissive: props.emissive_factor,
                        occlusion_strength: props.occlusion_strength,
                        normal_scale: props.normal_scale,
                        alpha_cutoff: props.alpha_cutoff,
                        tint_index: -1,
                        is_transparent: props.base_color_factor[3] < 1.0,
                        texture_index: mesh.texture_index,
                        normal_texture_index: mesh.normal_texture_index,
                        metallic_roughness_texture_index: mesh.metallic_roughness_texture_index,
                        occlusion_texture_index: mesh.occlusion_texture_index,
                        emissive_texture_index: mesh.emissive_texture_index,
                    };
                    
                    let material_clone = material.clone();
                    handle_mat = scene.material_manager.register_material(material);
                    
                    // CRITICAL: Auto-generated materials must be uploaded to GPU!
                    self.upload_material_to_gpu(handle_mat.index as u32, &material_clone)?;
                    
                    log::debug!(
                        "Registered auto-material for mesh '{}': handle={:?}, metallic={:.2}, roughness={:.2}",
                        &*mesh.name, handle_mat, props.metallic_factor, props.roughness_factor
                    );
                }
            }
            material_handle = handle_mat;
            
            // FIX: Update mesh material handle so cluster packing sees it!
            if !material_handle.is_null() {
                mesh.material_handle = Some(material_handle.index as u32);
            }

            flags = TexturePresenceFlags::from_mesh(mesh);

            indices = [
                mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.metallic_roughness_texture_index
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
            ];
            emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

            // Calculate bounding box from mesh vertices
            bounds = if !mesh.vertices.is_empty() {
                let mut min = Vec3::splat(f32::MAX);
                let mut max = Vec3::splat(f32::MIN);
                for vertex in &mesh.vertices {
                    let pos = Vec3::from(vertex.position);
                    min = min.min(pos);
                    max = max.max(pos);
                }
                CullBoundingBox::from_min_max(min, max)
            } else {
                // Fallback to unit cube if no vertices
                CullBoundingBox::new(Vec3::ZERO, Vec3::ONE)
            };
        }

        // ---------------------------------------------------------
        // PART 2: Cluster Upload (Including FIX: Uses valid mesh.material_handle)
        // ---------------------------------------------------------
        let cluster_buffer = self.global_cluster_buffer.as_ref();
        if let Some(buffer) = cluster_buffer {
            if !mesh.clusters.is_empty() {
                // Convert MeshCluster to CullObjectData
                let cull_objects: Vec<CullObjectData> = mesh.clusters.iter().map(|c| {
                    // Start with IDENTITY matrix to prevent geometry squashing
                    let identity = glam::Mat4::IDENTITY;
                    let cols = identity.to_cols_array_2d();

                    let data = CullObjectData {
                        // Pack matrix rows (GPU expects column-major for mat4, which is cols[0..3] in memory)
                        model_row0: cols[0],
                        model_row1: cols[1],
                        model_row2: cols[2],
                        model_row3: cols[3],
                        
                        // Sphere packing: vec4(center.xyz, radius)
                        bounds: crate::renderer::vcgs::CullBoundingBox {
                            center: [c.bounds_center[0], c.bounds_center[1], c.bounds_center[2], c.bounds_radius],
                            extents: [c.bounds_radius, c.bounds_radius, c.bounds_radius, 0.0],
                        },
                        
                        parent_index: c.parent_index,
                        first_index: c.first_index,
                        index_count: c.index_count,
                        error_metric: c.error_metric,
                        
                        // Set Flag 1 (Enabled)
                        flags: 1,
                        
                        // Material from mesh (NOW HAS VALID HANDLE!)
                        material_index: mesh.material_handle.unwrap_or(0),
                        
                        ..Default::default()
                    };
                    
                    data
                }).collect();
                
                unsafe {
                    let element_size = std::mem::size_of::<CullObjectData>();
                    let total_size = (cull_objects.len() * element_size) as u64;

                    let mut staging_buffer = crate::renderer::resources::BufferHandle::new_with_flags(
                        Arc::clone(&alloc),
                        total_size,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        vk_mem::MemoryUsage::Auto,
                        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                            | vk_mem::AllocationCreateFlags::MAPPED,
                        Some(format!("Staging_Clusters_{}", mesh.name)),
                    )
                    .map_err(|e| AshError::VulkanError(format!("Staging buffer alloc: {}", e)))?;

                    // 2. Copy data to staged memory
                    let total_size_val = total_size; // Avoid borrow issues
                    {
                        let mut guard = alloc
                            .map_allocation_guarded(staging_buffer.allocation_mut(), total_size_val)
                            .map_err(|e| {
                                AshError::VulkanError(format!("Map cluster staging memory: {}", e))
                            })?;
                        guard.copy_from_slice(&cull_objects);
                    }

                    // 3. Record Copy Command (No submit, no wait here - we are batching!)
                    let start_idx = buffer.upload_clusters(
                        upload_cmd,
                        staging_buffer.handle(),
                        0,
                        cull_objects.len() as u32,
                    ).map_err(|e| AshError::VulkanError(format!("Cluster upload recording: {}", e)))?;

                    // Store staging buffer to keep it alive until command buffer finishes (Move ownership)
                    staging_resources.push(staging_buffer);

                    cluster_start_index = start_idx;
                    cluster_count = cull_objects.len() as u32;
                    log::debug!(
                        "Recorded {} clusters for batch upload of mesh '{}' at index {}",
                        cluster_count,
                        mesh.name,
                        start_idx
                    );
                }
            }
        }
        // Update mesh tracking
        mesh.cluster_start_index = Some(cluster_start_index);
        mesh.cluster_count = Some(cluster_count);

        // ---------------------------------------------------------
        // PART 3: MeshData Update
        // ---------------------------------------------------------
            let mesh_data = MeshData {
                name: Arc::clone(&key),
                texture_indices: indices,
                emissive_index,
                texture_flags: flags,
                material_handle,
                is_hidden: false,
                bounds,
                cluster_start_index,
                cluster_count,
            };

            if handle as usize >= self.mesh_data.len() {
                self.mesh_data
                    .resize(handle as usize + 1, MeshData::default());
            }
            self.mesh_data[handle as usize] = mesh_data;

        Ok(())
    }

    pub fn get_stats(&self) -> crate::renderer::diagnostics::RendererStats {
        crate::renderer::diagnostics::RendererStats {
            vram_usage: self.vram_budget.get_stats(),
            draw_calls_per_frame: self.diagnostics.frame_stats.draw_calls,
            triangles_rendered: self.diagnostics.frame_stats.triangles,
            cull_efficiency: {
                // Calculate cull efficiency: (potential_draws - actual_draws) / potential_draws
                let potential_draws = (self.diagnostics.frame_stats.triangles / 1000).max(1) as f32;
                let actual_draws = self.diagnostics.frame_stats.draw_calls as f32;
                ((potential_draws - actual_draws) / potential_draws.max(1.0)).clamp(0.0, 1.0)
            },
            gpu_frame_ms: self.diagnostics.gpu_timings.total_ms,
            hiz_quality: format!("{:?}", self.hiz_pass.as_ref().map(|h| h.quality()).unwrap_or(crate::renderer::passes::hiz::HiZQuality::Balanced)),
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
    pub fn upload_material_to_gpu(&mut self, handle: u32, material: &Material) -> Result<()> {
        if let Some(buffer) = self.material_storage_buffer.as_mut() {
            let capacity = buffer.capacity();
            
            // Validate handle within capacity
            if (handle as usize) >= capacity {
                return Err(AshError::VulkanError(format!(
                    "Material handle {handle} exceeds buffer capacity {capacity}"
                )));
            }
            
            // Create MaterialUniform from the material
            let mut mat_uniform = resources::uniform::MaterialUniform::default();
            mat_uniform.set_base_color_factor(glam::Vec4::from_array(material.color));
            mat_uniform.set_emissive_factor(glam::Vec4::from_array(material.emissive));
            mat_uniform.set_metallic_roughness(material.metallic, material.roughness);
            mat_uniform.set_occlusion_strength(material.occlusion_strength);
            mat_uniform.set_normal_scale(material.normal_scale);
            mat_uniform.set_alpha_cutoff(material.alpha_cutoff);
            
            // Enable texture access by mapping indices from the material
            let base_idx = material.texture_index.unwrap_or(u32::MAX) as i32;
            let normal_idx = material.normal_texture_index.unwrap_or(u32::MAX) as i32;
            let mr_idx = material
                .metallic_roughness_texture_index
                .unwrap_or(u32::MAX) as i32;
            let occ_idx = material.occlusion_texture_index.unwrap_or(u32::MAX) as i32;
            let emissive_idx = material.emissive_texture_index.unwrap_or(u32::MAX) as i32;

            // Validate texture indices before upload
            let max_res = vulkan::BindlessManager::DEFAULT_MAX_TEXTURES;
            resources::bindless_validator::BindlessValidator::validate_indices(
                &[base_idx, normal_idx, mr_idx, occ_idx, emissive_idx],
                max_res,
            )?;

            mat_uniform.set_texture_indices(
                base_idx,
                normal_idx,
                mr_idx,
                occ_idx,
                emissive_idx,
                material.tint_index,
            );

            // AAA PATTERN: Direct streaming write instead of read-modify-write
            // This avoids:
            // 1. GPU stalls from reading entire buffer back to CPU
            // 2. Race conditions where GPU reads old data while CPU writes
            // 3. Unnecessary flush of unmodified elements
            unsafe {
                buffer.write_element_at(handle as usize, &mat_uniform)?;
            }

            // Track material as GPU-resident (UE5 pattern)
            self.uploaded_material_indices.insert(handle);

            // Synchronization handled by memory barrier in render_frame()
            // No need for device_wait_idle - this was a workaround
            // UE5/RAGE pattern: explicit barriers only, no global waits

            // Log the material data we just uploaded for debugging
            log::warn!(
                "âœ“ Material[{}] Streamed -> color={:?}, metallic={:.2}, roughness={:.2}",
                handle,
                material.color,
                material.metallic,
                material.roughness
            );

            #[cfg(debug_assertions)]
            {
                // Verify material was uploaded correctly (Unity/UE5 pattern)
                if let Some(buffer) = self.material_storage_buffer.as_ref() {
                    let test_mat = unsafe { buffer.read_element_at(handle as usize) };
                    log::debug!(
                        "Material[{}] verification: base_color={:?}, metallic={:.2}, roughness={:.2}",
                        handle,
                        test_mat.base_color_factor,
                        test_mat.parameters.x,
                        test_mat.parameters.y
                    );
                }
            }

            Ok(())
        } else {
            Err(AshError::VulkanError("Material storage buffer not initialized".to_string()))
        }
    }
    
    /// Synchronizes all materials from MaterialManager to the GPU buffer.
    pub fn sync_materials_to_gpu(&mut self, scene: &mut super::Scene) -> Result<()> {
        let sync_list: Vec<(u32, resources::Material)> = {
            scene.material_manager.iter_unsynced(&self.uploaded_material_indices)
                .map(|(id, material)| (id, material.clone()))
                .collect()
        };

        for (handle, material) in sync_list {
            self.upload_material_to_gpu(handle, &material)?;
        }

        Ok(())
    }

    /// Standardized material registration and upload helper.
    /// 
    /// This handles both registering the material with the manager and 
    /// uploading its data to the GPU in a single call.
    pub fn register_and_upload_material(&mut self, scene: &mut Scene, material: Material) -> Result<MaterialHandle> {
        let handle = scene.material_manager.register_material(material.clone());
        self.upload_material_to_gpu(handle.index as u32, &material)?;
        Ok(handle)
    }

    /// Get access to the material manager (for testing)
    pub fn material_manager<'a>(&self, scene: &'a Scene) -> &'a MaterialManager {
        &scene.material_manager
    }

    /// Registers mesh data described by a [`MeshDescriptor`] with the renderer and returns the
    /// internal key used for lookup.
    pub fn register_mesh_descriptor(
        &mut self,
        scene: &mut Scene,
        handle: u32,
        descriptor: &MeshDescriptor,
        upload_cmd: vk::CommandBuffer,
        staging_resources: &mut Vec<crate::renderer::resources::BufferHandle>,
    ) -> Result<String> {
        let mut mesh = Mesh::from_descriptor(descriptor);
        let key = Arc::clone(&mesh.name);

        self.register_mesh_handle(scene, handle, &mut mesh, upload_cmd, staging_resources)?;

        Ok(key.to_string())
    }

    /// AAA-grade transient transform upload (Phase 19).
    /// Offloads heavy matrices to a storage buffer via BDA.
    pub fn upload_transform(&mut self, model: Mat4) -> Result<u32> {
        let size = std::mem::size_of::<Mat4>() as u32;
        
        // Ensure 64-byte alignment (standard for mat4)
        debug_assert!(self.transform_arena_offset % 64 == 0);
        
        let index = self.transform_arena_offset / size;

        // Check for overflow (1MB limit) - UE5/RAGE safety pattern
        if self.transform_arena_offset + size > 1024 * 1024 {
             log::error!("Transform arena overflow (1MB)! Skipping transform upload for this object.");
             return Err(AshError::TransformArenaOverflow);
        }

        unsafe {
            // AAA Pattern: Use persistent mapping (assigned during create_buffer with MAPPED flag)
            let mapping = self.alloc.vma.get_allocation_info(&self.transform_arena_alloc);
            let ptr = mapping.mapped_data as *mut u8;
            
            if !ptr.is_null() {
                let dst = ptr.add(self.transform_arena_offset as usize);
                
                ptr::copy_nonoverlapping(
                    model.as_ref().as_ptr() as *const u8,
                    dst,
                    size as usize
                );
                
                // Flush only the modified range to ensure GPU visibility
                let _ = self.alloc.vma.flush_allocation(
                    &self.transform_arena_alloc,
                    self.transform_arena_offset as u64,
                    size as u64
                );
            } else {
                log::error!("Transform arena NOT mapped! Visuals will be broken.");
            }
        }

        self.transform_arena_offset += size;
        Ok(index)
    }

    /// Converts a material descriptor into a renderer material and registers it.
    pub fn register_material_descriptor(
        &mut self,
        scene: &mut Scene,
        _handle: u32,
        descriptor: &MaterialDescriptor,
    ) -> MaterialHandle {
        let material = descriptor.material.clone();
        scene.material_manager.register_material(material)
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
            let mesh_data = &self.mesh_data;
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
                                .push(instance);
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
                if let Some(mesh_data) = self.mesh_data.get(command.mesh_handle as usize) {
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
            .pipeline_layout
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Pipeline layout missing".to_string()))?
            .handle();
        let render_pass = if self.hdr_system.is_some() {
             self.hdr_render_pass
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("HDR Render pass missing".to_string()))?
                .handle()
        } else {
             self.render_pass
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("Render pass missing".to_string()))?
                .handle()
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

        let mut builder = vulkan::Pipeline::builder(Arc::clone(&self.device.device))
            .with_layout(layout)
            .with_render_pass(render_pass)
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
        let render_pass_id = self.render_pass_id.ok_or_else(|| {
            AshError::VulkanError("Render pass ID missing during recreation".into())
        })?;

        let pipeline_id = self
            .resources
            .register_pipeline(new_pipeline.pipeline, &[pipeline_layout_id, render_pass_id])
            .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;

        new_pipeline.mark_managed_by_registry();
        self.pipeline = Some(new_pipeline);
        self.pipeline_id = Some(pipeline_id);

        log::info!("Pipeline recompiled successfully!");
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

    pub(crate) fn create_render_pass_and_framebuffers(
        &mut self,
        extent: vk::Extent2D,
        color_format: vk::Format,
        image_views: &[vk::ImageView],
    ) -> Result<()> {
        // Cleanup already done by cleanup_framebuffers() and cleanup_render_pass()

        let depth_buffer = self.depth_buffer.as_ref().ok_or_else(|| {
            AshError::VulkanError("Depth buffer missing when rebuilding framebuffers".into())
        })?;

        let mut builder = vulkan::RenderPass::builder(Arc::clone(&self.device.device));

        let render_to_hdr = self.hdr_system.is_some();
        if render_to_hdr {
            let hdr = self
                .hdr_system
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("HDR system missing".to_string()))?;
            builder = builder
                .with_color_attachment(hdr.format(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        } else {
            let final_layout = if self.device.headless {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::PRESENT_SRC_KHR
            };
            builder = builder.with_swapchain_color(color_format, final_layout);
        }

        if self.gbuffer.is_some() {
            // Index 1: Normals (RGBA16F)
            builder = builder.with_color_attachment(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            // Index 2: Albedo (RGBA8)
            builder = builder.with_color_attachment(
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            // Index 3: Motion Vectors (RG16F)
            builder = builder.with_color_attachment(
                vk::Format::R16G16_SFLOAT,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        let mut render_pass = builder
            .with_depth_attachment(
                depth_buffer.format(),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )
            .build()?;

        let render_pass_id = self
            .resources
            .register_render_pass(render_pass.handle())
            .map_err(|e| AshError::VulkanError(format!("Failed to register render pass: {e}")))?;
        render_pass.mark_managed_by_registry();
        
        // If we are in HDR mode, this pass is the HDR pass.
        // If we are in Swapchain mode, it's the standard pass.
        // To fix format mismatches, we ensure we have a valid handle in the correct field.
        if render_to_hdr {
            self.hdr_render_pass = Some(render_pass);
            self.hdr_render_pass_id = Some(render_pass_id);
            // We also need a swapchain-compatible pass for post-processing/UI, 
            // but create_render_pass_and_framebuffers is usually called when swapchain changes.
            // For now, if render_pass is missing, we create a default one too.
            if self.render_pass.is_none() {
                 let sw_builder = vulkan::RenderPass::builder(Arc::clone(&self.device.device));
                 let final_layout = if self.device.headless {
                     vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                 } else {
                     vk::ImageLayout::PRESENT_SRC_KHR
                 };
                 let sw_pass = sw_builder.with_swapchain_color(color_format, final_layout)
                     .with_depth_attachment(depth_buffer.format(), vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                     .build()?;
                 let sw_id = self.resources.register_render_pass(sw_pass.handle())?;
                 
                 // CRITICAL FIX: Mark render pass as managed to prevent double-free
                 let mut pass = sw_pass;
                 pass.mark_managed_by_registry();
                 
                 self.render_pass = Some(pass);
                 self.render_pass_id = Some(sw_id);
            }
        } else {
            self.render_pass = Some(render_pass);
            self.render_pass_id = Some(render_pass_id);
        }

        let depth_buffer_id = self.depth_buffer_id.ok_or_else(|| {
            AshError::VulkanError("Depth buffer id missing while rebuilding framebuffers".into())
        })?;

        let mut framebuffers = Vec::with_capacity(image_views.len());
        let mut framebuffer_ids = Vec::with_capacity(image_views.len());

        let gbuffer = self.gbuffer.as_ref().ok_or_else(|| {
            AshError::VulkanError("G-Buffer missing when rebuilding framebuffers".into())
        })?;

        for (index, &view) in image_views.iter().enumerate() {
            // If rendering to HDR, we use a single HDR view for all framebuffers.
            // Otherwise we use the per-swapchain view.
            let color_view = if render_to_hdr {
                self.hdr_system
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("HDR system missing".to_string()))?
                    .view()
            } else {
                view
            };

            // Order must match RenderPass: [Color, GBufferNormal, GBufferAlbedo, GBufferMotion, Depth]
            let attachments = [
                color_view,
                gbuffer.normal_view(),
                gbuffer.albedo_view(),
                gbuffer.motion_view(),
                depth_buffer.view(),
            ];
            
            // CRITICAL FIX: Use the correct render pass for framebuffer creation.
            // When HDR is enabled, framebuffers must be created against hdr_render_pass,
            // not the swapchain render_pass, to ensure format compatibility.
            let active_render_pass = if render_to_hdr {
                self.hdr_render_pass
                    .as_ref()
                    .expect("HDR render pass just created")
                    .handle()
            } else {
                self.render_pass
                    .as_ref()
                    .expect("render pass just created")
                    .handle()
            };
            
            let framebuffer = vulkan::Framebuffer::new(
                Arc::clone(&self.device.device),
                active_render_pass,
                &attachments,
                extent,
            )?;

            let deps = if render_to_hdr {
                vec![render_pass_id, depth_buffer_id]
            } else {
                vec![
                    render_pass_id,
                    depth_buffer_id,
                    self.swapchain_image_view_ids[index],
                ]
            };

            let framebuffer_id = self
                .resources
                .register_framebuffer(framebuffer.handle(), &deps)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register framebuffer: {e}"))
                })?;

            let mut framebuffer = framebuffer;
            framebuffer.mark_managed_by_registry();
            framebuffers.push(framebuffer);
            framebuffer_ids.push(framebuffer_id);
        }

        self.framebuffers = framebuffers;
        self.framebuffer_ids = framebuffer_ids;

        // --- Post-Processing System Resize ---
        self.post_process.resize(image_views, extent)?;

        Ok(())
    }

    pub(crate) fn recreate_frame_syncs(&mut self, count: usize) -> Result<()> {
        for (image_available_id, render_finished_id, fence_id) in self.frame_sync_ids.drain(..) {
            if let Err(e) = self.resources.cleanup_resource(image_available_id) {
                log::warn!("Failed to cleanup image-available semaphore: {e}");
            }
            if let Err(e) = self.resources.cleanup_resource(render_finished_id) {
                log::warn!("Failed to cleanup render-finished semaphore: {e}");
            }
            if let Err(e) = self.resources.cleanup_resource(fence_id) {
                log::warn!("Failed to cleanup in-flight fence: {e}");
            }
        }

        self.queue.frame_syncs.clear();

        let mut frame_syncs = unsafe {
            initialization::create_frame_syncs_internal(&self.device.device, count)?
        };
        let mut frame_sync_ids = Vec::with_capacity(count);

        for sync in &mut frame_syncs {
            let image_available_id = self
                .resources
                .register_semaphore(sync.image_available)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to register image-available semaphore: {e}"
                    ))
                })?;
            let render_finished_id = self
                .resources
                .register_semaphore(sync.render_finished)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to register render-finished semaphore: {e}"
                    ))
                })?;
            let fence_id = self.resources.register_fence(sync.in_flight).map_err(|e| {
                AshError::VulkanError(format!("Failed to register in-flight fence: {e}"))
            })?;

            sync.mark_managed_by_registry();
            frame_sync_ids.push((image_available_id, render_finished_id, fence_id));
        }

        self.queue.frame_syncs = frame_syncs;
        self.frame_sync_ids = frame_sync_ids;
        self.queue.current_frame = 0;

        Ok(())
    }

    pub(crate) fn recreate_command_buffers(&mut self) -> Result<()> {
        self.queue.cmds
            .reset_primary_pool(vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;

        self.queue.command_buffers = self
            .queue.cmds
            .allocate_primary_buffers(self.framebuffers.len() as u32)?;
        self.queue.current_frame = 0;

        Ok(())
    }

    pub(crate) fn recreate_uniform_buffers(&mut self, count: usize) -> Result<()> {
        for ub in &mut self.uniform_buffers {
            let _ = ub.cleanup();
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
                self.uniform_buffers.push(buffer);
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
            let _count = self.queue.frame_syncs.len() as u32;
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
        if let Some(indirect_pass) = self.indirect_draw_pass.as_mut() {
            // No longer populating objects here; they were already populated in render_frame
            // with high-fidelity clusters.

            // 2. Upload and Execute Culling
            log::debug!("Occlusion culling object count: {}", params.scene.occlusion_culling.object_count());
            if params.scene.occlusion_culling.object_count() > 0 {
                let _frame_address = self.uniform_buffers[frame_index].device_address();
                
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
                        &[],
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
                        &[],
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
        let scene_pipeline = params.scene_pipeline;
        let pipeline_layout_handle = params.pipeline_layout_handle;
        let _batch_offsets = params.batch_offsets;
        let _view = params.view;
        let _projection = params.projection;
        let _swapchain_extent = params.swapchain_extent;

        // Resolve global debug state
        let debug_enabled = match self.debug_mode {
            DebugMode::None => false,
            _ => true,
        };

        // Modern Phase 4: Full GPU-Driven Indirect Draw (Opaque Path)
        if let Some(indirect_pass) = self.indirect_draw_pass.as_mut() {
            // Note: Culling and Upload (Compute) is done in cull_main_pass() BEFORE the render pass.

            // 3. Draw All Visible Objects in One Call
            if params.scene.occlusion_culling.object_count() > 0 {
                let frame_address = self.uniform_buffers[frame_index].device_address();
                let _ = frame_address; // Suppress unused warning as it is used in ctx below

                cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, scene_pipeline);

                // CRITICAL FIX: Bind Descriptor Sets
                // Missing these was causing the "Invisible Mesh" issue (shaders had no resources)
                let _descriptors = self.descriptors.as_ref().ok_or(AshError::VulkanError("Descriptors not initialized".into()))?;
                
                // Use public accessor methods instead of private fields
                // frame_set returns Option<vk::DescriptorSet>
                let bindless_set = self.assets.bindless_manager.descriptor_set();

                let sets = [bindless_set];
                
                unsafe {
                    self.device.device.cmd_bind_descriptor_sets(
                        cmd_ctx.handle(),
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout_handle,
                        0,
                        &sets,
                        &[],
                    );
                }

                // CRITICAL FIX: BDA Pointer Validation
                // Prevent crash if buffers are not yet allocated
                let vertex_ptr = params.scene.model_renderer.geometry_buffer.vertex_heap_address();
                let index_ptr = params.scene.model_renderer.geometry_buffer.index_heap_address();
                let instance_ptr = indirect_pass.object_buffer_address();
                let material_ptr = self.material_heap_address;

                if vertex_ptr == 0 || index_ptr == 0 || instance_ptr == 0 || material_ptr == 0 {
                    log::error!(
                        "CRITICAL: BDA Null Pointer - Skipping Draw. Vtx: {:#X}, Idx: {:#X}, Inst: {:#X}, Mat: {:#X}", 
                        vertex_ptr, index_ptr, instance_ptr, material_ptr
                    );
                    return Ok(());
                }

                let material_push = MaterialPushConstants::new(MaterialHandle::null())
                    .with_receive_shadows(true)
                    .with_debug_visualization(debug_enabled);

                // Note: This code path uses indirect draw, not direct mesh rendering
                // The uploaded field in DrawContext is not used for indirect drawing
                // (all data is pulled via BDA), but the struct requires it.
                // We use any available mesh from the cache as a placeholder.
                let Some(uploaded) = params.scene.model_renderer.uploaded_meshes().next().map(|(_, mesh)| mesh) else {
                    // If no meshes are uploaded yet, we can't proceed with indirect draw
                    log::warn!("No uploaded meshes available for indirect draw context");
                    return Ok(());
                };

                let (width, height) = match self.swapchain.as_ref() {
                    Some(sw) => (sw.extent.width, sw.extent.height),
                    None => (1, 1),
                };
                let _ = (width, height); 

                let draw_ctx = crate::renderer::model_renderer::DrawContext {
                    command_buffer: cmd_ctx.handle(),
                    pipeline_layout: pipeline_layout_handle,
                    uploaded,
                    material: &material_push,
                    frame_ptr: self.uniform_buffers[frame_index].device_address(),
                    vertex_ptr,
                    instance_ptr,
                    material_ptr,
                    index_ptr,
                    light_ptr: params.light_ptr,
                    tile_ptr: params.tile_ptr,
                    skybox_index: params.scene.skybox_texture_index,
                    vsm_page_index: self.shadow_system.as_ref().map(|s| s.vsm_page_index()).unwrap_or(0),
                    vsm_cache_index: self.shadow_system.as_ref().map(|s| s.vsm_cache_index()).unwrap_or(0),
                    transform_ptr: self.transform_arena_addr,
                    transform_index: 0, // Using index 0 for main indirect pass
                };
                
                let count_params = crate::renderer::model_renderer::IndirectDrawCountParams {
                    indirect_buffer: indirect_pass.indirect_buffer(),
                    indirect_offset: 0,
                    count_buffer: indirect_pass.count_buffer(),
                    count_offset: 0,
                    max_draw_count: params.scene.occlusion_culling.object_count() as u32,
                    stride: std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
                };

                unsafe {
                    params.scene.model_renderer.draw_indirect_count(&draw_ctx, &count_params);
                }
            }
        }

        // --- PHASE 4: STRICT MODERN - Legacy paths removed ---
        // (Only skinned meshes would remain here if we hadn't moved them, 
        // but for now we focus on opaque stability)

        // 4. Render Skybox (Sentinel Pattern: Only if ibl_prefilter_index is NOT MAX)
        if params.scene.scene_lighting.ibl_prefilter_index >= 0 {
            if let Err(e) = self.render_skybox(&cmd_ctx, frame_index, params.view, params.projection) {
                log::warn!("Skybox render failed: {e}");
            }
        }

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
            let frame_ptr = self.uniform_buffers[frame_index].device_address();
            
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
            .record_change(self.queue.current_frame as u64);

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
        self.queue.current_frame = 0;
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

    pub fn render_frame(
        &mut self,
        scene: &mut super::Scene,
        view: Mat4,
        projection: Mat4,
        camera_pos: glam::Vec3,
        model_matrix: Option<Mat4>,
    ) -> Result<()> {
        self.transform_system.update();
        
        // Task 3: Simplified resize handling
        if self.queue.is_resize_pending() {
            crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene)?;
        }
        self.queue.flush_old_swapchains(&self.device);

        // Phase 19: Reset Transform Arena for the new frame
        self.transform_arena_offset = 0;

        // Phase 20: Sync Materials to GPU
        if let Err(e) = self.sync_materials_to_gpu(scene) {
            log::error!("Failed to sync materials to GPU: {e}");
        }

        // Recycle per-frame descriptor pools (static pools are unaffected)
        if let Some(dm) = self.descriptors.as_mut() {
            dm.next_frame();
        }



        // Hot-reload shaders if changed (throttled to every ~1 second)
        const SHADER_CHECK_INTERVAL: usize = 60;

        // Ensure mutable borrow of pipeline scope ends prior to recreation call.
        let shaders_changed = if self.queue.current_frame % SHADER_CHECK_INTERVAL == 0 {
            if let Some(pipeline) = &mut self.pipeline {
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



        log::debug!(
            "Frame {}: Material synchronization complete",
            self.queue.current_frame
        );

        // Task 3: Simplified resize handling logic moved to start of frame

        unsafe {
            let swapchain_extent = self
                .swapchain
                .as_ref()
                .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
                .extent;
            let scene_pipeline = self
                .pipeline
                .as_ref()
                .map(|p| p.pipeline)
                .ok_or(AshError::VulkanError("Pipeline not available".to_string()))?;
            let main_render_pass = if self.hdr_system.is_some() {
                self.hdr_render_pass
                .as_ref()
                .map(|p| p.handle())
                .ok_or_else(|| AshError::VulkanError("HDR render pass not available".to_string()))?
            } else {
                self.render_pass
                .as_ref()
                .map(|p| p.handle())
                .ok_or_else(|| AshError::VulkanError("Render pass not available".to_string()))?
            };

            // Phase 1: Use RenderQueue to acquire next frame and synchronization objects
            let frame_index = self.queue.current_frame;
            let swapchain_ref = self.swapchain.as_ref().ok_or(AshError::VulkanError("Swapchain not available".to_string()))?;
            let (image_index, command_buffer, image_available, render_finished, in_flight_fence) = self.queue.acquire_next_frame(swapchain_ref)?;



            // Build object registry once per frame

            // Prepare culling data for this frame
            scene.occlusion_culling.begin_frame();
            for (i, item) in self.draw_items.iter().enumerate() {
                if let Some(uploaded) = scene.model_renderer.get(&item.key) {
                    // Use mesh clusters for fine-grained culling
                    // Fallback to mesh bounds if clusters are empty
                    let bounds = self
                        .mesh_data
                        .get(item.mesh_id as usize)
                        .map(|m| m.bounds)
                        .unwrap_or_else(|| CullBoundingBox::new(Vec3::ZERO, Vec3::ONE * 100.0));
                    let material_index = item.material_handle.index as u32;
                    let vertex_offset = uploaded.vertex_offset.unwrap_or(0) as i32 / 64; // 64 bytes per vertex
                    
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


            // Apply sub-pixel jitter for VSR/VSR if enabled
            let mut jittered_projection = projection;
            let mut jitter_uv = [0.0f32; 2];
            if let Some(ref mut vsr) = self.vsr_pass {
                let (jx, jy) = vsr.next_jitter();
                // Jitter is in pixels [-0.5, 0.5], convert to NDC/UV
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
                // SAFETY: frame_index validity verified by previous unchecked access logic.
                let uniform_buffer = self.uniform_buffers.get_unchecked_mut(frame_index);

                // Create dummy transform for legacy feature compatibility
                let mut dummy_transform = resources::Transform::identity();

                let elapsed = self.start_time.elapsed().as_secs_f32();
                let mut feature_ctx = FeatureFrameContext {
                    device: self.device.device.as_ref(),
                    descriptor_allocator: self.descriptors.as_ref(),
                    transform: &mut dummy_transform, // Use dummy
                    auto_rotate: false, // Auto-rotate now handled by examples
                    elapsed_seconds: elapsed,
                };
                self.features.before_frame(&mut feature_ctx);

                // Update VSM clipmap centers and page manager
                if let Some(shadow_system) = &mut self.shadow_system {
                    shadow_system.vsm_feature_mut().begin_frame(frame_index as u32, camera_pos);
                }

                // Matrices provided via function arguments.
                let matrices = uniform_buffer.matrices_mut();
                // Use pre-calculated normal matrix for the provided model matrix
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

                // Phase 2: Lean Engine "Studio Architecture" Logic
                // Single Source of Truth: environment map indices drive shader logic.
                // Indices < 0 indicate no IBL/Environment map is bound.

                matrices.set_lighting(&scene.scene_lighting);

                // Set light-space matrix for shadow mapping
                let light_space_matrix = glam::Mat4::IDENTITY; // VSM uses internal matrices
                matrices.set_light_space_matrix(light_space_matrix);
                // REDUNDANT OVERWRITE REMOVED: Using pre-calculated normal matrix from transform_system
                // matrices.normal_matrix = matrices.model.inverse().transpose();

                let view_proj = matrices.view_proj;
                uniform_buffer.update()?;
                self.prev_view_proj = view_proj;
            }

            // Matrix updates complete.
            // Note: global_barrier moved inside command buffer recording block below for BDA/Uniform safety.

            let device_arc = Arc::clone(&self.device.device);
            let cmd_ctx = CommandBufferContext::new(device_arc.as_ref(), command_buffer);
            cmd_ctx.reset()?;

            log::debug!(
                "Frame {}: Using image index {}",
                self.queue.current_frame,
                image_index
            );

            let worker_index = self.worker_index_for_frame(frame_index);
            debug_assert!(
                worker_index < self.queue.cmds.worker_count().max(1),
                "worker index {} out of bounds for {} workers",
                worker_index,
                self.queue.cmds.worker_count()
            );

            cmd_ctx.begin(vk::CommandBufferUsageFlags::empty())?;

            // CRITICAL: Ensure all host-written buffers (including bindless storage buffers) are visible to GPU
            // This is required because examples might update buffers directly on the host.
            // Moved inside cmd_ctx block to fix DEVICE_LOST crash on Intel Arc/Discrete GPUs.
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

            // Phase 10: Execute Hi-Z construction (Moved here to be inside command buffer recording)
            if let (Some(ref mut hiz), Some(depth_buffer)) =
                (&mut self.hiz_pass, &self.depth_buffer)
            {
                // Update adaptive quality based on previous frame's metrics
                if let Some(ref mut profiler) = self.gpu_profiler {
                    let timings = profiler.last_extended_timings();
                    if timings.valid {
                        let hiz_time_ms = timings.hiz_generate_ms as f64;
                        let new_quality: Option<hiz_pass::HiZQuality> = self.adaptive_hiz_manager.update(hiz.quality(), hiz_time_ms);
                        if let Some(new_quality) = new_quality {
                            hiz.set_quality(new_quality);
                        }
                    }
                }

                // Hiz pyramid is built from previous frame's depth
                hiz.build_pyramid(command_buffer, depth_buffer.image())?;
                
                // MODERN FIX: The "Indestructible" Descriptor Guard
                let hiz_view = if let Some(view) = hiz.hiz_view() {
                    view
                } else {
                    self._black_texture.view()
                };

                let hiz_sampler = if hiz.is_initialized() { 
                    hiz.hiz_sampler() 
                } else { 
                    self._black_texture.sampler() 
                };

                if let Some(ref mut indirect) = self.indirect_draw_pass {
                    indirect.update_hiz_descriptor(hiz_view, hiz_sampler);
                }
                
                if let Some(ref profiler) = self.gpu_profiler {
                    profiler.write_timestamp(command_buffer, crate::renderer::diagnostics::TimingScope::HiZGenerateEnd);
                }
            }

            // --- Phase 8: Instance Data Preparation ---
            // 1. Prepare and upload all instances to the InstanceBuffer
            let mut all_instances = Vec::new();
            let mut batch_offsets = HashMap::new();
            {
                for batch in self.instancing_manager.batches() {
                    batch_offsets.insert(batch.key.clone(), all_instances.len() as u32);
                    all_instances.extend_from_slice(&batch.instances);
                }
            }
            if !all_instances.is_empty() {
                self.instance_buffers[frame_index].update(&all_instances)?;
            }

            if let Some(shadow_system) = &mut self.shadow_system {
                    // ROBUST CHECK: Do not panic if pipeline failed to build.
                    // Just skip shadows for this frame
                    if shadow_system.vsm_feature().shadow_pipeline_layout().is_some() {
                        // Set 2 is gone. VSM resources are now in Set 1 (Bindless)
                        // and indices are passed via push constants in draw_context.

                        // VSM GPU-Driven Shadow Pass
                        let light_dir = glam::Vec3::from_slice(&scene.scene_lighting.directional.direction[0..3]);
                        let frame_descriptor_set = vk::DescriptorSet::null();
                        let bindless_descriptor_set = self.assets.bindless_manager.descriptor_set();

                        let light_ptr = self.forward_plus.as_ref().map(|fp| fp.get_lights().light_ptr(frame_index)).unwrap_or(0);
                        let tile_ptr = self.forward_plus.as_ref().map(|fp| fp.get_lights().tile_ptr(frame_index)).unwrap_or(0);

                        let vertex_ptr = scene.model_renderer.geometry_buffer.vertex_heap_address();
                        let index_ptr = scene.model_renderer.geometry_buffer.index_heap_address();

                        if vertex_ptr == 0 || index_ptr == 0 {
                            log::warn!("Shadow pass: Invalid BDA pointers (V: {}, I: {}). Skipping.", vertex_ptr, index_ptr);
                        } else {
                            shadow_system.vsm_feature_mut().render_shadows(
                                command_buffer,
                                light_dir,
                                all_instances.len() as u32,
                                frame_index,
                                frame_descriptor_set,
                                bindless_descriptor_set,
                                self.uniform_buffers[frame_index].device_address(),
                                vertex_ptr,
                                self.instance_buffer_addresses[frame_index],
                                self.material_heap_address,
                                index_ptr,
                                light_ptr,
                                tile_ptr,
                                self.transform_arena_addr,
                                0, // Using index 0 for shadows for now + 1MB is huge anyway
                            );
                        }
                    } else {
                        log::warn!("Shadow pipeline not ready, skipping shadow pass.");
                    }
                }

            // --- VSM TO MAIN PASS SYNCHRONIZATION ---
            // Barrier to ensure all shadow writes are visible to the main pass
            if let Some(shadow_system) = &self.shadow_system {
                let vsm_barrier = vk::ImageMemoryBarrier::default()
                    .image(shadow_system.vsm_feature().resources.physical_cache)
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

                self.device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vsm_barrier],
                );
            }

            // --- Light Culling Compute Dispatch ---
            if let Some(ref mut fp_integration) = self.forward_plus {
                // Ensure pipeline is initialized before dispatching
                if fp_integration.is_enabled() {
                    // Update light data from scene before uploading
                    fp_integration.update_lights(&scene.point_lights, &scene.directional_lights, &scene.spot_lights);

                    // Update GPU buffers and descriptors for the current frame
                    fp_integration.upload_to_gpu(&self.alloc, &self.device.device, frame_index as usize)?;

                    // Update camera buffer with current view/projection matrices
                    // We use the NON-JITTERED projection for culling to match frustum
                    fp_integration.update_camera(
                        &self.alloc,
                        frame_index as usize,
                        &view.to_cols_array_2d(),
                        &projection.to_cols_array_2d(),
                        &camera_pos.extend(1.0).to_array(),
                    )?;

                    // Dispatch the compute shader
                    fp_integration.dispatch(
                        command_buffer,
                        &self.device.device,
                        frame_index as usize
                    );

                    // Barrier: Ensure light buffers are ready for the fragment shader
                    let light_barrier = vk::BufferMemoryBarrier::default()
                        .buffer(fp_integration.lights().get_light_buffer(frame_index as usize).unwrap_or(vk::Buffer::null()))
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);

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

            let clear_values = if self.hdr_system.is_some() {
                vec![
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0], // 0: Color - Black
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0], // 1: G-Buffer Normal
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0], // 2: G-Buffer Albedo (Clear to opaque black)
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 0.0], // 3: G-Buffer Motion
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0, // REVERSE-Z FIX: 0.0 is Far/Infinity
                            stencil: 0,
                        },
                    },
                ]
            } else {
                vec![
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0], // 0: Swapchain Color - Black
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0, // REVERSE-Z FIX: 0.0 is Far/Infinity
                            stencil: 0,
                        },
                    },
                ]
            };

            let (framebuffer_handle, _hdr_attachment) = {
                let framebuffer = self.framebuffers.get(image_index as usize).ok_or_else(|| {
                    log::error!(
                        "Frame {}: Framebuffer index {} out of range (max: {})",
                        self.queue.current_frame,
                        image_index,
                        self.framebuffers.len()
                    );
                    AshError::VulkanError("Framebuffer index out of range".into())
                })?;
                (framebuffer.handle(), framebuffer.attachments()[0])
            };

            // PREPARE FOR CULLING & RENDERING (Moved out of Render Pass)
            let light_ptr = self.forward_plus.as_ref().map(|fp| fp.get_lights().light_ptr(frame_index)).unwrap_or(0);
            let tile_ptr = self.forward_plus.as_ref().map(|fp| fp.get_lights().tile_ptr(frame_index)).unwrap_or(0);

            let pipeline_layout = self.pipeline_layout.as_ref().ok_or_else(|| {
                AshError::VulkanError("Pipeline layout not available".to_string())
            })?;
            let pipeline_layout_handle = pipeline_layout.handle();

            let main_pass_params = MainPassParameters {
                cmd_ctx: &cmd_ctx,
                frame_index,
                scene_pipeline,
                pipeline_layout_handle,
                batch_offsets: &batch_offsets,
                view,
                projection: jittered_projection,
                swapchain_extent,
                light_ptr,
                tile_ptr,
                scene,
            };

            // CRITICAL FIX: Execute Compute Culling BEFORE Render Pass
            self.cull_main_pass(&main_pass_params)?;

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(main_render_pass)
                .framebuffer(framebuffer_handle)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .clear_values(&clear_values);

            cmd_ctx.begin_render_pass(&render_pass_begin, vk::SubpassContents::INLINE);
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

            let dummy_render_transform = Transform::identity(); // Local dummy for render ctx

            let render_ctx = FeatureRenderContext {
                device: self.device.device.as_ref(),
                descriptor_allocator: self.descriptors.as_ref(),
                command_buffer,
                transform: &dummy_render_transform,
                frame_index,
                screen_width: swapchain_extent.width,
                screen_height: swapchain_extent.height,
            };

            self.features.render(&render_ctx);

            let _ = (|| -> Result<vk::DescriptorSet> {
                // Bind Unified Bindless descriptor set (Set 0)
                // We check descriptors existence to ensure renderer is initialized,
                // but we bind strictly from bindless_manager.
                if self.descriptors.is_some() {
                    cmd_ctx.bind_descriptor_sets(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout_handle,
                        0, // Set 0: Unified Bindless (Textures, Materials, Instances, Shadows, Storage)
                        &[self.assets.bindless_manager.descriptor_set()],
                        &[],
                    );
                }
                Ok(vk::DescriptorSet::null())
            })()?;

            // --- Phase 10: GPU Instancing & Culling Integration ---
            self.render_main_pass(&main_pass_params)?;

            cmd_ctx.end_render_pass();


            // --- VSR (Upscaling) Pass ---
            if let (Some(ref mut vsr), Some(ref mut _gbuffer)) =
                (&mut self.vsr_pass, &mut self.gbuffer)
            {
                // Read back metrics from previous frame (non-blocking)
                let _: std::result::Result<vsr_pass::VsrMetricsReadback, vsr_pass::VsrError> = vsr.readback_metrics(command_buffer, &self.alloc.vma);

                let _depth_buffer = self
                    .depth_buffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Depth buffer missing".to_string()))?;
                // Pass current color result (which is now correctly the HDR buffer if initialized)
                // and upsample to VSR history.
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

                vsr.upscale_with_sharpening(
                    command_buffer,
                    vsr_inputs,
                    &vsr_config,
                    sharpen_config.as_ref(),
                )
                .map_err(|e| AshError::VulkanError(format!("VSR upscale failed: {e}")))?;
                vsr.next_frame();
            }

            // Update post-processing descriptors after VSR completes
            // This ensures we bind the current frame's output, not the previous frame's
            if let Some(hdr) = self.hdr_system.as_ref() {
                let input_view = if let Some(vsr) = self.vsr_pass.as_ref() {
                     vsr.active_view()
                } else {
                     hdr.view()
                };
                
                let bloom_view = self._black_texture.view();
                let ssgi_view = self._black_texture.view(); // Placeholder
                let sampler = hdr.sampler();
                
                self.post_process.update_descriptor_set(
                    image_index as usize,
                    input_view,
                    bloom_view,
                    ssgi_view,
                    sampler
                );
            }

            // --- Post-Processing (Tonemapping & Resolve) ---
            // NOTE: HDR buffer is already in SHADER_READ_ONLY_OPTIMAL layout
            // via the render pass final_layout, no manual barrier needed.

            // Resolve HDR target to swapchain (always needed even if tonemapping is disabled)
            log::debug!("DEBUG: About to call render_post_processing");
            self.post_process.render(command_buffer, image_index as usize, swapchain_extent)?;
            log::debug!("DEBUG: render_post_processing completed successfully");

            cmd_ctx.end()?;

            let is_headless = self.swapchain.as_ref().map_or(false, |s| s.is_headless());

            let wait_semaphores_all = [image_available];
            let wait_stages_all = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let (wait_semaphores, wait_stages) = if is_headless {
                (&[] as &[vk::Semaphore], &[] as &[vk::PipelineStageFlags])
            } else {
                (&wait_semaphores_all as &[vk::Semaphore], &wait_stages_all as &[vk::PipelineStageFlags])
            };

            let signal_semaphores = [render_finished];
            let command_buffers_submit = [command_buffer];

            self.queue.submit(
                &command_buffers_submit,
                wait_semaphores,
                wait_stages,
                &signal_semaphores,
                in_flight_fence,
            )?;

            let resize_needed = if let Some(swapchain) = self.swapchain.as_ref() {
                self.queue.present(swapchain, image_index, &signal_semaphores)?
            } else {
                false
            };
            self.last_image_index = image_index;

            if resize_needed {
                log::warn!(
                    "Frame {}: Swapchain out of date/suboptimal, requesting resize",
                    self.queue.current_frame
                );
                self.request_swapchain_resize(swapchain_extent);
            }

            self.queue.flush_old_swapchains(&self.device);

            self.queue.advance_frame();

            Ok(())
        }
    }


    pub fn buffer_pool(&self) -> Arc<BufferPool> {
        Arc::clone(&self.buffer_pool)
    }



    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Post-Processing API
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



    /// Enables or disables tonemapping

    /// Set the post-processing configuration
    pub fn set_post_processing_config(
        &mut self,
        config: crate::renderer::systems::post_process::PostProcessConfig,
    ) {
        self.post_process.config = config;
    }

    /// Returns whether tonemapping is enabled
    pub fn tonemapping_enabled(&self) -> bool {
        self.post_process.config.tonemapping_enabled
    }

    /// Sets the tonemapping exposure value
    pub fn set_tonemapping_exposure(&mut self, exposure: f32) {
        self.post_process.config.exposure = exposure.max(0.0);
    }

    /// Returns the tonemapping exposure value
    pub fn tonemapping_exposure(&self) -> f32 {
        self.post_process.config.exposure
    }

    /// Sets the tonemapping gamma value
    pub fn set_tonemapping_gamma(&mut self, gamma: f32) {
        self.post_process.config.gamma = gamma.max(0.1);
    }

    /// Returns the tonemapping gamma value
    pub fn tonemapping_gamma(&self) -> f32 {
        self.post_process.config.gamma
    }

    /// Enables or disables bloom
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.post_process.config.bloom_enabled = enabled;
    }

    /// Returns whether bloom is enabled
    pub fn bloom_enabled(&self) -> bool {
        self.post_process.config.bloom_enabled
    }

    /// Sets the bloom intensity
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        self.post_process.config.bloom_intensity = intensity.clamp(0.0, 2.0);
    }

    /// Returns the bloom intensity
    pub fn bloom_intensity(&self) -> f32 {
        self.post_process.config.bloom_intensity
    }

    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Forward+ Lighting API
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Update point lights for Forward+ rendering
    ///
    /// Call this each frame to update light positions and properties.
    pub fn update_point_lights(&mut self, lights: &[crate::renderer::features::PointLight]) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(lights, &[], &[]);
        }
    }

    /// Update directional lights for Forward+ rendering
    pub fn update_directional_lights(
        &mut self,
        lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(&[], lights, &[]);
        }
    }

    /// Update spot lights for Forward+ rendering
    pub fn update_spot_lights(&mut self, lights: &[crate::renderer::features::SpotLight]) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(&[], &[], lights);
        }
    }

    /// Update all lights (point and directional) for Forward+ rendering
    pub fn update_lights(
        &mut self,
        point_lights: &[crate::renderer::features::PointLight],
        directional_lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(point_lights, directional_lights, &[]);
        }
    }

    /// Returns whether Forward+ lighting is enabled
    pub fn forward_plus_enabled(&self) -> bool {
        self.forward_plus
            .as_ref()
            .map(|fp| fp.is_enabled())
            .unwrap_or(false)
    }

    /// Returns the number of active lights
    pub fn forward_plus_light_count(&self) -> usize {
        self.forward_plus
            .as_ref()
            .map(|fp| fp.light_count())
            .unwrap_or(0)
    }

    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // GPU-Driven Occlusion Culling API
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Enables GPU-driven occlusion culling using Hi-Z pyramid.
    ///
    /// This initializes the Hi-Z pass and indirect draw pass for GPU-based
    /// visibility culling. Objects are tested against a hierarchical depth
    /// buffer before rendering, reducing draw calls significantly.
    ///
    /// # Safety
    /// Should be called after the renderer is fully initialized.
    pub fn enable_occlusion_culling(&mut self) -> Result<()> {
        if self.hiz_pass.is_some() {
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
            hiz.init(&self.alloc.vma, &self.device, extent.width, extent.height)?;
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
                &self.alloc.vma,
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

        self.hiz_pass = Some(hiz);
        self.indirect_draw_pass = Some(indirect);

        log::info!("Occlusion culling enabled (Hi-Z + Indirect Draw)");
        Ok(())
    }

    /// Returns whether GPU-driven occlusion culling is enabled
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.hiz_pass.is_some() && self.indirect_draw_pass.is_some()
    }

    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Temporal Super-Resolution API
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

        self.post_process.config.tonemapping_enabled = true;
        
        // CRITICAL: Recreate main pipeline and framebuffers to use the NEW HDR render pass format
        crate::renderer::swapchain_manager::recreate_swapchain_resources(self, scene)?;
        
        log::info!("Post-processing pipeline enabled (HDR + Tonemapping)");
        Ok(())
    }



    /// Returns post-processing settings as a tuple (exposure, gamma, bloom_intensity)
    pub fn post_processing_settings(&self) -> (f32, f32, f32) {
        (
            self.post_process.config.exposure,
            self.post_process.config.gamma,
            self.post_process.config.bloom_intensity,
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
        if let Some(hiz) = &self.hiz_pass {
            log::info!("{}", hiz.quality_report());
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

            let _ = self.device.device.device_wait_idle();

            // Phase 7: Cleanup Global Cluster Buffer (BDA)
            // CRITICAL: This BDA buffer must be destroyed explicitly while the device is still valid
            // and BEFORE the allocator is dropped, as it depends on both.
            if let Some(mut cluster_buffer) = self.global_cluster_buffer.take() {
                cluster_buffer.destroy();
            }

            // CRITICAL FIX: Explicitly drop post-processing resources before general resource cleanup.
            // This prevents access violations during shutdown if the window/surface is destroyed.
            // ORDER MATTERS: Pipeline depends on RenderPass (in FullscreenPass), so destroy Pipeline FIRST.

            swapchain_manager::cleanup_render_pass(self);  // Drains hdr_render_pass
            swapchain_manager::cleanup_pipeline(self);

            // Cleanup VSM (Explicit)
            if let Some(mut shadow_system) = self.shadow_system.take() {
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

            // Cleanup Forward+ integration (Phase 5)
            if let Some(mut fp) = self.forward_plus.take() {
                fp.destroy(&self.alloc, &self.device.device);
            }

            // Cleanup UE5 feature modules (Phase 5)
            if let Some(mut hiz) = self.hiz_pass.take() {
                hiz.destroy(&self.alloc.vma);
            }
            if let Some(mut indirect) = self.indirect_draw_pass.take() {
                indirect.destroy(&self.alloc.vma);
            }
            if let Some(mut vsr) = self.vsr_pass.take() {
                vsr.destroy(&self.alloc.vma);
            }

            // Cleanup async readback manager


            for ub in &mut self.uniform_buffers {
                let _ = ub.cleanup();
            }
            self.uniform_buffers.clear();

            if let Some(mut buffer) = self.material_storage_buffer.take() {
                let _ = buffer.cleanup();
            }

            self.draw_items.clear();
            
            // Phase 19 Transient Arena Cleanup
            self.alloc.destroy_buffer(self.transform_arena, &mut self.transform_arena_alloc);


            self.depth_buffer = None;
            self.pipeline = None;
            self.render_pass = None;
            self.swapchain = None;


            log::info!("Ash Renderer shut down successfully");
        }
    }
}

/// Helper to register a single texture with the bindless manager.
fn register_single_texture(
    bindless_manager: &mut vulkan::BindlessManager,
    registry: &mut HashMap<u32, Arc<Texture>>,
    texture_name: &str,
    texture: Option<Arc<Texture>>,
) -> Result<Option<u32>> {
    match texture {
        Some(tex) => match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
            Ok(idx) => {
                log::debug!("Registered {texture_name} texture at bindless index {idx}");
                registry.insert(idx, tex);
                Ok(Some(idx))
            }
            Err(e) => {
                log::error!("Failed to register {texture_name} texture: {e}");
                Err(AshError::TextureBindingFailed(format!(
                    "{texture_name}: {e}"
                )))
            }
        },
        None => {
            log::debug!("{texture_name} texture not provided");
            Ok(None)
        }
    }
}

/// DRY helper to register all standard PBR textures for a mesh.
fn register_mesh_textures(
    mesh: &mut Mesh,
    bindless_manager: &mut vulkan::BindlessManager,
    registry: &mut HashMap<u32, Arc<Texture>>,
) -> Result<()> {
    mesh.texture_index =
        register_single_texture(bindless_manager, registry, "base_color", mesh.texture.clone())?;
    mesh.normal_texture_index =
        register_single_texture(bindless_manager, registry, "normal", mesh.normal_texture.clone())?;
    mesh.metallic_roughness_texture_index = register_single_texture(
        bindless_manager,
        registry,
        "metallic_roughness",
        mesh.metallic_roughness_texture.clone(),
    )?;
    mesh.occlusion_texture_index = register_single_texture(
        bindless_manager,
        registry,
        "occlusion",
        mesh.occlusion_texture.clone(),
    )?;
    mesh.emissive_texture_index = register_single_texture(
        bindless_manager,
        registry,
        "emissive",
        mesh.emissive_texture.clone(),
    )?;
    Ok(())
}
