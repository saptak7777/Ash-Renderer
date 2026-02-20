use crate::renderer::{
    assets::AssetManager,
    context::Context,
    initialization,
    instancing::{BatchKey, InstancingManager},
    resource_registry::ResourceId,
    vram_budget, DepthBuffer as DepthBufferType, RendererConfig, Texture as TextureType,
    TextureInitContext,
};
use crate::{vulkan, AshError};
use ash::vk;
use glam::Mat4;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// --- Submodules ---
pub mod bindless_validator;
pub mod buffer;
pub mod cluster_buffer;
pub mod depth_buffer;
pub mod descriptor;
pub mod gbuffer;
pub mod global_geometry_buffer;
pub mod ibl;
pub mod image;
pub mod material;
pub mod mesh;
pub mod motion;
pub mod optimized_buffer_pool;
pub mod pipeline;
pub mod render_targets;
pub mod safe_resource;
pub mod texture;
pub mod texture_compressor;

pub mod thread_safe_pool;
pub mod transform;
pub mod uniform;

#[cfg(feature = "gltf_loading")]
pub mod gltf_loader;

// --- Re-exports ---
pub use bindless_validator::BindlessValidator;
pub use buffer::BufferHandle;
pub use cluster_buffer::GlobalClusterBuffer;
pub use depth_buffer::DepthBuffer;
pub use descriptor::DescriptorSetHandle;
pub use gbuffer::GBuffer;
pub use global_geometry_buffer::DualHeapGeometryBuffer;
pub use ibl::{IblAssetHeader, IblUploadParams};
pub use image::{ImageCreateInfo, ImageHandle};
pub use material::{Material, MaterialHandle, MaterialManager};
pub use mesh::{MaterialDescriptor, Mesh, MeshDescriptor, Vertex};
pub use motion::ObjectMotionData;
pub use optimized_buffer_pool::{BufferAllocation, BufferPool, BufferPoolConfig, BufferPoolStats};
pub use pipeline::PipelineHandle;
pub use render_targets::HdrSystem;
pub use safe_resource::SafeResource;
pub use texture::{Texture, TextureData, TextureDesc};
pub use texture_compressor::{CompressionFormat, TextureCompressor};
pub use thread_safe_pool::{PoolStats, PooledResource, ThreadSafeResourcePool};
pub use transform::{Camera, TemporalCamera, Transform, TransformHandle, TransformSystem, MVP};
pub use uniform::{InstanceBuffer, MvpMatrices, UniformBuffer};

extern crate image as image_crate;

pub struct IblData {
    pub irradiance_map: Texture,
    pub prefilter_map: Texture,
    pub brdf_lut: Texture,
}

/// Resources manages the lifecycle of heavy GPU buffers and asset managers.
pub struct Resources {
    // Assets & Bindless Management
    pub assets: AssetManager,
    pub buffer_pool: Arc<BufferPool>,

    // Core Geometry Buffers
    pub geometry_buffer: Arc<DualHeapGeometryBuffer>,
    pub global_cluster_buffer: Option<Arc<GlobalClusterBuffer>>,

    // Shader Resources
    pub uniform_buffers: Vec<Arc<RwLock<UniformBuffer>>>,
    pub material_storage_buffer: Option<
        Arc<
            RwLock<
                crate::renderer::resources::uniform::StorageBuffer<
                    crate::renderer::resources::uniform::MaterialUniform,
                >,
            >,
        >,
    >,
    pub descriptors: Option<vulkan::DescriptorAllocator>,

    // Instance System
    pub(crate) _instancing_manager: InstancingManager,
    pub instance_buffers: Vec<InstanceBuffer>,
    pub instance_buffer_addresses: Vec<u64>,
    pub batch_offsets: HashMap<BatchKey, u32>,
    pub material_heap_address: u64,
    pub swapchain_extent: vk::Extent2D,
    pub current_view_proj: Mat4,

    #[cfg(debug_assertions)]
    pub(crate) instancing_guard_active: bool,

    // Render Targets & Debug
    pub readback_buffer: Option<BufferHandle>,
    pub gbuffer: Option<GBuffer>,
    pub depth_buffer: Option<DepthBufferType>,
    pub depth_buffer_id: Option<ResourceId>,

    // Default Textures
    pub default_texture: TextureType,
    pub black_texture: TextureType,
    pub white_texture: TextureType,
    pub default_skybox: TextureType,
    pub default_cube_black: TextureType,
    pub dummy_black_cube: TextureType,
    pub dummy_black_2d: TextureType,

    // IBL Data
    pub ibl_data: Option<IblData>,

    // --- Internal Resources ---
    pub(crate) pipelines: Option<crate::renderer::init_types::PipelineData>,
    pub(crate) passes: Option<crate::renderer::init_types::RenderingPasses>,
    pub(crate) lighting: Option<crate::renderer::init_types::LightingSystem>,
    pub(crate) post: Option<crate::renderer::init_types::PostProcessingSystem>,
    pub total_instance_count: u32,
    pub indirect_draw_enabled: bool,
}

pub struct ResourcesInitInfo<'a, S: vulkan::SurfaceProvider> {
    pub context: &'a Context,
    pub width: u32,
    pub height: u32,
    pub config: &'a RendererConfig,
    pub surface_provider: &'a S,
    pub pipeline_cache: &'a crate::renderer::pipeline_cache::PipelineCache,
    pub swapchain: &'a vulkan::SwapchainWrapper,
    pub cmds: &'a vulkan::CommandBufferManager,
}

impl Resources {
    /// Initializes all rendering resources.
    pub fn new<S: vulkan::SurfaceProvider>(info: ResourcesInitInfo<'_, S>) -> crate::Result<Self> {
        let ResourcesInitInfo {
            context,
            width,
            height,
            config,
            surface_provider: _surface_provider,
            pipeline_cache,
            swapchain,
            cmds,
        } = info;
        // SAFETY: Initialization involves raw Vulkan pointer manipulation and resource creation
        // that must follow strict device/allocator lifetimes.
        unsafe {
            let dev_mem_props = context.device.memory_properties;
            let vram_budget = vram_budget::VramBudget::new(&dev_mem_props);
            let mut pipeline_cfg = config.pipeline.clone();

            if !context.device.sample_rate_shading_supported
                && pipeline_cfg.sample_shading.enabled()
            {
                log::warn!("Sample rate shading requested but not supported by hardware. Falling back to disabled.");
                pipeline_cfg.sample_shading =
                    crate::renderer::types::SampleShadingQuality::Disabled;
            }

            let mut depth_buffer = crate::renderer::resources::depth_buffer::DepthBuffer::new(
                Arc::clone(&context.device.device),
                Arc::clone(&context.alloc),
                swapchain.extent.width,
                swapchain.extent.height,
            )?;
            let depth_buffer_id = depth_buffer.register_with_registry(&context.resources)?;

            let aspect = width as f32 / height as f32;

            // --- Core & Pass Initialization ---
            let mut core = initialization::init_core_infrastructure(
                &context.device,
                &context.alloc,
                &context.instance,
                &context.resources,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len(),
                aspect,
            )?;

            // Create VSM compute layout manually since Manager is not yet initialized
            let vsm_compute_layout =
                initialization::create_vsm_compute_layout(&context.device.device)?;

            let set_layouts = [
                core.bindless_manager.descriptor_set_layout(),
                vsm_compute_layout,
            ];
            let color_formats = vec![swapchain.format];

            let pipelines = initialization::init_pipelines(initialization::PipelineInitInfo {
                device: &context.device,
                resources: &context.resources,
                extent: swapchain.extent,
                color_formats: &color_formats,
                set_layouts: &set_layouts,
                pipeline_cfg: &pipeline_cfg,
                depth_format: depth_buffer.format(),
                pipeline_cache: pipeline_cache.handle(),
            })?;

            let mut passes =
                initialization::init_rendering_passes(initialization::RenderingPassesConfig {
                    device: &context.device,
                    alloc: &context.alloc,
                    resources: &context.resources,
                    bindless_manager: &mut core.bindless_manager,
                    renderer_resources: &core.renderer_resources,
                    swapchain_format: swapchain.format,
                    swapchain_extent: swapchain.extent,
                    depth_format: depth_buffer.format(),
                    depth_view: depth_buffer.view(),
                    pipeline_cache: pipeline_cache.handle(),
                    multisample_config: pipeline_cfg.multisample_config(),
                    set_layouts: &set_layouts,
                    model_renderer: &core.model_renderer,
                    upload_command_pool: cmds.upload_command_pool_handle(),
                })?;

            let lighting = initialization::init_lighting_system(
                &context.device,
                &context.alloc,
                &mut core.bindless_manager,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len() as u32,
                swapchain.extent,
            )?;

            let post = initialization::init_post_processing(
                &context.device.device,
                swapchain.image_views.len(),
                swapchain.extent,
                swapchain.format,
            )?;

            // --- Metadata & Buffers ---
            let material_heap_address = core
                .renderer_resources
                .material_storage_buffer
                .device_address();
            let mut instance_buffer_addresses =
                Vec::with_capacity(core.renderer_resources.instance_buffers.len());
            for buffer in &core.renderer_resources.instance_buffers {
                instance_buffer_addresses.push(buffer.device_address());
            }

            let readback_buffer = if context.device.headless {
                let buffer_size = (swapchain.extent.width * swapchain.extent.height * 4) as u64;
                Some(BufferHandle::new_with_flags(
                    Arc::clone(&context.alloc),
                    buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::Auto,
                    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    Some("Headless Readback Buffer".to_string()),
                )?)
            } else {
                None
            };

            let mut resources = Self {
                assets: {
                    let mut assets = AssetManager::new(core.bindless_manager, vram_budget);
                    assets.texture_compression = config.texture_compression;
                    assets
                },
                buffer_pool: core.buffer_pool,
                geometry_buffer: core.geometry_buffer,
                global_cluster_buffer: Some(Arc::clone(&lighting.global_cluster_buffer)),
                uniform_buffers: core
                    .renderer_resources
                    .uniform_buffers
                    .into_iter()
                    .map(|ub| Arc::new(RwLock::new(ub)))
                    .collect(),
                material_storage_buffer: Some(Arc::new(RwLock::new(
                    core.renderer_resources.material_storage_buffer,
                ))),
                descriptors: Some(core.descriptor_allocator),
                _instancing_manager: InstancingManager::new(),
                instance_buffers: core.renderer_resources.instance_buffers,
                instance_buffer_addresses,
                batch_offsets: HashMap::new(),
                material_heap_address,
                swapchain_extent: swapchain.extent,
                current_view_proj: Mat4::IDENTITY,
                readback_buffer,
                gbuffer: passes.gbuffer.take(),
                depth_buffer: Some(depth_buffer),
                depth_buffer_id: Some(depth_buffer_id),
                default_texture: core.renderer_resources.default_texture,
                black_texture: core.renderer_resources.black_texture,
                white_texture: core.renderer_resources.white_texture,
                default_skybox: core.renderer_resources.default_skybox,
                default_cube_black: core.renderer_resources.default_cube_black,
                dummy_black_cube: core.renderer_resources.dummy_black_cube,
                dummy_black_2d: core.renderer_resources.dummy_black_2d,
                ibl_data: None, // Will be filled below
                pipelines: Some(pipelines),
                passes: Some(passes),
                lighting: Some(lighting),
                post: Some(post),
                total_instance_count: 0,
                indirect_draw_enabled: true,

                #[cfg(debug_assertions)]
                instancing_guard_active: false,
            };

            // --- IBL Generation ---
            if let Some(env_path) = &config.environment_map {
                log::info!("IBL: Generating data from environment map: {env_path:?}");

                match (|| -> crate::Result<IblData> {
                    let img = image_crate::open(env_path)
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to open/decode environment map: {e}"
                            ))
                        })?
                        .to_rgba32f();

                    let (width, height) = img.dimensions();
                    let pixels = img.into_raw(); // Vec<f32>

                    let hdr_tex = texture::Texture::from_raw_data(
                        &TextureInitContext {
                            allocator: Arc::clone(&context.alloc),
                            device: Arc::clone(&context.device.device),
                            command_pool: cmds.upload_command_pool_handle(),
                            queue: context.device.graphics_queue,
                        },
                        bytemuck::cast_slice(&pixels),
                        &crate::renderer::types::TextureCreateInfo {
                            width,
                            height,
                            format: vk::Format::R32G32B32_SFLOAT,
                            mip_levels: 1,
                            name: Some("IBL_Source_HDR"),
                        },
                    )?;

                    use crate::renderer::lighting::IblProcessor;
                    let processor = IblProcessor::new(Arc::clone(&context.device.device))?;

                    let bundle = processor.generate(
                        &context.device,
                        Arc::clone(&context.alloc),
                        cmds.upload_command_pool_handle(),
                        context.device.graphics_queue,
                        &hdr_tex,
                    )?;

                    log::info!("IBL: Successfully generated Irradiance and Prefilter maps.");

                    Ok(IblData {
                        irradiance_map: bundle.irradiance_map,
                        prefilter_map: bundle.prefilter_map,
                        brdf_lut: bundle.brdf_lut,
                    })
                })() {
                    Ok(data) => resources.ibl_data = Some(data),
                    Err(e) => {
                        log::warn!("IBL: Generation failed: {e}. Falling back to default.");
                        resources.ibl_data = None;
                    }
                }
            }

            // --- Descriptor Plumbing ---
            if let Some(data) = &resources.ibl_data {
                resources.assets.bindless_manager.update_ibl_descriptors(
                    data.irradiance_map.view(),
                    data.irradiance_map.sampler(),
                    data.prefilter_map.view(),
                    data.prefilter_map.sampler(),
                    data.brdf_lut.view(),
                    data.brdf_lut.sampler(),
                )?;
            } else {
                // Fallback to dummy black textures to prevent shader crashes and washed-out shadows
                resources.assets.bindless_manager.update_ibl_descriptors(
                    resources.dummy_black_cube.view(),
                    resources.dummy_black_cube.sampler(),
                    resources.dummy_black_cube.view(),
                    resources.dummy_black_cube.sampler(),
                    resources.dummy_black_2d.view(),
                    resources.dummy_black_2d.sampler(),
                )?;
            }

            Ok(resources)
        }
    }

    /// The "pro" way to handle bindless storage buffers, providing type safety
    /// and automatic memory management.
    pub fn register_bindless_storage_buffer<T: Copy>(
        &mut self,
        context: &Context,
        data: &[T],
        name: &str,
    ) -> crate::Result<(
        Arc<parking_lot::Mutex<crate::renderer::resources::uniform::StorageBuffer<T>>>,
        u32,
    )> {
        let bindless_manager = &mut self.assets.bindless_manager;

        // SAFETY: Buffer creation and bindless registration involve raw Vulkan handles.
        // T is Copy, and the buffer is sized correctly for the provided data.
        unsafe {
            let mut buffer = crate::renderer::resources::uniform::StorageBuffer::new(
                Arc::clone(&context.alloc),
                Arc::clone(&context.device.device),
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

    pub fn read_headless_image(
        &mut self,
        context: &Context,
        frame: &crate::renderer::frame::Frame,
    ) -> crate::Result<Vec<u8>> {
        if !context.device.headless {
            return Err(AshError::VulkanError("Not in headless mode".to_string()));
        }

        let swapchain = frame.swapchain.as_ref().ok_or(AshError::VulkanError(
            "Swapchain not initialized".to_string(),
        ))?;

        let readback_buffer = self.readback_buffer.as_mut().ok_or(AshError::VulkanError(
            "Readback buffer not initialized".to_string(),
        ))?;

        // Wait for the device to be idle to ensure rendering is complete
        // SAFETY: Image-to-buffer copies and memory mapping require correct synchronization
        // and valid resource handles. headless mode ensures the readback buffer exists.
        unsafe {
            context.device.device.device_wait_idle().map_err(|e| {
                AshError::VulkanError(format!("Failed to wait for device idle: {e}"))
            })?;
        }

        let image_index = frame.last_image_index;
        if image_index >= swapchain.images.len() as u32 {
            return Err(AshError::VulkanError("Invalid image index".to_string()));
        }
        let src_image = swapchain.images[image_index as usize];

        // Create a command buffer for the copy
        let cmd = frame.cmds.get_transfer_command_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: Begin command buffer is safe as cmd is a fresh one-time-submit buffer.
        unsafe {
            context
                .device
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

            context.device.device.cmd_copy_image_to_buffer(
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

            context.device.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            context
                .device
                .device
                .end_command_buffer(cmd)
                .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

            let command_buffers = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            context
                .device
                .device
                .queue_submit(
                    context.device.graphics_queue,
                    &[submit_info],
                    vk::Fence::null(),
                )
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to submit copy command: {e}"))
                })?;

            context
                .device
                .device
                .queue_wait_idle(context.device.graphics_queue)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to wait for queue idle: {e}"))
                })?;

            context
                .device
                .device
                .free_command_buffers(frame.cmds.upload_command_pool_handle(), &[cmd]);
        }

        // Map memory and read
        let size = (swapchain.extent.width * swapchain.extent.height * 4) as usize;
        let mut data = vec![0u8; size];

        // SAFETY: Memory mapping and raw pointer copies for headless readback.
        // Size is validated against swapchain extent.
        unsafe {
            let ptr = context
                .alloc
                .vma
                .map_memory(readback_buffer.allocation_mut())
                .map_err(|e| AshError::VulkanError(format!("Failed to map memory: {e}")))?;

            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size);

            context
                .alloc
                .vma
                .unmap_memory(readback_buffer.allocation_mut());
        }

        Ok(data)
    }

    /// Extracted resize logic for DepthBuffer, GBuffer, and Uniform Buffers.
    pub fn resize(
        &mut self,
        context: &Context,
        gbuffer_indices: &mut crate::renderer::GBufferIndices,
        extent: vk::Extent2D,
        image_count: usize,
    ) -> crate::Result<()> {
        // --- 1. Recreate Depth Buffer ---
        if let Some(id) = self.depth_buffer_id.take() {
            if let Err(e) = context.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old depth buffer: {e}");
            }
        }

        let mut depth_buffer = // SAFETY: Fresh resource creation with valid device/allocator.
            unsafe {
            DepthBuffer::new(
                Arc::clone(&context.device.device),
                Arc::clone(&context.alloc),
                extent.width,
                extent.height,
            )?
        };

        let depth_buffer_id = depth_buffer
            .register_with_registry(&context.resources)
            .map_err(|e| AshError::VulkanError(format!("Failed to register depth buffer: {e}")))?;

        self.depth_buffer = Some(depth_buffer);
        self.depth_buffer_id = Some(depth_buffer_id);
        self.swapchain_extent = extent;

        if gbuffer_indices.depth_index == u32::MAX {
            gbuffer_indices.depth_index = self.assets.bindless_manager.add_sampled_image(
                self.depth_buffer.as_ref().unwrap().view(),
                self.default_texture.sampler(),
            )?;
        } else {
            self.assets.bindless_manager.update_sampled_image(
                gbuffer_indices.depth_index,
                self.depth_buffer.as_ref().unwrap().view(),
                self.default_texture.sampler(),
            )?;
        }

        // --- 2. Recreate GBuffer ---
        let gbuffer = // SAFETY: GBuffer involves raw resource creation; extent is validated.
            unsafe {
            GBuffer::new(
                Arc::clone(&context.device.device),
                Arc::clone(&context.alloc),
                extent.width,
                extent.height,
            )?
        };

        if gbuffer_indices.motion_index == u32::MAX {
            gbuffer_indices.motion_index = self
                .assets
                .bindless_manager
                .add_sampled_image(gbuffer.motion_view(), self.default_texture.sampler())?;
        } else {
            self.assets.bindless_manager.update_sampled_image(
                gbuffer_indices.motion_index,
                gbuffer.motion_view(),
                self.default_texture.sampler(),
            )?;
        }

        self.gbuffer = Some(gbuffer);

        // --- 3. Recreate Uniform Buffers ---
        for ub in &self.uniform_buffers {
            let _ = ub.write().unwrap().cleanup();
        }
        self.uniform_buffers.clear();

        // SAFETY: Uniform buffer recreation during resize. All previous resources are cleaned up.
        unsafe {
            for _ in 0..image_count {
                let mut buffer = UniformBuffer::new(
                    Arc::clone(&context.alloc),
                    Arc::clone(&context.device.device),
                )?;

                {
                    let matrices = buffer.matrices_mut();
                    matrices.model = Mat4::IDENTITY;
                    matrices.view = Mat4::IDENTITY;
                    matrices.projection = Mat4::IDENTITY;
                    matrices.view_proj = Mat4::IDENTITY;
                    matrices.camera_pos = glam::Vec4::W;
                }
                buffer.update()?;
                self.uniform_buffers.push(Arc::new(RwLock::new(buffer)));
            }
        }

        Ok(())
    }

    /// Update instance buffers and compute batch offsets
    pub fn update_instance_buffers(
        &mut self,
        instancing_manager: &crate::renderer::instancing::InstancingManager,
        frame_index: usize,
    ) -> crate::Result<()> {
        let mut all_instances = Vec::new();
        let mut batch_offsets = HashMap::new();

        for batch in instancing_manager.batches() {
            batch_offsets.insert(batch.key.clone(), all_instances.len() as u32);
            all_instances.extend_from_slice(&batch.instances);
        }

        if !all_instances.is_empty() {
            // SAFETY: Instance buffer updates with validated instance data.
            unsafe {
                self.instance_buffers[frame_index].update(&all_instances)?;
            }
        }

        self.total_instance_count = all_instances.len() as u32;
        self.batch_offsets = batch_offsets;
        Ok(())
    }

    /// Get a reference to the instancing manager
    pub fn instancing_manager(&self) -> &InstancingManager {
        #[cfg(debug_assertions)]
        debug_assert!(
            !self.instancing_guard_active,
            "InstancingManager accessed while checked out by guard!"
        );
        &self._instancing_manager
    }

    /// Get a mutable reference to the instancing manager
    pub fn instancing_manager_mut(&mut self) -> &mut InstancingManager {
        #[cfg(debug_assertions)]
        debug_assert!(
            !self.instancing_guard_active,
            "InstancingManager accessed while checked out by guard!"
        );
        &mut self._instancing_manager
    }
}
