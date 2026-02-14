use crate::error::Result;
use crate::renderer::{
    assets::AssetManager, context::Context, initialization, instancing::InstancingManager,
    resource_registry::ResourceId, vram_budget, DepthBuffer as DepthBufferType, RendererConfig,
    Texture as TextureType,
};
use crate::{vulkan, AshError};
use ash::vk;
use glam::Mat4;
use std::sync::{Arc, RwLock};

// --- Submodules (New Style: Entry point in resources.rs) ---
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
pub use image::ImageHandle;
pub use material::{Material, MaterialHandle, MaterialManager};
pub use mesh::{MaterialDescriptor, Mesh, MeshDescriptor, Vertex};
pub use motion::ObjectMotionData;
pub use optimized_buffer_pool::{BufferAllocation, BufferPool, BufferPoolConfig, BufferPoolStats};
pub use pipeline::PipelineHandle;
pub use render_targets::HdrSystem;
pub use safe_resource::SafeResource;
pub use texture::{Texture, TextureData};
pub use texture_compressor::{CompressionFormat, TextureCompressor};
pub use thread_safe_pool::{PoolStats, PooledResource, ThreadSafeResourcePool};
pub use transform::{Camera, TemporalCamera, Transform, TransformHandle, TransformSystem, MVP};
pub use uniform::{InstanceBuffer, MvpMatrices, UniformBuffer};

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
    pub instancing_manager: InstancingManager,
    pub instance_buffers: Vec<InstanceBuffer>,
    pub instance_buffer_addresses: Vec<u64>,
    pub material_heap_address: u64,

    // Render Targets & Debug
    pub readback_buffer: Option<BufferHandle>,
    pub gbuffer: Option<GBuffer>,
    pub depth_buffer: Option<DepthBufferType>,
    pub depth_buffer_id: Option<ResourceId>,

    // Default Textures (Moved from RenderSystems)
    pub default_texture: TextureType,
    pub black_texture: TextureType,
    pub white_texture: TextureType,
    pub default_skybox: TextureType,
    pub default_cube_black: TextureType,

    // --- Temporary Internal Items (Moved out in Phase 5.4) ---
    pub(crate) _model_renderer: Option<crate::renderer::model_renderer::ModelRenderer>,
    pub(crate) pipelines: Option<crate::renderer::init_types::PipelineData>,
    pub(crate) passes: Option<crate::renderer::init_types::RenderingPasses>,
    pub(crate) lighting: Option<crate::renderer::init_types::LightingSystem>,
    pub(crate) post: Option<crate::renderer::init_types::PostProcessingSystem>,
}

impl Resources {
    /// Initializes all rendering resources.
    pub fn new(
        context: &Context,
        width: u32,
        height: u32,
        config: &RendererConfig,
        _surface_provider: &impl vulkan::SurfaceProvider,
        pipeline_cache: &crate::renderer::pipeline_cache::PipelineCache,
        swapchain: &vulkan::SwapchainWrapper,
        cmds: &vulkan::CommandBufferManager,
    ) -> Result<Self> {
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

            // --- Phase 2: Core & Pass Initialization ---
            let mut core = initialization::init_core_infrastructure(
                &context.device,
                &context.alloc,
                &context.instance,
                &context.resources,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len(),
                aspect,
            )?;

            let set_layouts = [core.bindless_manager.descriptor_set_layout()];
            let color_formats = vec![swapchain.format];

            let pipelines = initialization::init_pipelines(
                &context.device,
                &context.resources,
                swapchain.extent,
                &color_formats,
                &set_layouts,
                &pipeline_cfg,
                depth_buffer.format(),
                pipeline_cache.handle(),
            )?;

            let mut passes = initialization::init_rendering_passes(
                &context.device,
                &context.alloc,
                &context.resources,
                &mut core.bindless_manager,
                &core.renderer_resources,
                swapchain.format,
                swapchain.extent,
                depth_buffer.format(),
                depth_buffer.view(),
                pipeline_cache.handle(),
                pipeline_cfg.multisample_config(),
                &set_layouts,
                &core.model_renderer,
                cmds.upload_command_pool_handle(),
            )?;

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

            // --- Phase 3: Metadata & Buffers ---
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

            Ok(Self {
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
                instancing_manager: InstancingManager::new(),
                instance_buffers: core.renderer_resources.instance_buffers,
                instance_buffer_addresses,
                material_heap_address,
                readback_buffer,
                gbuffer: passes.gbuffer.take(),
                depth_buffer: Some(depth_buffer),
                depth_buffer_id: Some(depth_buffer_id),
                default_texture: core.renderer_resources.default_texture,
                black_texture: core.renderer_resources.black_texture,
                white_texture: core.renderer_resources.white_texture,
                default_skybox: core.renderer_resources.default_skybox,
                default_cube_black: core.renderer_resources.default_cube_black,
                _model_renderer: Some(core.model_renderer),
                pipelines: Some(pipelines),
                passes: Some(passes),
                lighting: Some(lighting),
                post: Some(post),
            })
        }
    }

    /// The "pro" way to handle bindless storage buffers, providing type safety
    /// and automatic memory management.
    pub fn register_bindless_storage_buffer<T: Copy>(
        &mut self,
        context: &Context,
        data: &[T],
        name: &str,
    ) -> Result<(
        Arc<parking_lot::Mutex<crate::renderer::resources::uniform::StorageBuffer<T>>>,
        u32,
    )> {
        let bindless_manager = &mut self.assets.bindless_manager;

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
    ) -> Result<Vec<u8>> {
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
    ) -> Result<()> {
        // --- 1. Recreate Depth Buffer ---
        if let Some(id) = self.depth_buffer_id.take() {
            if let Err(e) = context.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old depth buffer: {e}");
            }
        }

        let mut depth_buffer = unsafe {
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
        let gbuffer = unsafe {
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

    /// Extracted frame preparation logic.
    pub fn update_global_data(
        &mut self,
        context: &Context,
        frame: &mut crate::renderer::frame::Frame,
        scene: &mut crate::renderer::Scene,
        systems: &mut crate::renderer::systems::Systems,
        view: Mat4,
        projection: Mat4,
        camera_pos: glam::Vec3,
        model_matrix: Option<Mat4>,
    ) -> Result<(usize, u32, vk::Extent2D, Mat4, [f32; 2])> {
        unsafe {
            let swapchain_extent = frame
                .swapchain
                .as_ref()
                .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
                .extent;

            // Phase 1: Use FrameManager to acquire next frame and synchronization objects
            let (image_index, _suboptimal) = frame.begin_frame(context)?;

            let frame_index = frame.frame_manager.get_current_frame_index();

            // Prepare culling data for this frame
            scene.occlusion_culling.begin_frame();
            for (i, item) in frame.draw_items.iter().enumerate() {
                if let Some(uploaded) = scene.model_renderer.get(&item.key) {
                    let bounds = scene
                        .mesh_data
                        .get(item.mesh_id as usize)
                        .map(|m| m.bounds)
                        .unwrap_or_else(|| {
                            crate::renderer::CullBoundingBox::new(
                                glam::Vec3::ZERO,
                                glam::Vec3::ONE * 100.0,
                            )
                        });
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
            if let Some(ref mut vsr) = systems.vsr_pass {
                let (jx, jy) = vsr.next_jitter();
                jitter_uv = [jx, jy];
                jittered_projection.col_mut(2).x += jx / swapchain_extent.width as f32;
                jittered_projection.col_mut(2).y += jy / swapchain_extent.height as f32;
            }

            // GPU synchronization confirmed; safe to update uniform buffer.
            {
                let uniform_buffer = self.uniform_buffers.get_unchecked_mut(frame_index);
                let mut dummy_transform =
                    crate::renderer::resources::transform::Transform::identity();
                let elapsed = frame.start_time.elapsed().as_secs_f32();
                let mut feature_ctx = crate::renderer::features::FeatureFrameContext {
                    device: context.device.device.as_ref(),
                    descriptor_allocator: self.descriptors.as_ref(),
                    transform: &mut dummy_transform,
                    auto_rotate: false,
                    elapsed_seconds: elapsed,
                };
                systems.features.before_frame(&mut feature_ctx);

                if let Some(shadow_system) = systems.pipeline.shadow_system_mut() {
                    shadow_system
                        .vsm_feature_mut()
                        .begin_frame(frame_index as u32, camera_pos);
                }

                let mut ub = uniform_buffer.write().unwrap();
                let matrices = ub.matrices_mut();
                let model = model_matrix.unwrap_or(Mat4::IDENTITY);
                let mut transform = crate::renderer::resources::transform::Transform::identity();
                transform.set_model(model);

                matrices.model = model;
                matrices.normal_matrix = Mat4::from_mat3(transform.normal_matrix());
                matrices.view = view;
                matrices.projection = jittered_projection;
                matrices.view_proj = jittered_projection * view;
                matrices.prev_view_proj = frame.prev_view_proj;
                matrices.camera_pos = camera_pos.extend(1.0);

                scene.scene_lighting.point_light_count = scene.point_lights.len() as u32;
                if let Some(fp) = &systems.pipeline.forward_plus {
                    let info = fp.read().unwrap().get_lights().get_forward_plus_info();
                    scene.scene_lighting.num_tiles_x = info.num_tiles[0];
                    scene.scene_lighting.num_tiles_y = info.num_tiles[1];
                    scene.scene_lighting.tile_size = info.tile_size;
                }
                matrices.set_lighting(&scene.scene_lighting);
                matrices.set_light_space_matrix(glam::Mat4::IDENTITY);

                // Phase 2: Host-side Forward+ updates (Lights and Camera)
                if let Some(ref fp_integration) = systems.pipeline.forward_plus {
                    let mut fp = fp_integration.write().unwrap();
                    fp.update_lights(
                        &scene.point_lights,
                        &scene.directional_lights,
                        &scene.spot_lights,
                    );
                    fp.upload_to_gpu(&context.alloc, &context.device.device, frame_index as usize)?;
                    fp.update_camera(
                        &context.alloc,
                        frame_index as usize,
                        &view.to_cols_array_2d(),
                        &projection.to_cols_array_2d(),
                        &camera_pos.extend(1.0).to_array(),
                    )?;
                }

                let view_proj = matrices.view_proj;
                ub.update()?;
                frame.prev_view_proj = view_proj;
            }

            Ok((
                frame_index,
                image_index,
                swapchain_extent,
                jittered_projection,
                jitter_uv,
            ))
        }
    }
}
