use crate::{
    renderer::{
        features::ShadowSystem,
        passes::hiz::{AdaptiveHiZManager, HiZPass},
        passes::vsr::VsrPass,
        systems::post_process::PostProcessSystem,
        vcgs::IndirectDrawPass,
        ForwardPlusIntegration, HdrSystem, Scene,
    },
    Result,
};
use ash::vk;
use glam::Vec3;
use std::sync::{Arc, RwLock};

/// RenderPipeline orchestrates the high-level rendering flow.
/// It owns the major rendering subsystems and manages their execution order.
pub struct RenderPipeline {
    shadow_system: Option<ShadowSystem>,
    post_process: PostProcessSystem,
    pub(crate) hiz_pass: Option<Arc<RwLock<HiZPass>>>, // Keep crate-public for Renderer access for now
    adaptive_hiz_manager: AdaptiveHiZManager,

    // Moved from Renderer (Chunk 3.2a)
    pub forward_plus: Option<Arc<RwLock<ForwardPlusIntegration>>>,
    pub indirect_draw_pass: Option<Arc<RwLock<IndirectDrawPass>>>,
    pub main_graphics_pipeline: Option<crate::vulkan::Pipeline>,
    pub pipeline_layout: Option<crate::vulkan::PipelineLayout>,
}

/// Context for rendering geometry, grouping multiple parameters to stabilize the API.
pub struct GeometryRenderContext<'a> {
    pub device: &'a crate::vulkan::VulkanDevice,
    pub command_buffer: &'a crate::vulkan::CommandBufferContext<'a>,
    pub scene: &'a Scene,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub swapchain_extent: vk::Extent2D,
    pub frame_ptr: u64,
    pub material_ptr: u64,
    pub light_ptr: u64,
    pub tile_ptr: u64,
    pub debug_enabled: bool,
    pub color_image: vk::Image,
    pub color_view: vk::ImageView,
    pub depth_image: vk::Image,
    pub depth_view: vk::ImageView,
    pub normal_view: Option<vk::ImageView>,
    pub albedo_view: Option<vk::ImageView>,
    pub motion_view: Option<vk::ImageView>,
    pub skybox: Option<&'a crate::renderer::passes::SkyboxPass>,
    pub features: Option<&'a crate::renderer::features::FeatureManager>,
    pub frame_index: usize,
    pub descriptor_allocator: Option<&'a crate::vulkan::DescriptorAllocator>,
    pub transform: &'a crate::renderer::Transform,
    pub is_swapchain_image: bool,
    pub depth_format: vk::Format,
}

impl RenderPipeline {
    pub fn new(
        shadow_system: Option<ShadowSystem>,
        post_process: PostProcessSystem,
        hiz_pass: Option<Arc<RwLock<HiZPass>>>,
        adaptive_hiz_manager: AdaptiveHiZManager,
        forward_plus: Option<Arc<RwLock<ForwardPlusIntegration>>>,
        indirect_draw_pass: Option<Arc<RwLock<IndirectDrawPass>>>,
        main_graphics_pipeline: Option<crate::vulkan::Pipeline>,
        pipeline_layout: Option<crate::vulkan::PipelineLayout>,
    ) -> Self {
        Self {
            shadow_system,
            post_process,
            hiz_pass,
            adaptive_hiz_manager,
            forward_plus,
            indirect_draw_pass,
            main_graphics_pipeline,
            pipeline_layout,
        }
    }

    /// Execute the Hi-Z depth pyramid construction pass.
    /// This is typically done at the start of the frame after the previous frame's depth is available.
    ///
    /// # Lock Ordering
    /// This method acquires locks in the following order:
    /// 1. `hiz_pass` (write lock)
    /// 2. `indirect_draw_pass` (write lock, if provided)
    ///
    /// Any other code that acquires both locks must follow this same ordering
    /// to avoid deadlocks.
    pub fn execute_hiz_pass(
        &mut self,
        command_buffer: vk::CommandBuffer,
        depth_image: vk::Image,
        gpu_profiler: Option<&crate::renderer::diagnostics::GpuProfiler>,
        black_texture_view: vk::ImageView,
        black_texture_sampler: vk::Sampler,
    ) -> Result<()> {
        if let Some(ref hiz_arc) = self.hiz_pass {
            let mut hiz = hiz_arc.write().map_err(|e| {
                log::error!("Hi-Z pass RwLock poisoned: {}", e);
                crate::AshError::VulkanError("Hi-Z pass RwLock poisoned".into())
            })?;

            // Update adaptive quality based on previous frame's metrics
            if let Some(profiler) = gpu_profiler {
                let timings = profiler.last_extended_timings();
                if timings.valid {
                    let hiz_time_ms = timings.hiz_generate_ms as f64;
                    if let Some(new_quality) =
                        self.adaptive_hiz_manager.update(hiz.quality(), hiz_time_ms)
                    {
                        hiz.set_quality(new_quality);
                    }
                }
            }

            // Build Hi-Z pyramid from depth buffer
            unsafe {
                hiz.build_pyramid(command_buffer, depth_image)?;
            }

            // Update descriptors for systems that depend on Hi-Z (like occlusion culling)
            let hiz_view = hiz.hiz_view().unwrap_or(black_texture_view);
            let hiz_sampler = if hiz.is_initialized() {
                hiz.hiz_sampler()
            } else {
                black_texture_sampler
            };

            if let Some(ref indirect_arc) = self.indirect_draw_pass {
                let indirect = indirect_arc.write().map_err(|e| {
                    log::error!("Indirect draw pass RwLock poisoned: {}", e);
                    crate::AshError::VulkanError("Indirect draw pass RwLock poisoned".into())
                })?;
                unsafe {
                    indirect.update_hiz_descriptor(hiz_view, hiz_sampler);
                }
            }

            if let Some(profiler) = gpu_profiler {
                unsafe {
                    profiler.write_timestamp(
                        command_buffer,
                        crate::renderer::diagnostics::TimingScope::HiZGenerateEnd,
                    );
                }
            }
        }
        Ok(())
    }

    /// Render shadow maps for the current frame.
    /// This implementation currently supports Virtual Shadow Maps (VSM).
    pub fn render_shadows(
        &mut self,
        command_buffer: vk::CommandBuffer,
        scene: &Scene,
        frame_index: usize,
        device: &ash::Device,
        bindless_descriptor_set: vk::DescriptorSet,
        uniform_buffer_address: u64,
        instance_buffer_address: u64,
        material_heap_address: u64,
        light_ptr: u64,
        tile_ptr: u64,
        all_instances_len: usize,
    ) -> Result<()> {
        if let Some(shadow_system) = &mut self.shadow_system {
            if shadow_system
                .vsm_feature()
                .shadow_pipeline_layout()
                .is_some()
            {
                let light_dir = Vec3::from_slice(&scene.scene_lighting.directional.direction[0..3]);

                let (vertex_ptr, index_ptr) = scene.get_geometry_buffer_addresses();

                if vertex_ptr == 0 || index_ptr == 0 {
                    // Throttled warning to avoid spamming while still aiding diagnostics
                    log::warn!("Shadow Pass: Invalid Geometry BDA pointers (V: {}, I: {}). Shadows will be skipped.", vertex_ptr, index_ptr);
                    return Ok(());
                }
                let vsm_feature = shadow_system.vsm_feature();
                unsafe {
                    vsm_feature.render_shadows(
                        command_buffer,
                        light_dir,
                        all_instances_len as u32,
                        frame_index,
                        vk::DescriptorSet::null(),
                        bindless_descriptor_set,
                        uniform_buffer_address,
                        vertex_ptr,
                        instance_buffer_address,
                        material_heap_address,
                        index_ptr,
                        light_ptr,
                        tile_ptr,
                        scene.transform_system.arena_addr,
                        0,
                    );
                }

                // Synchronization barrier for VSM results
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

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[vsm_barrier],
                    );
                }
            }
        }
        Ok(())
    }

    /// Execute post-processing effects including blooming, tonemapping, and upscaling resolve.
    pub fn render_post_process(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        swapchain_extent: vk::Extent2D,
        target_image: vk::Image,
        target_view: vk::ImageView,
        hdr_system: Option<&HdrSystem>,
        vsr_pass: Option<&VsrPass>,
        black_texture_view: vk::ImageView,
    ) -> Result<()> {
        if let Some(hdr) = hdr_system {
            let input_view = vsr_pass
                .map(|vsr| vsr.active_view())
                .unwrap_or_else(|| hdr.view());

            self.post_process.update_descriptor_set(
                image_index,
                input_view,
                black_texture_view, // bloom_view placeholder
                black_texture_view, // ssgi_view placeholder
                hdr.sampler(),
            );

            // Final tonemapping and swapchain resolve
            self.post_process.render(
                command_buffer,
                image_index,
                swapchain_extent,
                target_image,
                target_view,
            )?;
        }

        Ok(())
    }

    // Accessors
    pub fn shadow_system(&self) -> Option<&ShadowSystem> {
        self.shadow_system.as_ref()
    }

    pub fn shadow_system_mut(&mut self) -> Option<&mut ShadowSystem> {
        self.shadow_system.as_mut()
    }

    pub fn take_shadow_system(&mut self) -> Option<ShadowSystem> {
        self.shadow_system.take()
    }

    pub fn post_process(&self) -> &PostProcessSystem {
        &self.post_process
    }

    pub fn post_process_mut(&mut self) -> &mut PostProcessSystem {
        &mut self.post_process
    }

    pub fn hiz_pass(&self) -> Option<&Arc<RwLock<HiZPass>>> {
        self.hiz_pass.as_ref()
    }

    /// Validate that all required pipelines and layouts are initialized.
    pub fn validate(&self) -> Result<()> {
        use crate::AshError;
        if self.main_graphics_pipeline.is_none() {
            return Err(AshError::VulkanError(
                "RenderPipeline: main_graphics_pipeline not initialized".to_string(),
            ));
        }
        if self.pipeline_layout.is_none() {
            return Err(AshError::VulkanError(
                "RenderPipeline: pipeline_layout not initialized".to_string(),
            ));
        }
        Ok(())
    }

    /// Render the main geometry pass.
    /// Manages its own rendering scope via Dynamic Rendering (cmd_begin_rendering).
    pub fn render_geometry(&self, ctx: &GeometryRenderContext) -> Result<()> {
        use crate::renderer::model_renderer::{
            DrawContext, IndirectDrawCountParams, MaterialPushConstants,
        };
        use crate::renderer::MaterialHandle;
        use crate::AshError;

        let pipeline_handle = self
            .main_graphics_pipeline
            .as_ref()
            .map(|p| p.pipeline)
            .ok_or_else(|| {
                AshError::VulkanError(
                    "RenderPipeline: main_graphics_pipeline not initialized".to_string(),
                )
            })?;
        let layout_handle = self
            .pipeline_layout
            .as_ref()
            .map(|l| l.handle())
            .ok_or_else(|| {
                AshError::VulkanError("RenderPipeline: pipeline_layout not initialized".to_string())
            })?;

        // Dynamics 1: Validate Views
        if ctx.color_view == vk::ImageView::null() || ctx.depth_view == vk::ImageView::null() {
            return Ok(());
        }

        let mut color_attachments = vec![vk::RenderingAttachmentInfo::default()
            .image_view(ctx.color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            })];

        // G-Buffer Attachments: Clear to transparent black
        let gbuffer_clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };

        // Attachment 1: Normals
        if let Some(view) = ctx.normal_view {
            color_attachments.push(
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(gbuffer_clear),
            );
        }

        // Attachment 2: Albedo
        if let Some(view) = ctx.albedo_view {
            color_attachments.push(
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(gbuffer_clear),
            );
        }

        // Attachment 3: Motion Vectors
        if let Some(view) = ctx.motion_view {
            color_attachments.push(
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(gbuffer_clear),
            );
        }

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(ctx.depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: ctx.swapchain_extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        unsafe {
            // Pre-Render Barrier: Transition Color Image to Attachment Optimal
            let color_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .image(ctx.color_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // Determine depth aspect mask based on format
            let depth_aspect = match ctx.depth_format {
                vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D16_UNORM_S8_UINT => {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                }
                _ => vk::ImageAspectFlags::DEPTH,
            };

            // Pre-Render Barrier: Transition Depth Image to Depth Stencil Attachment Optimal
            let depth_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .image(ctx.depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: depth_aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            ctx.device.device.cmd_pipeline_barrier(
                ctx.command_buffer.handle(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[color_barrier, depth_barrier],
            );

            ctx.device
                .device
                .cmd_begin_rendering(ctx.command_buffer.handle(), &rendering_info);
        }
        unsafe {
            ctx.device.device.cmd_bind_pipeline(
                ctx.command_buffer.handle(),
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_handle,
            );
        }

        // 2. Dynamic State
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: ctx.swapchain_extent.width as f32,
            height: ctx.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: ctx.swapchain_extent,
        };
        unsafe {
            ctx.device
                .device
                .cmd_set_viewport(ctx.command_buffer.handle(), 0, &[viewport]);
            ctx.device
                .device
                .cmd_set_scissor(ctx.command_buffer.handle(), 0, &[scissor]);
        }

        // 3. Bind Descriptor Sets
        // Unified Bindless (Set 0)
        unsafe {
            ctx.device.device.cmd_bind_descriptor_sets(
                ctx.command_buffer.handle(),
                vk::PipelineBindPoint::GRAPHICS,
                layout_handle,
                0,
                &[ctx.bindless_descriptor_set],
                &[],
            );
        }

        // 4. Draw
        if let Some(ref indirect_arc) = self.indirect_draw_pass {
            let indirect_pass = indirect_arc.read().map_err(|e| {
                log::error!("Indirect draw pass RwLock poisoned: {}", e);
                crate::AshError::VulkanError("Indirect draw pass RwLock poisoned".into())
            })?;

            if ctx.scene.occlusion_culling.object_count() > 0 {
                let (vertex_ptr, index_ptr) = ctx.scene.get_geometry_buffer_addresses();
                let instance_ptr = indirect_pass.object_buffer_address();

                if vertex_ptr == 0 || index_ptr == 0 || instance_ptr == 0 || ctx.material_ptr == 0 {
                    log::error!("CRITICAL: BDA Null Pointer in render_geometry. Skipping draw.");
                    return Ok(());
                }

                // Placeholder mesh for DrawContext (required but not used for BDA indirect)
                let Some(uploaded) = ctx
                    .scene
                    .model_renderer
                    .uploaded_meshes()
                    .next()
                    .map(|(_, m)| m)
                else {
                    return Ok(());
                };

                let material_push = MaterialPushConstants::new(MaterialHandle::null())
                    .with_receive_shadows(true)
                    .with_debug_visualization(ctx.debug_enabled);

                let draw_ctx = DrawContext {
                    command_buffer: ctx.command_buffer.handle(),
                    pipeline_layout: layout_handle,
                    uploaded,
                    material: &material_push,
                    frame_ptr: ctx.frame_ptr,
                    vertex_ptr,
                    instance_ptr,
                    material_ptr: ctx.material_ptr,
                    index_ptr,
                    light_ptr: ctx.light_ptr,
                    tile_ptr: ctx.tile_ptr,
                    skybox_index: ctx.scene.skybox_texture_index,
                    vsm_page_index: self
                        .shadow_system()
                        .map(|s| s.vsm_page_index())
                        .unwrap_or(0),
                    vsm_cache_index: self
                        .shadow_system()
                        .map(|s| s.vsm_cache_index())
                        .unwrap_or(0),
                    transform_ptr: ctx.scene.transform_system.arena_addr,
                    transform_index: 0,
                };

                let count_params = IndirectDrawCountParams {
                    indirect_buffer: indirect_pass.indirect_buffer(),
                    indirect_offset: 0,
                    count_buffer: indirect_pass.count_buffer(),
                    count_offset: 0,
                    max_draw_count: ctx.scene.occlusion_culling.object_count() as u32,
                    stride: std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
                };

                unsafe {
                    ctx.scene
                        .model_renderer
                        .draw_indirect_count(&draw_ctx, &count_params);
                }
            }
        }

        // INJECTION: Render Features within the active dynamic rendering pass
        if let Some(features) = ctx.features {
            let feature_ctx = crate::renderer::features::FeatureRenderContext {
                device: &ctx.device.device,
                descriptor_allocator: ctx.descriptor_allocator,
                command_buffer: ctx.command_buffer.handle(),
                transform: ctx.transform,
                frame_index: ctx.frame_index,
                screen_width: ctx.swapchain_extent.width,
                screen_height: ctx.swapchain_extent.height,
            };
            unsafe {
                features.render(&feature_ctx);
            }
        }

        // INJECTION: Render Skybox within the active dynamic rendering pass
        if let Some(skybox) = ctx.skybox {
            if ctx.scene.scene_lighting.ibl_prefilter_index >= 0 {
                if let Err(e) = unsafe {
                    skybox.render(
                        ctx.device,
                        ctx.command_buffer,
                        ctx.bindless_descriptor_set,
                        ctx.frame_ptr,
                    )
                } {
                    log::warn!("Skybox render failed: {e}");
                }
            }
        }

        unsafe {
            ctx.device
                .device
                .cmd_end_rendering(ctx.command_buffer.handle());

            // Post-Render Barrier: Transition Color Image based on whether it is swapchain or HDR
            let (target_layout, target_access, target_stage) = if ctx.is_swapchain_image {
                // Case 1: Direct to Swapchain (Ready for Presentation)
                (
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::AccessFlags::empty(),
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                )
            } else {
                // Case 2: Offscreen HDR (Ready for Sampling/Tonemapping)
                (
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                )
            };

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(target_layout)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(target_access)
                .image(ctx.color_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // Determine depth aspect mask based on format
            let depth_aspect = match ctx.depth_format {
                vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D16_UNORM_S8_UINT => {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                }
                _ => vk::ImageAspectFlags::DEPTH,
            };

            // Determine precise depth layouts based on format (Vulkan 1.3+ separate layouts)
            let depth_target_layout = match ctx.depth_format {
                vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D16_UNORM_S8_UINT => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                _ => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            };

            let depth_post_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .new_layout(depth_target_layout)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(ctx.depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: depth_aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            ctx.device.device.cmd_pipeline_barrier(
                ctx.command_buffer.handle(),
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                target_stage | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier, depth_post_barrier],
            );
        }

        Ok(())
    }
}
