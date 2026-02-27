use crate::{
    Result,
    renderer::{
        ForwardPlusIntegration, Scene, features::vsm::VsmManager, passes::hiz::HiZPass,
        systems::post_process::PostProcessSystem, vcgs::IndirectDrawPass,
    },
};
use ash::vk;
use std::sync::{Arc, RwLock};

/// RenderPipeline orchestrates the high-level rendering flow.
/// It owns the major rendering subsystems and manages their execution order.
pub struct RenderPipeline {
    pub post_process: PostProcessSystem,
    pub(crate) hiz_pass: Arc<RwLock<HiZPass>>,

    pub forward_plus: Arc<RwLock<ForwardPlusIntegration>>,
    pub indirect_draw_pass: Arc<RwLock<IndirectDrawPass>>,
    pub main_graphics_pipeline: crate::vulkan::Pipeline,
    pub pipeline_layout: crate::vulkan::PipelineLayout,
}

/// Context for rendering geometry, grouping multiple parameters to stabilize the API.
pub struct GeometryRenderContext<'a> {
    pub device: &'a crate::vulkan::VulkanDevice,
    pub command_buffer: &'a crate::vulkan::CommandBufferContext<'a>,
    pub scene: &'a Scene,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub vsm_manager: &'a VsmManager, // Added to provide bindless indices
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
    pub motion_image: Option<vk::Image>,
    pub motion_view: Option<vk::ImageView>,
    pub skybox: Option<&'a crate::renderer::passes::SkyboxPass>,
    pub features: Option<&'a crate::renderer::features::FeatureManager>,
    pub frame_index: usize,
    pub descriptor_allocator: Option<&'a crate::vulkan::DescriptorAllocator>,
    pub transform: &'a crate::renderer::Transform,
    pub depth_format: vk::Format,
    pub vsm_ptr: u64,
    /// Pre-extracted BDA fields — populated once under a short-lived read-lock
    /// before any HiZ work. No RwLock is held when render_main_view runs.
    pub indirect_buffer: vk::Buffer,
    pub count_buffer: vk::Buffer,
    /// Cached object-buffer device address (u64 BDA, always valid after init).
    pub instance_ptr: u64,
}

impl RenderPipeline {
    pub fn new(
        post_process: PostProcessSystem,
        hiz_pass: Arc<RwLock<HiZPass>>,
        forward_plus: Arc<RwLock<ForwardPlusIntegration>>,
        indirect_draw_pass: Arc<RwLock<IndirectDrawPass>>,
        main_graphics_pipeline: crate::vulkan::Pipeline,
        pipeline_layout: crate::vulkan::PipelineLayout,
    ) -> Self {
        Self {
            post_process,
            hiz_pass,
            forward_plus,
            indirect_draw_pass,
            main_graphics_pipeline,
            pipeline_layout,
        }
    }

    /// Execute the Hi-Z depth pyramid construction pass.
    ///
    /// Delegates directly to [`HiZPass::record_commands`] which now owns the
    /// full lifecycle: adaptive quality, pyramid generation, and descriptor routing.
    ///
    /// # Lock ordering
    /// Acquires the `hiz_pass` write-lock; the pass acquires the
    /// `indirect_draw_pass` write-lock internally after all Hi-Z work.
    pub fn execute_hiz_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        depth_view: vk::ImageView,
        gpu_profiler: Option<&crate::renderer::diagnostics::GpuProfiler>,
    ) -> Result<()> {
        let mut hiz = self.hiz_pass.write().map_err(|e| {
            log::error!("Hi-Z pass RwLock poisoned: {e}");
            crate::AshError::VulkanError("Hi-Z pass RwLock poisoned".into())
        })?;

        let hiz_time_ms = gpu_profiler
            .map(|p| p.last_extended_timings())
            .filter(|t| t.valid)
            .map(|t| t.hiz_generate_ms as f64);

        unsafe {
            hiz.record_commands(command_buffer, depth_view, hiz_time_ms)?;

            // Emit end-of-pass GPU timestamp if profiling is active.
            if let Some(profiler) = gpu_profiler {
                profiler.write_timestamp(
                    command_buffer,
                    crate::renderer::diagnostics::TimingScope::HiZGenerateEnd,
                );
            }
        }

        Ok(())
    }

    pub fn post_process(&self) -> &PostProcessSystem {
        &self.post_process
    }

    pub fn post_process_mut(&mut self) -> &mut PostProcessSystem {
        &mut self.post_process
    }

    pub fn hiz_pass(&self) -> &Arc<RwLock<HiZPass>> {
        &self.hiz_pass
    }

    /// Validate that all required pipelines and layouts are initialized.
    pub fn validate(&self) -> Result<()> {
        Ok(())
    }

    /// Render the main geometry pass.
    /// Manages its own rendering scope via Dynamic Rendering (cmd_begin_rendering).
    pub fn render_geometry(
        &self,
        ctx: &GeometryRenderContext,
        scene: &Scene,
        frame_index: usize,
    ) -> Result<()> {
        // Dynamics 1: Validate Views
        if ctx.color_view == vk::ImageView::null() || ctx.depth_view == vk::ImageView::null() {
            return Ok(());
        }

        let mut color_attachments = vec![
            vk::RenderingAttachmentInfo::default()
                .image_view(ctx.color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }),
        ];

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
            // Pre-Render Barrier: Transition Color and Depth images using Synchronization2
            let color_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(ctx.color_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let depth_aspect = get_depth_aspect_mask(ctx.depth_format);
            let depth_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(ctx.depth_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: depth_aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_barriers = [color_barrier, depth_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
            ctx.command_buffer.pipeline_barrier2(&dep_info);

            ctx.device
                .device
                .cmd_begin_rendering(ctx.command_buffer.handle(), &rendering_info);
        }

        // DELEGATED RECORDING
        self.render_main_view(ctx.command_buffer.handle(), ctx, scene, frame_index)?;

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
        }

        // Post-Render Barrier: Transition HDR buffer for Post-Processing (Sync2)
        let (target_layout, target_access, target_stage) = (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        let color_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(target_stage)
            .dst_access_mask(target_access)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(target_layout)
            .image(ctx.color_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let depth_aspect = get_depth_aspect_mask(ctx.depth_format);
        // D32_SFLOAT uses DEPTH_READ_ONLY_OPTIMAL; D32_SFLOAT_S8_UINT needs DEPTH_STENCIL variant.
        let depth_target_layout = match ctx.depth_format {
            vk::Format::D32_SFLOAT_S8_UINT => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            _ => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
        };

        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .new_layout(depth_target_layout)
            .image(ctx.depth_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: depth_aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let mut image_barriers = vec![color_barrier, depth_barrier];

        if let Some(motion_image) = ctx.motion_image {
            image_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::FRAGMENT_SHADER
                            | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    )
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(motion_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            );
        }

        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
        ctx.command_buffer.pipeline_barrier2(&dep_info);

        Ok(())
    }

    pub fn render_main_view(
        &self,
        cmd: vk::CommandBuffer,
        ctx: &GeometryRenderContext,
        scene: &Scene,
        _frame_index: usize,
    ) -> Result<()> {
        use crate::renderer::MaterialHandle;
        use crate::renderer::model_renderer::{
            DrawContext, IndirectDrawCountParams, MaterialPushConstants,
        };

        let pipeline_handle = self.main_graphics_pipeline.pipeline;
        let layout_handle = self.pipeline_layout.handle();

        // 1. Dynamic State
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
            ctx.device.device.cmd_set_viewport(cmd, 0, &[viewport]);
            ctx.device.device.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        // 2. Bind Pipeline
        unsafe {
            ctx.device.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_handle,
            );
        }

        // 3. Bind Descriptors
        unsafe {
            ctx.device.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout_handle,
                0,
                &[ctx.bindless_descriptor_set],
                &[],
            );
        }

        // 4. Draw Dispatch
        // NOTE: No RwLock acquired here. indirect_buffer, count_buffer, and instance_ptr
        // are pre-extracted from IndirectDrawPass under a short-lived read-lock at frame
        // start in record_and_submit, before any HiZ work begins. This eliminates the
        // ABBA lock-order inversion between indirect_draw_pass and hiz_pass.

        if scene.occlusion_culling.object_count() > 0 {
            let (vertex_ptr, index_ptr) = scene.get_geometry_buffer_addresses();
            let instance_ptr = ctx.instance_ptr;

            if vertex_ptr == 0 || index_ptr == 0 || instance_ptr == 0 || ctx.material_ptr == 0 {
                log::error!("CRITICAL: BDA Null Pointer in render_main_view. Skipping draw.");
                return Ok(());
            }

            // Placeholder mesh for DrawContext (required but not used for BDA indirect)
            let Some(uploaded) = scene
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
                command_buffer: cmd,
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
                skybox_index: scene.skybox_texture_index,
                vsm_page_index: ctx.vsm_manager.page_table_bindless_index,
                vsm_cache_index: ctx.vsm_manager.physical_memory_bindless_index,
                transform_ptr: scene.transform_system.arena_addr,
                transform_index: 0,
                vsm_ptr: ctx.vsm_ptr,
            };

            let count_params = IndirectDrawCountParams {
                indirect_buffer: ctx.indirect_buffer,
                indirect_offset: 0,
                count_buffer: ctx.count_buffer,
                count_offset: 0,
                max_draw_count: scene.occlusion_culling.object_count() as u32,
                stride: std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
            };

            unsafe {
                scene.model_renderer.draw_indirect(&draw_ctx, &count_params);
            }
        }

        Ok(())
    }
}

fn get_depth_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    // Modern path: D32_SFLOAT (no stencil), D32_SFLOAT_S8_UINT (stencil).
    // Legacy D24/D16 formats are not supported by this renderer.
    match format {
        vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::DEPTH,
    }
}
