use crate::{
    renderer::{model_renderer::UploadedMesh, ResourceId, ResourceRegistry},
    vulkan::{self, CommandBufferContext, MultisampleConfig},
    AshError, Result,
};
use ash::vk;
use std::sync::Arc;

/// Push constants for skybox rendering
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyboxPushConstants {
    frame_ptr: u64,
    skybox_index: u32,
    _pad: u32, // Explicit padding for 8-byte alignment of u64 vertex_heap_ptr
    vertex_heap_ptr: u64,
}

/// Self-contained skybox rendering pass.
///
/// Follows the two-phase initialization pattern:
/// 1. `new()` – Logical construction. Owns the mesh and bindless index; Vulkan
///    pipeline handles are `None` (or null) until `init` is called.
/// 2. `init()` – Vulkan resource allocation. Creates the pipeline layout and
///    graphics pipeline. Called once the swapchain format is known.
pub struct SkyboxPass {
    // Logical data — always present after `new()`
    mesh: UploadedMesh,
    bindless_index: u32,

    // Vulkan resources — present only after `init()`
    pipeline: Option<vulkan::Pipeline>,
    pipeline_layout: Option<vulkan::PipelineLayout>,
    pipeline_id: Option<ResourceId>,
}

/// Parameters for skybox pass initialization.
pub struct SkyboxInitContext<'a> {
    pub device: &'a vulkan::VulkanDevice,
    pub resources: &'a Arc<ResourceRegistry>,
    pub color_format: vk::Format,
    pub extent: vk::Extent2D,
    pub pipeline_cache: vk::PipelineCache,
    pub depth_format: vk::Format,
    pub multisample_config: MultisampleConfig,
    pub set_layouts: &'a [vk::DescriptorSetLayout],
}

impl SkyboxPass {
    /// Phase 1: Logical construction.
    ///
    /// Creates the pass descriptor with mesh data and the bindless texture
    /// index. No GPU resources are allocated here — call `init` afterwards.
    pub fn new(mesh: UploadedMesh, bindless_index: u32) -> Self {
        Self {
            mesh,
            bindless_index,
            pipeline: None,
            pipeline_layout: None,
            pipeline_id: None,
        }
    }

    /// Phase 2: Vulkan resource allocation.
    ///
    /// Creates the pipeline layout (with push constants) and the graphics
    /// pipeline for skybox rendering with Dynamic Rendering.
    ///
    /// # Safety
    /// - `device` must be valid for the lifetime of this pass.
    /// - `color_format` must match the format used in Dynamic Rendering.
    /// - `set_layouts` must match the shader's expected descriptor set layout.
    /// - Must be called exactly once before `render()`.
    pub unsafe fn init(&mut self, ctx: SkyboxInitContext) -> Result<()> {
        // ── Pipeline Layout ────────────────────────────────────────────────
        let mut layout_builder = vulkan::PipelineLayout::builder(Arc::clone(&ctx.device.device));
        for layout in ctx.set_layouts {
            layout_builder = layout_builder.add_set_layout(*layout);
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SkyboxPushConstants>() as u32);
        layout_builder = layout_builder.add_push_constant(push_range);

        let mut pipeline_layout = layout_builder.build()?;
        let pipeline_layout_id = ctx
            .resources
            .register_pipeline_layout(pipeline_layout.handle())
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to register skybox pipeline layout: {e}"))
            })?;
        pipeline_layout.mark_managed_by_registry();

        // ── Graphics Pipeline ──────────────────────────────────────────────
        // REVERSE-Z: Skybox renders at depth 0.0 (Far Plane).
        // GREATER_OR_EQUAL correctly handles z=0.0 vs clear depth of 0.0.
        let mut pipeline = vulkan::Pipeline::builder(Arc::clone(&ctx.device.device))
            .with_layout(pipeline_layout.handle())
            .with_dynamic_rendering(&[ctx.color_format], Some(ctx.depth_format), None)
            .with_extent(ctx.extent)
            .with_pipeline_cache(ctx.pipeline_cache)
            .with_depth_format(ctx.depth_format)
            .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, false)
            .with_cull_mode(vk::CullModeFlags::FRONT) // Inside cube
            .with_front_face(vk::FrontFace::CLOCKWISE)
            .with_multisampling(ctx.multisample_config)
            .add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/skybox.vert.spv")),
                vk::ShaderStageFlags::VERTEX,
                "main",
            )?
            .add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/skybox.frag.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                "main",
            )?
            .build()?;

        let pipeline_id = ctx
            .resources
            .register_pipeline(pipeline.pipeline, &[pipeline_layout_id])
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to register skybox pipeline: {e}"))
            })?;
        pipeline.mark_managed_by_registry();

        self.pipeline_layout = Some(pipeline_layout);
        self.pipeline = Some(pipeline);
        self.pipeline_id = Some(pipeline_id);

        log::debug!("SkyboxPass initialized.");
        Ok(())
    }

    /// Cleanup skybox-specific GPU resources.
    ///
    /// # Safety
    /// The GPU must not be using the skybox pipeline when this is called.
    pub unsafe fn cleanup(&self, resources: &Arc<ResourceRegistry>) -> Result<()> {
        if let Some(id) = self.pipeline_id {
            resources.cleanup_resource(id).map_err(|e| {
                AshError::VulkanError(format!("Failed to cleanup skybox pipeline: {e}"))
            })?;
        }
        Ok(())
    }

    /// Record the skybox draw commands into the current command buffer.
    ///
    /// # Safety
    /// - Must be called within a render pass with Dynamic Rendering active.
    /// - `cmd_ctx` must be in recording state.
    /// - `frame_ptr` must point to a valid, GPU-visible uniform buffer.
    /// - `bindless_set` must remain valid for the duration of execution.
    /// - `init()` must have been called successfully before this.
    pub unsafe fn render(
        &self,
        device: &vulkan::VulkanDevice,
        cmd_ctx: &CommandBufferContext,
        bindless_set: vk::DescriptorSet,
        frame_ptr: u64,
    ) -> Result<()> {
        let (pipeline, layout) = match (&self.pipeline, &self.pipeline_layout) {
            (Some(p), Some(l)) => (p, l),
            _ => {
                log::error!("SkyboxPass::render called before init(). Skipping.");
                return Ok(());
            }
        };

        cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

        let sets = [bindless_set];
        device.device.cmd_bind_descriptor_sets(
            cmd_ctx.handle(),
            vk::PipelineBindPoint::GRAPHICS,
            layout.handle(),
            0,
            &sets,
            &[],
        );

        let vertex_ptr = self.mesh.vertex_heap_address.unwrap_or(0);

        // CRITICAL BDA SAFETY: null vertex heap address will cause DEVICE_LOST.
        if vertex_ptr == 0 {
            log::error!(
                "CRITICAL: Skybox mesh has null BDA (vertex_heap_address=0). Skipping skybox draw."
            );
            return Ok(());
        }

        let push = SkyboxPushConstants {
            frame_ptr,
            skybox_index: self.bindless_index,
            _pad: 0,
            vertex_heap_ptr: vertex_ptr,
        };

        device.device.cmd_push_constants(
            cmd_ctx.handle(),
            layout.handle(),
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&push),
        );

        // Draw skybox (36 vertices for a cube, no index buffer needed)
        device.device.cmd_draw(cmd_ctx.handle(), 36, 1, 0, 0);

        Ok(())
    }

    /// Returns the bindless index for the skybox cubemap texture.
    pub fn bindless_index(&self) -> u32 {
        self.bindless_index
    }
}
