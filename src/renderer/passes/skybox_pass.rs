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

/// Self-contained skybox rendering pass
///
/// Manages skybox pipeline, mesh, and rendering logic.
/// Extracted from the monolithic `Renderer` struct to improve modularity.
pub struct SkyboxPass {
    pipeline: vulkan::Pipeline,
    pipeline_layout: vulkan::PipelineLayout,
    mesh: UploadedMesh,
    bindless_index: u32,
    pipeline_id: ResourceId,
}

impl SkyboxPass {
    /// Create a new skybox pass
    ///
    /// # Safety
    /// - `device` must be valid
    /// - `render_pass` must be compatible with the skybox shaders
    /// - `set_layouts` must match the expected descriptor set layout
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        device: &vulkan::VulkanDevice,
        resources: &Arc<ResourceRegistry>,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        pipeline_cache: vk::PipelineCache,
        depth_format: vk::Format,
        multisample_config: MultisampleConfig,
        set_layouts: &[vk::DescriptorSetLayout],
        mesh: UploadedMesh,
        bindless_index: u32,
    ) -> Result<Self> {
        // Create pipeline layout
        let mut pipeline_layout_builder =
            vulkan::PipelineLayout::builder(Arc::clone(&device.device));
        for layout in set_layouts {
            pipeline_layout_builder = pipeline_layout_builder.add_set_layout(*layout);
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SkyboxPushConstants>() as u32);
        pipeline_layout_builder = pipeline_layout_builder.add_push_constant(push_range);

        let mut pipeline_layout = pipeline_layout_builder.build()?;
        let pipeline_layout_id = resources
            .register_pipeline_layout(pipeline_layout.handle())
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to register skybox pipeline layout: {e}"))
            })?;
        pipeline_layout.mark_managed_by_registry();

        // Create pipeline
        let pipeline_builder = vulkan::Pipeline::builder(Arc::clone(&device.device))
            .with_layout(pipeline_layout.handle())
            .with_render_pass(render_pass)
            .with_extent(extent)
            .with_pipeline_cache(pipeline_cache)
            .with_depth_format(depth_format)
            // REVERSE-Z: Skybox at 0.0 (Far)
            // Depth Test: GREATER_OR_EQUAL handles z=0.0 (far) vs z=0.0 (clear) correctly.
            .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, false)
            .with_cull_mode(vk::CullModeFlags::FRONT) // Inside cube
            .with_front_face(vk::FrontFace::CLOCKWISE)
            .with_multisampling(multisample_config)
            .add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/skybox.vert.spv")),
                vk::ShaderStageFlags::VERTEX,
                "main",
            )?
            .add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/skybox.frag.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                "main",
            )?;

        let mut pipeline = pipeline_builder.build()?;
        let pipeline_id = resources
            .register_pipeline(pipeline.pipeline, &[pipeline_layout_id])
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to register skybox pipeline: {e}"))
            })?;
        pipeline.mark_managed_by_registry();

        Ok(Self {
            pipeline,
            pipeline_layout,
            mesh,
            bindless_index,
            pipeline_id,
        })
    }

    /// Cleanup skybox-specific resources
    ///
    /// # Safety
    /// - The GPU must not be using the skybox pipeline
    pub unsafe fn cleanup(&self, resources: &Arc<ResourceRegistry>) -> Result<()> {
        resources.cleanup_resource(self.pipeline_id).map_err(|e| {
            AshError::VulkanError(format!("Failed to cleanup skybox pipeline: {e}"))
        })?;
        Ok(())
    }

    /// Render the skybox
    ///
    /// # Safety
    /// - Must be called within a render pass
    /// - `cmd_ctx` must be in recording state
    /// - `frame_ptr` must point to valid uniform buffer
    /// - `bindless_set` must remain valid for the duration of execution
    pub unsafe fn render(
        &self,
        device: &vulkan::VulkanDevice,
        cmd_ctx: &CommandBufferContext,
        bindless_set: vk::DescriptorSet,
        frame_ptr: u64,
    ) -> Result<()> {
        // Bind pipeline
        cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);

        // Bind descriptor sets (bindless)
        let sets = [bindless_set];
        device.device.cmd_bind_descriptor_sets(
            cmd_ctx.handle(),
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout.handle(),
            0,
            &sets,
            &[],
        );

        // Push constants
        let vertex_ptr = self.mesh.vertex_heap_address.unwrap_or(0);

        // CRITICAL BDA SAFETY: Check for null vertex heap address
        if vertex_ptr == 0 {
            log::error!(
                "CRITICAL: Skybox mesh has null BDA (vertex_heap_address=0). Skipping skybox draw to prevent DEVICE_LOST."
            );
            return Ok(());
        }

        let push = SkyboxPushConstants {
            frame_ptr,
            skybox_index: self.bindless_index,
            _pad: 0,
            vertex_heap_ptr: vertex_ptr,
        };

        let push_bytes = bytemuck::bytes_of(&push);

        device.device.cmd_push_constants(
            cmd_ctx.handle(),
            self.pipeline_layout.handle(),
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        // Draw skybox (36 vertices for cube)
        device.device.cmd_draw(cmd_ctx.handle(), 36, 1, 0, 0);

        Ok(())
    }

    /// Get the bindless index for the skybox cubemap
    pub fn bindless_index(&self) -> u32 {
        self.bindless_index
    }
}
