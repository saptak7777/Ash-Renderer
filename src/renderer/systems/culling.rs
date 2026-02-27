use crate::Result;
use crate::renderer::{
    context::Context, resources::Resources, scene::Scene, vcgs::IndirectDrawPass,
};
use ash::vk;
use std::sync::{Arc, RwLock};

/// CullingSystem handles GPU-driven culling operations.
/// It orchestrates compute shaders that filter geometry before drawing.
pub struct CullingSystem {
    pub(crate) indirect_draw_pass: Arc<RwLock<IndirectDrawPass>>,
}

impl CullingSystem {
    pub fn new(indirect_draw_pass: Arc<RwLock<IndirectDrawPass>>) -> Self {
        Self { indirect_draw_pass }
    }

    /// Executes the culling compute pass and records necessary barriers.
    pub fn execute_culling(
        &self,
        cmd: vk::CommandBuffer,
        context: &Context,
        resources: &Resources,
        scene: &Scene,
        hiz_buffer_addr: u64,
        frame_index: usize,
    ) -> Result<()> {
        let indirect_arc = &self.indirect_draw_pass;

        let indirect_pass = indirect_arc.read().map_err(|e| {
            crate::AshError::VulkanError(format!(
                "CullingSystem: Indirect draw pass lock poisoned: {e}"
            ))
        })?;

        if !indirect_pass.is_initialized() {
            return Ok(());
        }

        // Only dispatch if there is something to cull
        if scene.occlusion_culling.object_count() > 0 {
            let device = &context.device.device;

            unsafe {
                // 1. Dispatch Culling Compute Shader

                // Note: IndirectDrawPass::execute_culling handles the count buffer reset and synchronization internally
                let camera_buffer_addr = resources.uniform_buffers
                    [frame_index % resources.uniform_buffers.len()]
                .read()
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Camera buffer lock poisoned: {e}"))
                })?
                .device_address();

                let culling_ctx = crate::renderer::types::CullingContext {
                    view_proj: resources.current_view_proj,
                    width: resources.swapchain_extent.width,
                    height: resources.swapchain_extent.height,
                    object_offset: 0,
                    object_count: scene.occlusion_culling.object_count() as u32,
                    indirect_offset: 0,
                    hiz_buffer_addr,
                    camera_buffer_addr,
                };

                indirect_pass.execute_culling(cmd, &scene.occlusion_culling, &culling_ctx)?;

                // 4. CRITICAL BARRIER: Compute-to-Graphics for Indirect Buffers (Sync2)
                let indirect_barrier = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .buffer(indirect_pass.indirect_buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);

                let count_barrier = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .buffer(indirect_pass.count_buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE);

                let buffer_barriers = [indirect_barrier, count_barrier];
                let dep_info =
                    vk::DependencyInfo::default().buffer_memory_barriers(&buffer_barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }
        }

        Ok(())
    }
}
