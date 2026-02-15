use crate::renderer::{
    context::Context, resources::Resources, scene::Scene, vcgs::IndirectDrawPass,
};
use crate::Result;
use ash::vk;
use std::sync::{Arc, RwLock};

/// CullingSystem handles GPU-driven culling operations.
/// It orchestrates compute shaders that filter geometry before drawing.
pub struct CullingSystem {
    pub(crate) indirect_draw_pass: Option<Arc<RwLock<IndirectDrawPass>>>,
}

impl CullingSystem {
    pub fn new(indirect_draw_pass: Option<Arc<RwLock<IndirectDrawPass>>>) -> Self {
        Self { indirect_draw_pass }
    }

    /// Executes the culling compute pass and records necessary barriers.
    pub fn execute_culling(
        &self,
        cmd: vk::CommandBuffer,
        context: &Context,
        resources: &Resources,
        scene: &Scene,
        _frame_index: usize,
    ) -> Result<()> {
        let indirect_arc = match &self.indirect_draw_pass {
            Some(arc) => arc,
            None => return Ok(()),
        };

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
                // 1. Reset count buffer to 0 before compute pass
                device.cmd_fill_buffer(cmd, indirect_pass.count_buffer(), 0, 4, 0);

                // 2. Buffer Barrier: ensure fill is done before compute shader reads/writes
                let fill_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(indirect_pass.count_buffer())
                    .size(4)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[fill_barrier],
                    &[],
                );

                // 3. Dispatch Culling Compute Shader
                let cluster_buffer_addr = resources
                    .global_cluster_buffer
                    .as_ref()
                    .map(|b| b.device_address())
                    .unwrap_or(0);

                // Note: IndirectDrawPass::execute_culling also does some internal barriers,
                // but we explicitly manage the critical ones here for orchestrator clarity.
                indirect_pass.execute_culling(
                    cmd,
                    &scene.occlusion_culling,
                    resources.current_view_proj,
                    resources.swapchain_extent.width,
                    resources.swapchain_extent.height,
                    0,
                    scene.occlusion_culling.object_count() as u32,
                    0,
                    cluster_buffer_addr,
                )?;

                // 4. CRITICAL BARRIER: Compute-to-Graphics for Indirect Buffers
                // Transition DRAW_INDIRECT_BUFFER and COUNT_BUFFER from SHADER_WRITE to INDIRECT_COMMAND_READ
                let indirect_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
                    .buffer(indirect_pass.indirect_buffer())
                    .size(vk::WHOLE_SIZE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

                let count_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
                    .buffer(indirect_pass.count_buffer())
                    .size(vk::WHOLE_SIZE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::DRAW_INDIRECT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[indirect_barrier, count_barrier],
                    &[],
                );
            }
        }

        Ok(())
    }
}
