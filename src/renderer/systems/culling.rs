use crate::Result;
use crate::renderer::{
    context::Context, resources::Resources, scene::Scene, vcgs::IndirectDrawPass,
};
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
        hiz_buffer_addr: u64,
        frame_index: usize,
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
                // 1. Dispatch Culling Compute Shader
                let cluster_buffer_addr = resources
                    .global_cluster_buffer
                    .as_ref()
                    .map(|b| b.device_address())
                    .unwrap_or(0);

                // Note: IndirectDrawPass::execute_culling handles the count buffer reset and synchronization internally
                let camera_buffer_addr = resources.uniform_buffers[frame_index]
                    .read()
                    .map_err(|e| {
                        crate::AshError::VulkanError(format!("Camera buffer lock poisoned: {}", e))
                    })?
                    .device_address();

                let culling_ctx = crate::renderer::types::CullingContext {
                    view_proj: resources.current_view_proj,
                    width: resources.swapchain_extent.width,
                    height: resources.swapchain_extent.height,
                    object_offset: 0,
                    object_count: scene.occlusion_culling.object_count() as u32,
                    indirect_offset: 0,
                    cluster_buffer_addr,
                    hiz_buffer_addr,
                    camera_buffer_addr,
                };

                indirect_pass.execute_culling(cmd, &scene.occlusion_culling, &culling_ctx)?;

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
