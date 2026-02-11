use ash::vk;
use std::sync::Arc;

/// Simplified command list wrapper with state tracking.
///
/// This wrapper provides a safer API over raw `vk::CommandBuffer` by tracking
/// bound state and preventing common errors like binding incompatible resources.
///
/// # Example
/// ```ignore
/// # use ash_renderer::renderer::CommandList;
/// let mut cmd = CommandList::new(device, command_buffer);
/// unsafe {
///     cmd.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline)?;
///     cmd.bind_descriptor_sets(vk::PipelineBindPoint::COMPUTE, layout, &[set])?;
///     cmd.dispatch(64, 64, 1);
/// }
/// ```
pub struct CommandList {
    device: Arc<ash::Device>,
    cmd: vk::CommandBuffer,
    state: CommandState,
}

#[derive(Default)]
struct CommandState {
    bound_compute_pipeline: Option<vk::Pipeline>,
    bound_graphics_pipeline: Option<vk::Pipeline>,
    bound_descriptor_sets: [Option<vk::DescriptorSet>; 4],
}

impl CommandList {
    /// Creates a new command list wrapper.
    ///
    /// # Safety
    ///
    /// The command buffer must be in the recording state and the device must
    /// remain valid for the lifetime of this wrapper.
    pub unsafe fn new(device: Arc<ash::Device>, cmd: vk::CommandBuffer) -> Self {
        Self {
            device,
            cmd,
            state: CommandState::default(),
        }
    }

    /// Returns the underlying command buffer handle
    pub fn handle(&self) -> vk::CommandBuffer {
        self.cmd
    }

    /// Binds a pipeline.
    ///
    /// # Safety
    ///
    /// The pipeline must be valid and compatible with the current render pass (if any).
    pub unsafe fn bind_pipeline(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) -> crate::Result<()> {
        match bind_point {
            vk::PipelineBindPoint::COMPUTE => {
                self.state.bound_compute_pipeline = Some(pipeline);
            }
            vk::PipelineBindPoint::GRAPHICS => {
                self.state.bound_graphics_pipeline = Some(pipeline);
            }
            _ => {
                return Err(crate::AshError::VulkanError(format!(
                    "Unsupported pipeline bind point: {bind_point:?}"
                )))
            }
        }

        self.device
            .cmd_bind_pipeline(self.cmd, bind_point, pipeline);
        Ok(())
    }

    /// Binds descriptor sets.
    ///
    /// # Safety
    ///
    /// The descriptor sets must be valid and compatible with the bound pipeline.
    pub unsafe fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
    ) -> crate::Result<()> {
        // Track bound sets
        for (i, &set) in descriptor_sets.iter().enumerate() {
            let set_index = (first_set as usize + i).min(3);
            self.state.bound_descriptor_sets[set_index] = Some(set);
        }

        self.device.cmd_bind_descriptor_sets(
            self.cmd,
            bind_point,
            layout,
            first_set,
            descriptor_sets,
            &[],
        );
        Ok(())
    }

    /// Dispatches a compute shader.
    ///
    /// # Safety
    ///
    /// A compute pipeline must be bound before calling this.
    pub unsafe fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        if self.state.bound_compute_pipeline.is_none() {
            log::warn!("Dispatching without bound compute pipeline");
        }
        self.device
            .cmd_dispatch(self.cmd, group_count_x, group_count_y, group_count_z);
    }

    /// Inserts a pipeline barrier (simplified API).
    ///
    /// # Safety
    ///
    /// Barrier parameters must be valid for the current command buffer state.
    pub unsafe fn pipeline_barrier(
        &mut self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        self.device.cmd_pipeline_barrier(
            self.cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            image_barriers,
        );
    }

    /// Pushes constants to the pipeline.
    ///
    /// # Safety
    ///
    /// The data must match the pipeline layout's push constant range.
    pub unsafe fn push_constants<T: bytemuck::Pod>(
        &mut self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        data: &T,
    ) {
        let bytes = bytemuck::bytes_of(data);
        self.device
            .cmd_push_constants(self.cmd, layout, stage_flags, offset, bytes);
    }

    /// Binds an index buffer.
    ///
    /// # Safety
    ///
    /// The buffer must be valid and contain index data.
    pub unsafe fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        if self.state.bound_graphics_pipeline.is_none() {
            log::warn!("Binding index buffer without bound graphics pipeline");
        }
        self.device
            .cmd_bind_index_buffer(self.cmd, buffer, offset, index_type);
    }

    /// Returns the current bound compute pipeline (if any)
    pub fn bound_compute_pipeline(&self) -> Option<vk::Pipeline> {
        self.state.bound_compute_pipeline
    }

    /// Returns the current bound graphics pipeline (if any)
    pub fn bound_graphics_pipeline(&self) -> Option<vk::Pipeline> {
        self.state.bound_graphics_pipeline
    }
}

impl std::fmt::Debug for CommandList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandList")
            .field("cmd", &self.cmd)
            .field("bound_compute_pipeline", &self.state.bound_compute_pipeline)
            .field(
                "bound_graphics_pipeline",
                &self.state.bound_graphics_pipeline,
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_tracking() {
        // Verify that state tracking logic is correct
        let mut state = CommandState::default();
        assert!(state.bound_compute_pipeline.is_none());

        state.bound_compute_pipeline = Some(vk::Pipeline::null());
        assert!(state.bound_compute_pipeline.is_some());
    }
}
