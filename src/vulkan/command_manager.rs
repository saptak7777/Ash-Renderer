use ash::vk;
use parking_lot::Mutex;
use std::sync::Arc;

use super::command::CommandPool;
#[cfg(feature = "parallel")]
use super::parallel_command::ParallelCommandManager;
use crate::{AshError, Result};

/// Thread-safe command buffer manager that can optionally leverage parallel recording helpers.
pub struct CommandBufferManager {
    #[cfg(feature = "parallel")]
    parallel_manager: ParallelCommandManager,
    device: Arc<ash::Device>,
    queue_family_index: u32,
    upload_pool: Mutex<CommandPool>,
}

/// Lightweight helper for recording commands without exposing raw `ash::Device` operations.
pub struct CommandBufferContext<'a> {
    device: &'a ash::Device,
    command_buffer: vk::CommandBuffer,
}

impl<'a> CommandBufferContext<'a> {
    pub fn handle(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to reset command buffer: {e}"))
                })?
        }
        Ok(())
    }

    pub fn begin(&self, flags: vk::CommandBufferUsageFlags) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default().flags(flags);
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
                })?
        }
        Ok(())
    }

    pub fn end(&self) -> Result<()> {
        unsafe {
            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?
        }
        Ok(())
    }

    pub fn set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, first_viewport, viewports);
        }
    }

    pub fn set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device
                .cmd_set_scissor(self.command_buffer, first_scissor, scissors);
        }
    }

    pub fn begin_render_pass(
        &self,
        begin_info: &vk::RenderPassBeginInfo,
        contents: vk::SubpassContents,
    ) {
        unsafe {
            self.device
                .cmd_begin_render_pass(self.command_buffer, begin_info, contents);
        }
    }

    pub fn end_render_pass(&self) {
        unsafe {
            self.device.cmd_end_render_pass(self.command_buffer);
        }
    }

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.command_buffer, bind_point, pipeline);
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                bind_point,
                layout,
                first_set,
                sets,
                dynamic_offsets,
            );
        }
    }

    /// Dispatch compute work groups.
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device.cmd_dispatch(
                self.command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    /// Insert a pipeline barrier for synchronization.
    pub fn pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }
}

impl CommandBufferManager {
    pub fn new(
        device: Arc<ash::Device>,
        queue_family_index: u32,
        thread_count: usize,
    ) -> Result<Self> {
        #[cfg(feature = "parallel")]
        let parallel_manager = ParallelCommandManager::new(
            Arc::clone(&device),
            queue_family_index,
            thread_count.max(1),
        )?;

        #[cfg(not(feature = "parallel"))]
        let _ = thread_count;

        let upload_pool = CommandPool::new(Arc::clone(&device), queue_family_index)?;

        Ok(Self {
            #[cfg(feature = "parallel")]
            parallel_manager,
            device,
            queue_family_index,
            upload_pool: Mutex::new(upload_pool),
        })
    }

    pub fn device(&self) -> &Arc<ash::Device> {
        &self.device
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn allocate_primary_buffers(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        #[cfg(feature = "parallel")]
        {
            self.parallel_manager.allocate_primary_buffers(count)
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut pool = self.upload_pool.lock();
            pool.allocate_primary_buffers(count)
        }
    }

    pub fn upload_command_pool_handle(&self) -> vk::CommandPool {
        self.upload_pool.lock().handle()
    }

    pub fn mark_pool_managed_by_registry(&self) {
        self.upload_pool.lock().mark_managed_by_registry();
    }

    pub fn context<'a>(&'a self, command_buffer: vk::CommandBuffer) -> CommandBufferContext<'a> {
        CommandBufferContext {
            device: self.device.as_ref(),
            command_buffer,
        }
    }

    pub fn reset_primary_pool(&self, flags: vk::CommandPoolResetFlags) -> Result<()> {
        let mut pool = self.upload_pool.lock();
        pool.reset(flags)
    }

    pub fn submit(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<()> {
        unsafe {
            self.device
                .queue_submit(queue, submits, fence)
                .map_err(|e| AshError::VulkanError(format!("Failed to submit commands: {e}")))?
        }
        Ok(())
    }

    pub fn destroy_pools(&self) -> Result<()> {
        #[cfg(feature = "parallel")]
        {
            self.parallel_manager.destroy_all_pools()
        }

        #[cfg(not(feature = "parallel"))]
        {
            Ok(())
        }
    }

    pub fn worker_count(&self) -> usize {
        #[cfg(feature = "parallel")]
        {
            self.parallel_manager.worker_count()
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    }

    pub fn acquire_secondary(&self, worker_index: usize) -> Result<vk::CommandBuffer> {
        #[cfg(feature = "parallel")]
        {
            self.parallel_manager.acquire_secondary(worker_index)
        }

        #[cfg(not(feature = "parallel"))]
        {
            let _ = worker_index;
            let pool = self.upload_pool.lock();
            let mut buffers = pool.allocate_secondary_buffers(1)?;
            buffers.pop().ok_or_else(|| {
                AshError::VulkanError("Failed to allocate secondary command buffer".into())
            })
        }
    }

    pub fn recycle_secondary(&self, worker_index: usize, buffer: vk::CommandBuffer) -> Result<()> {
        if buffer == vk::CommandBuffer::null() {
            return Ok(());
        }

        #[cfg(feature = "parallel")]
        {
            self.parallel_manager
                .recycle_secondary(worker_index, buffer)
        }

        #[cfg(not(feature = "parallel"))]
        {
            let _ = worker_index;
            unsafe {
                self.device
                    .reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to reset secondary command buffer: {e}"
                        ))
                    })?
            }
            Ok(())
        }
    }
}
