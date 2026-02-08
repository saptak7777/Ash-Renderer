use ash::vk;
use std::sync::Arc;

use crate::vulkan::{CommandBufferManager, FrameSync, SwapchainWrapper as Swapchain};
use crate::Result;

/// Specialized queue wrapper for rendering setup and submission.
/// Also manages per-frame synchronization and command buffers.
pub struct RenderQueue {
    device: Arc<ash::Device>,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // Phase 1: Frame Synchronization & Commands
    pub frame_syncs: Vec<FrameSync>,
    pub cmds: CommandBufferManager,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub current_frame: usize,
}

impl RenderQueue {
    pub fn new(
        device: Arc<ash::Device>,
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        queue_family_index: u32,
        frames_in_flight: usize,
        worker_count: usize,
    ) -> Result<Self> {
        let cmds =
            CommandBufferManager::new(Arc::clone(&device), queue_family_index, worker_count)?;

        let command_buffers = cmds.allocate_primary_buffers(frames_in_flight as u32)?;

        let mut frame_syncs = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            frame_syncs.push(FrameSync::new(Arc::clone(&device))?);
        }

        Ok(Self {
            device,
            graphics_queue,
            present_queue,
            frame_syncs,
            cmds,
            command_buffers,
            current_frame: 0,
        })
    }

    /// Acquires the next frame for rendering.
    /// Returns (image_index, command_buffer, (image_available, render_finished, in_flight_fence))
    /// This method blocks until the fence for the current in-flight frame is signaled.
    pub fn acquire_next_frame(
        &mut self,
        swapchain: &Swapchain,
    ) -> Result<(
        u32,
        vk::CommandBuffer,
        vk::Semaphore,
        vk::Semaphore,
        vk::Fence,
    )> {
        let frame_index = self.current_frame;
        let sync = &self.frame_syncs[frame_index];

        // 1. Wait for and reset fence
        sync.wait()?;
        sync.reset()?;

        // 2. Acquire next swapchain image
        let image_index = unsafe { swapchain.acquire_next_image(sync.image_available)? };

        // 3. Get command buffer for this frame
        let command_buffer = self.command_buffers[frame_index];
        let image_available = sync.image_available;
        let render_finished = sync.render_finished;
        let in_flight_fence = sync.in_flight;

        Ok((
            image_index,
            command_buffer,
            image_available,
            render_finished,
            in_flight_fence,
        ))
    }

    /// Advances to the next frame in the rotation.
    pub fn advance_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.frame_syncs.len();
    }

    /// Submit command buffers to the graphics queue
    pub fn submit(
        &self,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> Result<()> {
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], fence)?;
        }

        Ok(())
    }

    /// Present swapchain image
    /// Returns true if swapchain needs resize (Suboptimal or OutOfDate)
    pub fn present(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool> {
        if wait_semaphores.is_empty() {
            return Err(crate::AshError::VulkanError(
                "No wait semaphore provided for presentation".to_string(),
            ));
        }

        let result =
            unsafe { swapchain.present(self.present_queue, image_index, wait_semaphores[0]) };

        match result {
            Ok(()) => Ok(false),
            Err(crate::AshError::SwapchainOutOfDate(_)) => Ok(true),
            Err(e) => Err(e),
        }
    }
}
