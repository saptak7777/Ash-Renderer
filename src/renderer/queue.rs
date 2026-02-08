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

    // Phase 2: Swapchain Management
    pub old_swapchain_handles: Vec<vk::SwapchainKHR>,
    pub resize_pending: bool,
    pub pending_extent: Option<vk::Extent2D>,
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
            old_swapchain_handles: Vec::new(),
            resize_pending: false,
            pending_extent: None,
        })
    }

    /// Acquires the next frame for rendering.
    ///
    /// This method:
    /// 1. Blocks on the CPU until the GPU has finished using the command buffer for this frame slot.
    /// 2. Resets the in-flight fence.
    /// 3. Acquires the next available swapchain image index.
    ///
    /// Returns:
    /// - `u32`: The index of the swapchain image to render into.
    /// - `vk::CommandBuffer`: The primary command buffer for this frame slot.
    /// - `vk::Semaphore`: Semaphore signaled when image is available for rendering.
    /// - `vk::Semaphore`: Semaphore to be signaled when rendering is finished.
    /// - `vk::Fence`: Fence to be signaled when the command buffer execution completes.
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

        // 1. Safe access to synchronization objects
        let sync = self.frame_syncs.get(frame_index).ok_or_else(|| {
            crate::AshError::VulkanError(format!(
                "Frame index {} out of bounds for {} frame syncs",
                frame_index,
                self.frame_syncs.len()
            ))
        })?;

        // 2. Wait for and reset fence
        sync.wait()?;
        sync.reset()?;

        // 3. Acquire next swapchain image
        let image_index = unsafe { swapchain.acquire_next_image(sync.image_available)? };

        // 4. Safe access to command buffer
        let command_buffer = *self.command_buffers.get(frame_index).ok_or_else(|| {
            crate::AshError::VulkanError(format!(
                "Frame index {} out of bounds for {} command buffers",
                frame_index,
                self.command_buffers.len()
            ))
        })?;

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

    /// Requests a swapchain resize.
    pub fn request_resize(&mut self, extent: vk::Extent2D) {
        self.resize_pending = true;
        self.pending_extent = Some(extent);
    }

    pub fn is_resize_pending(&self) -> bool {
        self.resize_pending
    }

    /// Recreates the swapchain using the pending extent.
    pub fn recreate_swapchain(
        &mut self,
        swapchain: &mut Swapchain,
        device: &crate::vulkan::VulkanDevice,
    ) -> Result<()> {
        let extent = match self.pending_extent {
            Some(e) if e.width > 0 && e.height > 0 => e,
            _ => return Ok(()),
        };

        log::info!(
            "Recreating swapchain at RenderQueue level: {}x{}",
            extent.width,
            extent.height
        );

        // Safety: ensure all GPU work is done.
        unsafe {
            device.device.device_wait_idle().map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to wait for device idle: {e:?}"))
            })?;
        }

        // Recreate the swapchain. SwapchainWrapper::recreate returns the OLD handle.
        let old_handle = unsafe { swapchain.recreate(device)? };

        if old_handle != vk::SwapchainKHR::null() {
            self.old_swapchain_handles.push(old_handle);
        }

        self.resize_pending = false;
        self.pending_extent = None;

        Ok(())
    }

    /// Waits for all in-flight frames to finish.
    pub fn wait_for_inflight_frames(&self) -> Result<()> {
        for sync in &self.frame_syncs {
            sync.wait()?;
        }
        Ok(())
    }

    /// Defers the destruction of an old swapchain handle.
    pub fn defer_old_swapchain(&mut self, handle: vk::SwapchainKHR) {
        if handle == vk::SwapchainKHR::null() {
            return;
        }
        self.old_swapchain_handles.push(handle);
    }

    /// Destroys all deferred old swapchains.
    pub fn flush_old_swapchains(&mut self, device: &crate::vulkan::VulkanDevice) {
        if self.old_swapchain_handles.is_empty() {
            return;
        }

        let loader = ash::khr::swapchain::Device::new(device.instance.instance(), &device.device);

        for handle in self.old_swapchain_handles.drain(..) {
            unsafe {
                loader.destroy_swapchain(handle, None);
            }
        }
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
