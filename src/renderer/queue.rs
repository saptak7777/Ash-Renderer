use ash::vk;
use std::sync::Arc;

use crate::vulkan::SwapchainWrapper as Swapchain;
use crate::Result;

/// Specialized queue wrapper for rendering setup and submission.
/// Also manages per-frame synchronization and command buffers.
pub struct RenderQueue {
    pub device: Arc<ash::Device>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub queue_family_index: u32,

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
    ) -> Result<Self> {
        Ok(Self {
            device,
            graphics_queue,
            present_queue,
            queue_family_index,
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
}
