use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Synchronization primitives required for a single in-flight frame.
pub struct FrameSync {
    device: Arc<ash::Device>,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
    managed_by_registry: bool,
}

impl FrameSync {
    pub fn new(device: Arc<ash::Device>) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        let image_available = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to create image-available semaphore: {e}"
                    ))
                })?
        };

        let render_finished = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to create render-finished semaphore: {e}"
                    ))
                })?
        };

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let in_flight = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(|e| AshError::VulkanError(format!("Failed to create frame fence: {e}")))?
        };

        Ok(Self {
            device,
            image_available,
            render_finished,
            in_flight,
            managed_by_registry: false,
        })
    }

    pub fn wait(&self) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight], true, u64::MAX)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to wait for in-flight fence: {e}"))
                })?
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.device.reset_fences(&[self.in_flight]).map_err(|e| {
                AshError::VulkanError(format!("Failed to reset in-flight fence: {e}"))
            })?
        }
        Ok(())
    }

    pub fn mark_managed_by_registry(&mut self) {
        self.managed_by_registry = true;
    }
}

impl Drop for FrameSync {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }

        unsafe {
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_semaphore(self.render_finished, None);
            self.device.destroy_fence(self.in_flight, None);
        }
    }
}
