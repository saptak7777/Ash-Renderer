use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// RAII wrapper for Vulkan framebuffers.
pub struct Framebuffer {
    handle: vk::Framebuffer,
    device: Arc<ash::Device>,
    managed_by_registry: bool,
}

impl Framebuffer {
    pub fn new(
        device: Arc<ash::Device>,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let handle = unsafe {
            device
                .create_framebuffer(&info, None)
                .map_err(|e| AshError::VulkanError(format!("Failed to create framebuffer: {e}")))?
        };

        Ok(Self {
            handle,
            device,
            managed_by_registry: false,
        })
    }

    pub fn handle(&self) -> vk::Framebuffer {
        self.handle
    }

    pub fn mark_managed_by_registry(&mut self) {
        self.managed_by_registry = true;
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }

        unsafe {
            if self.handle != vk::Framebuffer::null() {
                self.device.destroy_framebuffer(self.handle, None);
            }
        }
    }
}
