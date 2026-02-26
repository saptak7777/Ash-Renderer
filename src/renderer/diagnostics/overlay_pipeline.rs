//! Overlay pipeline for rendering diagnostics text

use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Overlay rendering pipeline
pub struct OverlayPipeline {
    device: Arc<ash::Device>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// BDA Vertex heap for overlay (recreated each frame)
    vertex_heap: Option<crate::renderer::resources::BufferHandle>,
}

impl OverlayPipeline {
    /// Create overlay pipeline
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this pipeline.
    pub unsafe fn new(device: Arc<ash::Device>, _swapchain_format: vk::Format) -> Result<Self> {
        log::info!("[OverlayPipeline] Creating overlay pipeline");

        // Create pipeline layout (no descriptors, no push constants)
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| AshError::VulkanError(format!("Overlay layout failed: {e}")))?
        };

        log::info!("[OverlayPipeline] Overlay pipeline created");

        Ok(Self {
            device,
            pipeline_layout,
            pipeline: vk::Pipeline::null(),
            vertex_heap: None,
        })
    }

    /// Check if pipeline needs to be created
    pub fn needs_pipeline(&self) -> bool {
        self.pipeline == vk::Pipeline::null()
    }

    /// Get pipeline handle
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Get pipeline layout
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for OverlayPipeline {
    fn drop(&mut self) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.vertex_heap = None;

            log::info!("[OverlayPipeline] Overlay pipeline destroyed");
        }
    }
}
