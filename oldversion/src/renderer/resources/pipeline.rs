use ash::vk;
use std::sync::Arc;

/// Safe pipeline wrapper with automatic cleanup
pub struct PipelineHandle {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    device: Arc<ash::Device>,
    name: Option<String>,
}

impl PipelineHandle {
    /// Creates a new pipeline handle.
    ///
    /// # Safety
    ///
    /// The device must remain valid for the lifetime of this handle.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        name: Option<String>,
    ) -> crate::Result<Self> {
        if let Some(ref n) = name {
            log::info!("Creating pipeline '{n}'");
        } else {
            log::info!("Creating pipeline");
        }

        Ok(Self {
            pipeline,
            layout,
            device,
            name,
        })
    }

    /// Returns the Vulkan pipeline handle
    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Returns the pipeline layout
    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// Returns the name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref name) = self.name {
                log::debug!("Destroying pipeline '{name}'");
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl std::fmt::Debug for PipelineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineHandle")
            .field("pipeline", &self.pipeline)
            .field("layout", &self.layout)
            .field("name", &self.name)
            .finish()
    }
}
