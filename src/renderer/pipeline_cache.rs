use std::sync::Arc;

use ash::{Device, vk};

use crate::{AshError, Result};

pub struct PipelineCache {
    device: Arc<Device>,
    cache: vk::PipelineCache,
}

impl PipelineCache {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let create_info = vk::PipelineCacheCreateInfo::default();

        let cache = unsafe {
            device
                .create_pipeline_cache(&create_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create pipeline cache: {e}"))
                })?
        };

        Ok(Self { device, cache })
    }

    pub fn from_handle(device: Arc<Device>, cache: vk::PipelineCache) -> Self {
        Self { device, cache }
    }

    pub fn handle(&self) -> vk::PipelineCache {
        self.cache
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
