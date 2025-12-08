use std::sync::Arc;

use ash::{vk, Device};

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

    #[allow(dead_code)]
    pub fn merge(&self, caches: &[vk::PipelineCache]) -> Result<()> {
        unsafe {
            self.device
                .merge_pipeline_caches(self.cache, caches)
                .map_err(|e| AshError::VulkanError(format!("Failed to merge pipeline caches: {e}")))
        }
    }

    #[allow(dead_code)]
    pub fn get_data(&self) -> Result<Vec<u8>> {
        unsafe {
            self.device
                .get_pipeline_cache_data(self.cache)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to read pipeline cache data: {e}"))
                })
        }
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
