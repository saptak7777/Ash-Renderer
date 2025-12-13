use std::path::PathBuf;
use std::sync::Arc;

use ash::{vk, Device};

use crate::{AshError, Result};

/// Pipeline cache with optional disk persistence.
///
/// Saves compiled shader bytecode to disk for faster startup times.
pub struct PipelineCache {
    device: Arc<Device>,
    cache: vk::PipelineCache,
    cache_file: Option<PathBuf>,
}

impl PipelineCache {
    /// Creates a new pipeline cache without persistence.
    pub fn new(device: Arc<Device>) -> Result<Self> {
        Self::with_persistence(device, None)
    }

    /// Creates a pipeline cache with optional disk persistence.
    ///
    /// If `cache_file` is provided, the cache will be loaded from disk on creation
    /// and saved to disk on drop.
    pub fn with_persistence(device: Arc<Device>, cache_file: Option<PathBuf>) -> Result<Self> {
        // Try to load existing cache data from disk
        let initial_data = cache_file
            .as_ref()
            .and_then(|path| {
                if path.exists() {
                    match std::fs::read(path) {
                        Ok(data) => {
                            log::info!("Loaded pipeline cache from: {}", path.display());
                            Some(data)
                        }
                        Err(e) => {
                            log::warn!("Failed to load pipeline cache: {e}");
                            None
                        }
                    }
                } else {
                    log::debug!("No existing pipeline cache at: {}", path.display());
                    None
                }
            })
            .unwrap_or_default();

        let create_info = if initial_data.is_empty() {
            vk::PipelineCacheCreateInfo::default()
        } else {
            vk::PipelineCacheCreateInfo::default().initial_data(&initial_data)
        };

        let cache = unsafe {
            device
                .create_pipeline_cache(&create_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create pipeline cache: {e}"))
                })?
        };

        Ok(Self {
            device,
            cache,
            cache_file,
        })
    }

    pub fn from_handle(device: Arc<Device>, cache: vk::PipelineCache) -> Self {
        Self {
            device,
            cache,
            cache_file: None,
        }
    }

    pub fn handle(&self) -> vk::PipelineCache {
        self.cache
    }

    pub fn merge(&self, caches: &[vk::PipelineCache]) -> Result<()> {
        unsafe {
            self.device
                .merge_pipeline_caches(self.cache, caches)
                .map_err(|e| AshError::VulkanError(format!("Failed to merge pipeline caches: {e}")))
        }
    }

    pub fn get_data(&self) -> Result<Vec<u8>> {
        unsafe {
            self.device
                .get_pipeline_cache_data(self.cache)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to read pipeline cache data: {e}"))
                })
        }
    }

    /// Saves the cache to disk (if persistence is enabled).
    pub fn save(&self) -> Result<()> {
        if let Some(ref path) = self.cache_file {
            let data = self.get_data()?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    AshError::VulkanError(format!("Failed to create cache directory: {e}"))
                })?;
            }
            std::fs::write(path, &data).map_err(|e| {
                AshError::VulkanError(format!("Failed to write pipeline cache: {e}"))
            })?;
            log::info!(
                "Saved pipeline cache ({} bytes) to: {}",
                data.len(),
                path.display()
            );
        }
        Ok(())
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        // Save cache to disk before destroying
        if self.cache_file.is_some() {
            if let Err(e) = self.save() {
                log::warn!("Failed to save pipeline cache on drop: {e}");
            }
        }
        unsafe {
            self.device.destroy_pipeline_cache(self.cache, None);
        }
    }
}
