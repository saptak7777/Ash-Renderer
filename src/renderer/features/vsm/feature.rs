//! VSM Feature - High-level integration of Virtual Shadow Maps

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::compute_pipelines::VsmComputePipelines;
use super::page_manager::{PageManager, PageManagerStats};
use super::resources::{VsmConfig, VsmResources};
use super::shadow_pass::VsmShadowPass;

/// VSM Feature - Complete virtual shadow map system
pub struct VsmFeature {
    /// GPU resources
    resources: VsmResources,

    /// CPU-side page manager
    page_manager: PageManager,

    /// Compute pipelines
    compute_pipelines: VsmComputePipelines,

    /// Shadow rendering pass
    shadow_pass: VsmShadowPass,

    /// Descriptor pool for VSM
    descriptor_pool: vk::DescriptorPool,

    /// Descriptor sets (per frame)
    analysis_descriptor_sets: Vec<vk::DescriptorSet>,
    allocator_descriptor_sets: Vec<vk::DescriptorSet>,

    /// Current frame index
    current_frame: u32,

    /// VSM device handle
    device: Arc<ash::Device>,

    /// Enabled state
    enabled: bool,
}

impl VsmFeature {
    /// Create a new VSM feature
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this feature.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        config: VsmConfig,
        frame_count: u32,
    ) -> Result<Self> {
        log::info!("Creating VSM feature with {} frames", frame_count);

        // Create resources
        let resources =
            VsmResources::new(Arc::clone(&device), Arc::clone(&allocator), config.clone())?;

        // Create page manager
        let page_manager = PageManager::new(
            config.virtual_resolution,
            config.physical_resolution,
            config.page_size,
        );

        // Create compute pipelines
        let compute_pipelines =
            VsmComputePipelines::new(Arc::clone(&device), Arc::clone(&allocator))?;

        // Create shadow pass
        let shadow_pass =
            VsmShadowPass::new(Arc::clone(&device), Arc::clone(&allocator), &resources)?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: frame_count * 2, // Depth buffer + page table
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: frame_count * 4, // Metadata, camera, light x2
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: frame_count * 6, // Request, allocation, free pool x2
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: frame_count * 2, // Page table x2
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(frame_count * 2); // Analysis + Allocator per frame

        let descriptor_pool = device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create VSM descriptor pool: {e:?}"))
            })?;

        // Allocate descriptor sets (will be populated later)
        let mut analysis_descriptor_sets = Vec::new();
        let mut allocator_descriptor_sets = Vec::new();

        for _ in 0..frame_count {
            let analysis_layouts = [compute_pipelines.analysis_descriptor_set_layout];
            let analysis_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&analysis_layouts);

            let analysis_sets = device
                .allocate_descriptor_sets(&analysis_alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to allocate analysis descriptor set: {e:?}"
                    ))
                })?;
            analysis_descriptor_sets.push(analysis_sets[0]);

            let allocator_layouts = [compute_pipelines.allocator_descriptor_set_layout];
            let allocator_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&allocator_layouts);

            let allocator_sets = device
                .allocate_descriptor_sets(&allocator_alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to allocate allocator descriptor set: {e:?}"
                    ))
                })?;
            allocator_descriptor_sets.push(allocator_sets[0]);
        }

        log::info!("VSM feature created successfully");

        Ok(Self {
            resources,
            page_manager,
            compute_pipelines,
            shadow_pass,
            descriptor_pool,
            analysis_descriptor_sets,
            allocator_descriptor_sets,
            current_frame: 0,
            device,
            enabled: true,
        })
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self, frame_index: u32) {
        self.current_frame = frame_index;
        self.page_manager.begin_frame(frame_index);

        // Update metadata buffer
        if let Err(e) = self.resources.update_metadata(frame_index) {
            log::error!("Failed to update VSM metadata: {e:?}");
        }
    }

    /// Get page manager statistics
    pub fn stats(&self) -> PageManagerStats {
        self.page_manager.stats()
    }

    /// Check if VSM is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable VSM
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get physical cache image view for binding
    pub fn physical_cache_view(&self) -> vk::ImageView {
        self.resources.physical_cache_view
    }

    /// Get physical cache sampler
    pub fn physical_cache_sampler(&self) -> vk::Sampler {
        self.resources.physical_cache_sampler
    }

    /// Get page table image view
    pub fn page_table_view(&self) -> vk::ImageView {
        self.resources.page_table_view
    }

    /// Get page table sampler
    pub fn page_table_sampler(&self) -> vk::Sampler {
        self.resources.page_table_sampler
    }

    /// Get shadow render pass
    pub fn shadow_render_pass(&self) -> vk::RenderPass {
        self.shadow_pass.render_pass
    }

    /// Get shadow framebuffer
    pub fn shadow_framebuffer(&self) -> vk::Framebuffer {
        self.shadow_pass.framebuffer
    }

    /// Get configuration
    pub fn config(&self) -> &VsmConfig {
        self.resources.config()
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying VSM feature");

        self.shadow_pass.destroy();
        self.compute_pipelines.destroy();

        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }

        self.resources.destroy();

        log::debug!("VSM feature destroyed");
    }
}

impl Drop for VsmFeature {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}

/// Helper to create a default VSM configuration
pub fn default_vsm_config() -> VsmConfig {
    VsmConfig::default()
}

/// Helper to create a high-quality VSM configuration
pub fn high_quality_vsm_config() -> VsmConfig {
    VsmConfig {
        virtual_resolution: 16384,
        physical_resolution: 8192, // Larger cache
        page_size: 128,
        max_requests_per_frame: 2048,
        debug_mode: false,
    }
}

/// Helper to create a performance-focused VSM configuration
pub fn performance_vsm_config() -> VsmConfig {
    VsmConfig {
        virtual_resolution: 8192,
        physical_resolution: 2048, // Smaller cache
        page_size: 128,
        max_requests_per_frame: 512,
        debug_mode: false,
    }
}
