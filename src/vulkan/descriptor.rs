// src/vulkan/descriptor_pool.rs - Phase 2 Descriptor Set Management (NEW)

use ash::vk;
use std::sync::Arc;

/// Descriptor pool for managing descriptor sets
pub struct DescriptorPool {
    pool: vk::DescriptorPool,
    device: Arc<ash::Device>,
}

impl DescriptorPool {
    /// Creates a new descriptor pool
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this pool
    pub unsafe fn new(
        device: Arc<ash::Device>,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> crate::Result<Self> {
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(pool_sizes)
            .max_sets(max_sets)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to create descriptor pool: {e:?}"))
            })?;

        log::info!("Created descriptor pool with max {max_sets} sets");

        Ok(Self { pool, device })
    }

    /// Create a default descriptor pool for Phase 2 (uniforms + textures)
    ///
    /// # Safety
    ///
    /// The provided `device` must remain valid for the lifetime of the returned pool.
    pub unsafe fn create_default(device: Arc<ash::Device>) -> crate::Result<Self> {
        let pool_sizes = [
            // Uniform buffers (for MVP matrices, materials, etc.)
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 100,
            },
            // Combined image samplers (for textures)
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 100,
            },
        ];

        Self::new(device, 100, &pool_sizes)
    }

    /// Allocate descriptor sets from this pool
    ///
    /// # Safety
    ///
    /// The `layouts` must have been created from the same logical device as this pool and the
    /// caller must ensure the returned sets are not used after the pool is reset or destroyed.
    pub unsafe fn allocate_sets(
        &self,
        layouts: &[vk::DescriptorSetLayout],
    ) -> crate::Result<Vec<vk::DescriptorSet>> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        self.device
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to allocate descriptor sets: {e:?}"))
            })
    }

    /// Get the raw Vulkan handle
    pub fn handle(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Reset the pool (frees all allocated sets)
    ///
    /// # Safety
    ///
    /// All descriptor sets allocated from this pool must no longer be in use on the GPU when this
    /// function is called.
    pub unsafe fn reset(&self) -> crate::Result<()> {
        self.device
            .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to reset descriptor pool: {e:?}"))
            })
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
            log::debug!("Descriptor pool destroyed");
        }
    }
}

/// Descriptor set layout builder
pub struct DescriptorSetLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorSetLayoutBuilder {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// Add a uniform buffer binding
    pub fn add_uniform_buffer(mut self, binding: u32, stage: vk::ShaderStageFlags) -> Self {
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: stage,
            p_immutable_samplers: std::ptr::null(),
        });
        self
    }

    /// Add a combined image sampler binding (texture)
    pub fn add_sampler(mut self, binding: u32, stage: vk::ShaderStageFlags) -> Self {
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: stage,
            p_immutable_samplers: std::ptr::null(),
        });
        self
    }

    /// Build the descriptor set layout
    ///
    /// # Safety
    ///
    /// The caller must ensure the returned layout is destroyed before the associated device.
    pub unsafe fn build(self, device: Arc<ash::Device>) -> crate::Result<DescriptorSetLayout> {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);

        let layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| {
                crate::AshError::VulkanError(format!(
                    "Failed to create descriptor set layout: {e:?}"
                ))
            })?;
        let binding_count = self.bindings.len();
        log::info!("Created descriptor set layout with {binding_count} bindings");

        Ok(DescriptorSetLayout { layout, device })
    }
}

impl Default for DescriptorSetLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII wrapper for descriptor set layout
pub struct DescriptorSetLayout {
    layout: vk::DescriptorSetLayout,
    device: Arc<ash::Device>,
}

impl DescriptorSetLayout {
    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
            log::debug!("Descriptor set layout destroyed");
        }
    }
}
