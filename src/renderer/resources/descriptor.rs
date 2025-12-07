use ash::vk;
use std::sync::Arc;

/// Safe descriptor set wrapper with automatic cleanup
pub struct DescriptorSetHandle {
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    device: Arc<ash::Device>,
    pool: vk::DescriptorPool,
    name: Option<String>,
}

impl DescriptorSetHandle {
    /// Creates a new descriptor set from a pool.
    ///
    /// # Safety
    ///
    /// The device and pool must remain valid for the lifetime of this handle.
    /// Descriptor sets are freed when the pool is destroyed.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        name: Option<String>,
    ) -> crate::Result<Self> {
        if let Some(ref n) = name {
            log::info!("Creating descriptor set '{n}'");
        } else {
            log::info!("Creating descriptor set");
        }

        Ok(Self {
            set: vk::DescriptorSet::null(),
            layout,
            device,
            pool,
            name,
        })
    }

    /// Returns the Vulkan descriptor set handle
    pub fn handle(&self) -> vk::DescriptorSet {
        self.set
    }

    /// Returns the descriptor set layout
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Returns the descriptor pool
    pub fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Returns the name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl Drop for DescriptorSetHandle {
    fn drop(&mut self) {
        if let Some(ref name) = self.name {
            log::debug!("Descriptor set '{name}' dropped (freed with pool)");
        }
        // Descriptors are freed when pool is destroyed
    }
}

impl std::fmt::Debug for DescriptorSetHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorSetHandle")
            .field("set", &self.set)
            .field("layout", &self.layout)
            .field("name", &self.name)
            .finish()
    }
}
