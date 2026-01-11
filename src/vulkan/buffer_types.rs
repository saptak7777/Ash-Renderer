use ash::vk;

/// Describes what a buffer will be used for
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: vk_mem::MemoryUsage,
    pub mappable: bool,
    pub persistent_mapping: bool,
    pub name: Option<String>,
}

impl BufferDescriptor {
    pub fn new(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::empty(),
            memory_usage: vk_mem::MemoryUsage::AutoPreferDevice,
            mappable: false,
            persistent_mapping: false,
            name: None,
        }
    }

    /// Validate descriptor before allocation
    pub fn validate(&self) -> crate::Result<()> {
        // Check: size is reasonable
        if self.size == 0 {
            return Err(crate::AshError::VulkanError(
                "Buffer size must be > 0".into(),
            ));
        }

        // Check: GPU-only buffer can't be mapped
        if self.mappable && self.memory_usage == vk_mem::MemoryUsage::AutoPreferDevice {
            return Err(crate::AshError::VulkanError(
                "GPU-only buffers cannot be mapped. Use MemoryUsage::AutoPreferHost or CpuToGpu for mappable buffers."
                    .into(),
            ));
        }

        // Check: Persistent mapping requires mappable
        if self.persistent_mapping && !self.mappable {
            return Err(crate::AshError::VulkanError(
                "Persistent mapping requires the buffer to be mappable. Call .cpu_writable() first.".into(),
            ));
        }

        Ok(())
    }

    /// Convert to VMA allocation flags
    pub fn to_vma_flags(&self) -> vk_mem::AllocationCreateFlags {
        let mut flags = vk_mem::AllocationCreateFlags::empty();

        if self.mappable {
            // For general mappable buffers, we use RANDOM to allow both reading and writing.
            // SEQUENTIAL_WRITE is an optimization for CPU -> GPU only.
            flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM;
        }

        if self.persistent_mapping {
            flags |= vk_mem::AllocationCreateFlags::MAPPED;
        }

        flags
    }
}
