use crate::vulkan::allocator::Allocator;
use crate::vulkan::buffer_types::BufferDescriptor;
use ash::vk;

/// Type-safe builder for GPU buffers
pub struct BufferBuilder {
    descriptor: BufferDescriptor,
}

impl BufferBuilder {
    /// Create new buffer builder
    pub fn new(size: vk::DeviceSize) -> Self {
        Self {
            descriptor: BufferDescriptor::new(size),
        }
    }

    /// Buffer will store index data
    pub fn index_buffer(mut self) -> Self {
        self.descriptor.usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        self
    }

    /// Buffer will store uniform data (constants, matrices)
    pub fn uniform_buffer(mut self) -> Self {
        self.descriptor.usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        self
    }

    /// Buffer will store structured shader storage data
    pub fn storage_buffer(mut self) -> Self {
        self.descriptor.usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        self
    }

    /// Buffer will be target of indirect draw commands
    pub fn indirect_buffer(mut self) -> Self {
        self.descriptor.usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        self
    }

    /// Buffer receives data from transfers (COPY_DST)
    pub fn transfer_dst(mut self) -> Self {
        self.descriptor.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        self
    }

    /// Buffer is source for transfers (COPY_SRC)
    pub fn transfer_src(mut self) -> Self {
        self.descriptor.usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        self
    }

    /// Buffer needs CPU write access
    pub fn cpu_writable(mut self) -> Self {
        self.descriptor.mappable = true;
        self.descriptor.memory_usage = vk_mem::MemoryUsage::AutoPreferHost;
        self
    }

    /// Buffer needs CPU read access
    pub fn cpu_readable(mut self) -> Self {
        self.descriptor.mappable = true;
        self.descriptor.memory_usage = vk_mem::MemoryUsage::AutoPreferHost;
        self
    }

    /// Keep buffer persistently mapped (faster updates)
    pub fn persistent_mapping(mut self) -> Self {
        self.descriptor.persistent_mapping = true;
        self.descriptor.mappable = true;
        self
    }

    /// GPU-only, no CPU access
    pub fn gpu_only(mut self) -> Self {
        self.descriptor.memory_usage = vk_mem::MemoryUsage::AutoPreferDevice;
        self
    }

    /// Give buffer a debug name
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.descriptor.name = Some(name.into());
        self
    }

    /// Build the buffer
    pub fn build(self, allocator: &Allocator) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        // Validate first
        self.descriptor.validate()?;

        // Allocate using internal helper
        unsafe {
            allocator.create_buffer_with_flags_and_name(
                self.descriptor.size,
                self.descriptor.usage,
                self.descriptor.memory_usage,
                self.descriptor.to_vma_flags(),
                self.descriptor.name,
            )
        }
    }

    /// Return a reference to the internal descriptor (for tests or inspection)
    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}
