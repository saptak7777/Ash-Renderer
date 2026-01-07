use crate::vulkan::Allocator;
use ash::vk;
use std::sync::Arc;

/// Safe buffer wrapper with automatic cleanup via RAII
pub struct BufferHandle {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: Arc<Allocator>,
    size: u64,
    name: Option<String>,
}

impl BufferHandle {
    /// Creates a new GPU buffer.
    ///
    /// # Safety
    ///
    /// The allocator must outlive this buffer handle. The buffer is automatically
    /// destroyed when this handle is dropped.
    pub unsafe fn new(
        allocator: Arc<Allocator>,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        name: Option<String>,
    ) -> crate::Result<Self> {
        Self::new_with_flags(
            allocator,
            size,
            usage,
            memory_usage,
            vk_mem::AllocationCreateFlags::empty(),
            name,
        )
    }

    /// Creates a new GPU buffer with custom allocation flags.
    ///
    /// # Safety
    ///
    /// The allocator must outlive this buffer handle. The buffer is automatically
    /// destroyed when this handle is dropped.
    pub unsafe fn new_with_flags(
        allocator: Arc<Allocator>,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        name: Option<String>,
    ) -> crate::Result<Self> {
        if let Some(ref n) = name {
            log::info!("Creating buffer '{n}' ({size}B)");
        } else {
            log::info!("Creating buffer ({size}B)");
        }

        let (buffer, allocation) = allocator.create_buffer_with_flags(size, usage, memory_usage, flags)?;

        Ok(Self {
            buffer,
            allocation,
            allocator,
            size,
            name,
        })
    }

    /// Returns the Vulkan buffer handle
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Returns the buffer size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the buffer name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the allocation (internal use)
    pub fn allocation(&self) -> &vk_mem::Allocation {
        &self.allocation
    }

    /// Returns the allocation (internal use)
    pub fn allocation_mut(&mut self) -> &mut vk_mem::Allocation {
        // log::trace!("Accessing allocation_mut for buffer: {:?}", self.name);
        &mut self.allocation
    }
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref name) = self.name {
                log::debug!("Destroying buffer '{name}'");
            }
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

impl std::fmt::Debug for BufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferHandle")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("name", &self.name)
            .finish()
    }
}
