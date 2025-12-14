use ash::vk;
use vk_mem::Alloc;

pub struct Allocator {
    pub vma: vk_mem::Allocator,
}

impl Allocator {
    /// Creates a new VMA allocator for GPU memory management.
    ///
    /// # Safety
    ///
    /// This function creates a Vulkan memory allocator. Caller must ensure:
    /// - `device` references a valid and initialized Vulkan device
    /// - The device is not being destroyed while this allocator is in use
    /// - All buffers allocated from this allocator are destroyed before dropping the allocator
    pub unsafe fn new(device: &crate::vulkan::VulkanDevice) -> crate::Result<Self> {
        let vma = vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
            device.instance.instance(),
            &device.device,
            device.physical_device,
        ))
        .map_err(|e| crate::AshError::VulkanError(format!("VMA init failed: {e:?}")))?;

        log::info!("VMA allocator created");

        Ok(Self { vma })
    }

    /// Allocates a GPU buffer with the specified parameters.
    ///
    /// # Safety
    ///
    /// This function allocates GPU memory. Caller must ensure:
    /// - Parameters are valid (size > 0, valid usage flags)
    /// - The returned allocation is properly destroyed with `destroy_buffer`
    /// - The allocator outlives all buffers created from it
    /// - Concurrent allocations use proper synchronization
    pub unsafe fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        let flags = if memory_usage == vk_mem::MemoryUsage::AutoPreferHost {
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        } else {
            vk_mem::AllocationCreateFlags::empty()
        };

        self.vma
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: memory_usage,
                    flags,
                    ..Default::default()
                },
            )
            .map_err(|e| crate::AshError::VulkanError(format!("Buffer creation failed: {e:?}")))
    }

    /// Creates an image with the specified parameters.
    ///
    /// # Safety
    /// Caller must ensure the returned image is destroyed with `vk_mem::Allocator::destroy_image`
    /// before dropping the allocator and that the image is no longer in use by the GPU when
    /// destroyed.
    pub unsafe fn create_image(
        &self,
        image_info: &vk::ImageCreateInfo,
        memory_usage: vk_mem::MemoryUsage,
    ) -> crate::Result<(vk::Image, vk_mem::Allocation)> {
        self.vma
            .create_image(
                image_info,
                &vk_mem::AllocationCreateInfo {
                    usage: memory_usage,
                    ..Default::default()
                },
            )
            .map_err(|e| crate::AshError::VulkanError(format!("Image creation failed: {e:?}")))
    }

    /// Destroys a previously allocated buffer.
    ///
    /// # Safety
    ///
    /// This function deallocates GPU memory. Caller must ensure:
    /// - `buffer` was allocated from this allocator
    /// - `buffer` is no longer in use by the GPU
    /// - `allocation` corresponds to the buffer being destroyed
    /// - The buffer is not accessed after destruction
    pub unsafe fn destroy_buffer(&self, buffer: vk::Buffer, allocation: &mut vk_mem::Allocation) {
        self.vma.destroy_buffer(buffer, allocation);
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        log::info!("VMA allocator destroyed");
    }
}
