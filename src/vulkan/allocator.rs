use ash::vk;
use vk_mem::Alloc;

pub struct Allocator {
    pub vma: vk_mem::Allocator,
}

impl Allocator {
    /// VMA allocator initialization.
    ///
    /// # Safety
    /// // SAFETY: Assumes a valid Vulkan instance/device. Device must outlive the allocator.
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

    /// Allocate a GPU buffer.
    ///
    /// # Safety
    /// // SAFETY: Standard VMA allocation. Params must be valid for the device.
    pub unsafe fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        debug_assert!(size > 0, "Buffer size must be non-zero"); // Catch simple logic errors in dev

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

    /// Create a Vulkan image.
    ///
    /// # Safety
    /// // SAFETY: Image handle must be destroyed before the allocator.
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

    /// Deallocate buffer memory.
    ///
    /// # Safety
    /// // SAFETY: Handles must be valid and not currently in use by the GPU.
    pub unsafe fn destroy_buffer(&self, buffer: vk::Buffer, allocation: &mut vk_mem::Allocation) {
        self.vma.destroy_buffer(buffer, allocation);
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        log::info!("VMA allocator destroyed");
    }
}
