use ash::vk;
use std::ops::{Deref, DerefMut};
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
        debug_assert!(size > 0, "Buffer size must be non-zero");
        // Defensive: catch accidental massive allocations (e.g. 2GB sanity limit)
        debug_assert!(
            size < 2 * 1024 * 1024 * 1024,
            "Buffer size exceeds 2GB sanity limit"
        );

        let flags = vk_mem::AllocationCreateFlags::empty();

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
            .map_err(|e| {
                crate::AshError::VulkanError(format!(
                    "Buffer creation failed (size={size}, usage={usage:?}): {e:?}"
                ))
            })
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

    /// Map an allocation and return an RAII guard.
    ///
    /// # Safety
    /// // SAFETY: Allocation must be host-visible. Unmap is handled by the guard.
    pub unsafe fn map_allocation_guarded<'a>(
        &'a self,
        allocation: &'a mut vk_mem::Allocation,
        size: u64,
    ) -> crate::Result<MapGuard<'a>> {
        let ptr = self
            .vma
            .map_memory(allocation)
            .map_err(|e| crate::AshError::VulkanError(format!("Map memory failed: {e:?}")))?;

        Ok(MapGuard {
            vma: &self.vma,
            allocation,
            ptr,
            size,
        })
    }
}

/// RAII guard for mapped GPU memory.
///
/// Ensures memory is unmapped when dropped, providing panic safety.
pub struct MapGuard<'a> {
    vma: &'a vk_mem::Allocator,
    allocation: &'a mut vk_mem::Allocation,
    ptr: *mut u8,
    size: u64,
}

impl<'a> MapGuard<'a> {
    /// Access mapped memory as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    /// Copy data from a slice into the mapped memory.
    pub fn copy_from_slice<T: Copy>(&mut self, data: &[T]) {
        let size = std::mem::size_of_val(data);
        debug_assert!(size <= self.size as usize, "Copy size exceeds mapped range");

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, self.ptr, size);
        }
    }
}

impl<'a> Deref for MapGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }
}

impl<'a> DerefMut for MapGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }
}

impl<'a> Drop for MapGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            self.vma.unmap_memory(self.allocation);
        }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        log::info!("VMA allocator destroyed");
    }
}
