use ash::vk;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use vk_mem::Alloc;

#[derive(Clone)]
struct BufferAllocation {
    _buffer: vk::Buffer,
    size: vk::DeviceSize,
    name: Option<String>,
    created_at: std::time::Instant,
}

pub struct Allocator {
    pub vma: vk_mem::Allocator,
    pub device: Arc<ash::Device>,
    pub debug_utils: Option<ash::ext::debug_utils::Device>,
    buffer_allocations: parking_lot::Mutex<HashMap<vk::Buffer, BufferAllocation>>,
}

impl Allocator {
    /// VMA allocator initialization.
    ///
    /// # Safety
    /// // SAFETY: Assumes a valid Vulkan instance/device. Device must outlive the allocator.
    pub unsafe fn new(device: &crate::vulkan::VulkanDevice) -> crate::Result<Self> {
        let mut create_info = vk_mem::AllocatorCreateInfo::new(
            device.instance.instance(),
            &device.device,
            device.physical_device,
        );
        create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let vma = unsafe { vk_mem::Allocator::new(create_info) }
            .map_err(|e| crate::AshError::VulkanError(format!("VMA init failed: {e:?}")))?;

        log::info!("VMA allocator created");

        Ok(Self {
            vma,
            device: Arc::clone(&device.device),
            debug_utils: device.debug_utils.clone(),
            buffer_allocations: parking_lot::Mutex::new(HashMap::new()),
        })
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
        unsafe {
            self.create_buffer_with_flags(
                size,
                usage,
                memory_usage,
                vk_mem::AllocationCreateFlags::empty(),
            )
        }
    }

    /// Allocate a GPU buffer with custom VMA allocation flags.
    ///
    /// # Arguments
    /// * `size` - Buffer size in bytes
    /// * `usage` - Vulkan buffer usage flags
    /// * `memory_usage` - VMA memory usage type
    /// * `flags` - VMA allocation flags (use HOST_ACCESS_SEQUENTIAL_WRITE for CPU writes)
    ///
    /// # Important
    /// If you plan to map this buffer (call `map_allocation_guarded`), you MUST
    /// include `AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE` in the flags.
    /// Otherwise, VMA will reject the mapping at runtime.
    ///
    /// # Safety
    /// // SAFETY: Standard VMA allocation. Params must be valid for the device.
    pub unsafe fn create_buffer_with_flags(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        unsafe { self.create_buffer_with_flags_and_name(size, usage, memory_usage, flags, None) }
    }

    /// Allocate a GPU buffer with custom VMA allocation flags and a debug name.
    ///
    /// # Safety
    /// // SAFETY: Standard VMA allocation. Params must be valid for the device.
    /// Internal validation helper to catch common Vulkan/VMA errors and performance pitfalls.
    fn validate_config(
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
    ) -> crate::Result<()> {
        // [Error] Hard Errors
        if size == 0 {
            return Err(crate::AshError::VulkanError(
                "Buffer size must be > 0".into(),
            ));
        }

        if memory_usage == vk_mem::MemoryUsage::AutoPreferDevice
            && flags.contains(vk_mem::AllocationCreateFlags::MAPPED)
        {
            return Err(crate::AshError::VulkanError(
                "Cannot map GPU-only buffer. Use CpuToGpu for CPU access.".into(),
            ));
        }

        if usage.is_empty() {
            return Err(crate::AshError::VulkanError(
                "Buffer must have at least one usage flag".into(),
            ));
        }

        // [Warning] Defensive Warnings
        if size > 4 * 1024 * 1024 * 1024 {
            log::warn!("Buffer size is very large ({size} bytes), may cause issues");
        }

        // Check for conflicting usage flags
        let index_related = vk::BufferUsageFlags::INDEX_BUFFER;
        let storage_related =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER;

        if usage.contains(index_related) && usage.contains(storage_related) {
            log::warn!("Buffer has conflicting usage flags: index + storage. Unusual combo.");
        }

        // Transfer-only buffers (warning)
        if (usage == vk::BufferUsageFlags::TRANSFER_DST
            || usage == vk::BufferUsageFlags::TRANSFER_SRC)
            && usage != (vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
        {
            log::warn!("Buffer is ONLY for one-way transfers. Verify this is intentional.");
        }

        // CPU-writable without transfer capability
        if flags.contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE)
            && !usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
            && !usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
        {
            log::warn!(
                "CPU-writable buffer missing TRANSFER_DST or STORAGE_BUFFER usage. \n             VMA might fail or performance will be poor (size={size}, usage={usage:?})."
            );
        }

        Ok(())
    }

    /// Allocate a GPU buffer with custom VMA allocation flags and a debug name.
    ///
    /// # Safety
    /// // SAFETY: Standard VMA allocation. Params must be valid for the device.
    pub unsafe fn create_buffer_with_flags_and_name(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        name: Option<String>,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        Self::validate_config(size, usage, memory_usage, flags)?;

        // Log for debugging
        if let Some(ref n) = name {
            log::debug!(
                "Creating buffer '{n}': size={size} bytes, usage={usage:?}, memory={memory_usage:?}"
            );
        } else {
            log::debug!(
                "Creating unnamed buffer: size={size} bytes, usage={usage:?}, memory={memory_usage:?}"
            );
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            flags,
            ..Default::default()
        };

        let (buffer, allocation) =
            unsafe { self.vma.create_buffer(&buffer_info, &allocation_info) }.map_err(|e| {
                crate::AshError::VulkanError(format!(
                    "Buffer creation failed (size={size}, usage={usage:?}, name={name:?}): {e:?}",
                    name = name.as_deref().unwrap_or("None")
                ))
            })?;

        self.buffer_allocations.lock().insert(
            buffer,
            BufferAllocation {
                _buffer: buffer,
                size,
                name: name.clone(),
                created_at: std::time::Instant::now(),
            },
        );

        if let Some(ref n) = name {
            #[cfg(debug_assertions)]
            crate::vulkan::set_debug_object_name(
                self.debug_utils.as_ref(),
                buffer,
                vk::ObjectType::BUFFER,
                n,
            );
            log::info!("Created buffer '{n}': {size} bytes");
        }

        Ok((buffer, allocation))
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
        unsafe {
            self.vma.create_image(
                image_info,
                &vk_mem::AllocationCreateInfo {
                    usage: memory_usage,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| crate::AshError::VulkanError(format!("Image creation failed: {e:?}")))
    }

    /// Create a Vulkan image and an associated image view in one step.
    ///
    /// # Safety
    /// // SAFETY: Standard Vulkan/VMA restrictions apply.
    pub unsafe fn create_image_with_view(
        &self,
        image_info: vk::ImageCreateInfo,
        allocation_info: vk_mem::AllocationCreateInfo,
        view_type: vk::ImageViewType,
        aspect_mask: vk::ImageAspectFlags,
    ) -> crate::Result<(vk::Image, vk::ImageView, vk_mem::Allocation)> {
        let (image, allocation) = unsafe { self.vma.create_image(&image_info, &allocation_info) }
            .map_err(|e| {
            crate::AshError::VulkanError(format!("Image creation failed: {e:?}"))
        })?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(image_info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: image_info.mip_levels,
                base_array_layer: 0,
                layer_count: image_info.array_layers,
            });

        let view = match unsafe { self.device.create_image_view(&view_info, None) } {
            Ok(view) => view,
            Err(e) => {
                let mut allocation = allocation;
                unsafe {
                    self.vma.destroy_image(image, &mut allocation);
                }
                return Err(crate::AshError::VulkanError(format!(
                    "Image view creation failed: {e:?}"
                )));
            }
        };

        Ok((image, view, allocation))
    }

    /// Deallocate buffer memory.
    ///
    /// # Safety
    /// // SAFETY: Handles must be valid and not currently in use by the GPU.
    pub unsafe fn destroy_buffer(&self, buffer: vk::Buffer, allocation: &mut vk_mem::Allocation) {
        // Track removal
        let (name, age) = {
            if let Some(alloc) = self.buffer_allocations.lock().remove(&buffer) {
                (
                    Some(alloc.name.clone()),
                    Some(alloc.created_at.elapsed().as_secs_f32()),
                )
            } else {
                (None, None)
            }
        };

        if let (Some(name), Some(age)) = (name, age) {
            let name_str = name.as_deref().unwrap_or("unnamed");
            log::debug!("Destroying buffer '{name_str}' (age: {age:.1}s)");
        } else {
            log::warn!("Destroying untracked buffer: {buffer:?}");
        }

        unsafe {
            self.vma.destroy_buffer(buffer, allocation);
        }
    }

    /// Print current buffer allocation statistics
    pub fn print_buffer_stats(&self) {
        let allocations_copy: Vec<_> = {
            let allocations = self.buffer_allocations.lock();
            allocations.values().cloned().collect()
        };

        log::info!("=== Buffer Allocation Statistics ===");
        let total: vk::DeviceSize = allocations_copy.iter().map(|b| b.size).sum();

        log::info!("Total buffers: {}", allocations_copy.len());
        log::info!("Total memory: {:.2} MB", total as f32 / 1_000_000.0);

        for alloc in allocations_copy {
            let age = alloc.created_at.elapsed().as_secs_f32();
            let name = alloc.name.as_deref().unwrap_or("unnamed");
            log::info!("  {} ({} bytes, {:.1}s old)", name, alloc.size, age);
        }
    }

    /// Map an allocation and return the raw pointer.
    ///
    /// # Safety
    /// // SAFETY: Must be matched with unmap_allocation.
    pub unsafe fn map_allocation(
        &self,
        allocation: &mut vk_mem::Allocation,
    ) -> crate::Result<*mut u8> {
        unsafe { self.vma.map_memory(allocation) }
            .map_err(|e| crate::AshError::VulkanError(format!("Map memory failed: {e:?}")))
    }

    /// Unmap an allocation.
    ///
    /// # Safety
    /// // SAFETY: Must be called only if mapped.
    pub unsafe fn unmap_allocation(&self, allocation: &mut vk_mem::Allocation) {
        unsafe {
            self.vma.unmap_memory(allocation);
        }
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
        let ptr = unsafe { self.map_allocation(allocation) }?;

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

    /// Access mapped memory as a slice of a specific type.
    ///
    /// # Safety
    /// // SAFETY: Type T must be compatible with the mapped data.
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let count = self.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, count) }
    }

    /// Access mapped memory as a mutable slice of a specific type.
    ///
    /// # Safety
    /// // SAFETY: Type T must be compatible with the mapped data.
    pub unsafe fn as_mut_slice_t<T: Copy>(&mut self) -> &mut [T] {
        let count = self.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, count) }
    }

    /// Copy data from a slice into the mapped memory.
    pub fn copy_from_slice<T: Copy>(&mut self, data: &[T]) {
        let size = std::mem::size_of_val(data);
        assert!(size <= self.size as usize, "Copy size exceeds mapped range");

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_validation_logic() {
        // Test size 0 (Hard Error)
        let res = Allocator::validate_config(
            0,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        );
        assert!(res.is_err(), "Size 0 should be rejected");

        // Test GPU-only + Mapped (Hard Error)
        let res = Allocator::validate_config(
            1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::MAPPED,
        );
        assert!(res.is_err(), "GPU-only + Mapped should be rejected");

        // Test Empty Usage (Hard Error)
        let res = Allocator::validate_config(
            1024,
            vk::BufferUsageFlags::empty(),
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        );
        assert!(res.is_err(), "Empty usage should be rejected");

        // Test valid params should pass
        let res = Allocator::validate_config(
            1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        );
        assert!(res.is_ok(), "Valid params should pass");
    }
}
