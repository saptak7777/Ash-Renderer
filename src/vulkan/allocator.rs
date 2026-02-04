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
    pub instance: Arc<crate::vulkan::VulkanInstance>,
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

        let vma = vk_mem::Allocator::new(create_info)
            .map_err(|e| crate::AshError::VulkanError(format!("VMA init failed: {e:?}")))?;

        log::info!("VMA allocator created");

        Ok(Self {
            vma,
            device: Arc::clone(&device.device),
            instance: Arc::clone(&device.instance),
            debug_utils: device.debug_utils.clone(),
            buffer_allocations: parking_lot::Mutex::new(HashMap::new()),
        })
    }

    /// Validate buffer creation parameters
    fn validate_buffer_params(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
    ) -> crate::Result<()> {
        validate_buffer_params_impl(size, usage, memory_usage, flags)
    }
}

/// Standalone implementation of buffer validation logic (for testing without Allocator instance)
fn validate_buffer_params_impl(
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_usage: vk_mem::MemoryUsage,
    flags: vk_mem::AllocationCreateFlags,
) -> crate::Result<()> {
    // Check 1: Size > 0
    if size == 0 {
        return Err(crate::AshError::VulkanError(
            "Buffer size must be > 0".into(),
        ));
    }

    if size > 4 * 1024 * 1024 * 1024 {
        log::warn!("Buffer size is very large ({size} bytes), may cause issues");
    }

    // Check 2: GPU-only buffers can't be mapped
    if memory_usage == vk_mem::MemoryUsage::AutoPreferDevice {
        if flags.contains(vk_mem::AllocationCreateFlags::MAPPED) {
            return Err(crate::AshError::VulkanError(
                "Cannot map GPU-only buffer. Use CpuToGpu for CPU access.".into(),
            ));
        }

        if flags.contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE) {
            return Err(crate::AshError::VulkanError(
                "GPU-only buffer cannot be CPU-writable. Use CpuToGpu.".into(),
            ));
        }
    }

    // Check 3: Conflicting usage flags
    let index_related = vk::BufferUsageFlags::INDEX_BUFFER;
    let storage_related =
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER;

    if usage.contains(index_related) && usage.contains(storage_related) {
        log::warn!("Buffer has conflicting usage flags: index + storage. Unusual combo.");
    }

    // Check 4: Transfer-only buffers (warning)
    if (usage == vk::BufferUsageFlags::TRANSFER_DST || usage == vk::BufferUsageFlags::TRANSFER_SRC)
        && usage != (vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
    {
        log::warn!("Buffer is ONLY for one-way transfers. Verify this is intentional.");
    }

    // Check 5: CPU-writable without transfer capability
    if flags.contains(vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE)
        && !usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
        && !usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
    {
        log::warn!(
            "CPU-writable buffer missing TRANSFER_DST or STORAGE_BUFFER usage. \
             VMA might fail or performance will be poor (size={size}, usage={usage:?})."
        );
    }

    Ok(())
}

impl Allocator {
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
        self.create_buffer_with_flags(
            size,
            usage,
            memory_usage,
            vk_mem::AllocationCreateFlags::empty(),
        )
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
        self.create_buffer_with_flags_and_name(size, usage, memory_usage, flags, None)
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
        // ✅ Validate first (catches errors early)
        self.validate_buffer_params(size, usage, memory_usage, flags)?;

        // ✅ Debug assertions for sanity checks
        debug_assert!(size > 0, "Buffer size must be > 0");

        debug_assert!(
            !(memory_usage == vk_mem::MemoryUsage::AutoPreferDevice
                && flags.contains(vk_mem::AllocationCreateFlags::MAPPED)),
            "Cannot map GPU-only buffer"
        );

        debug_assert!(
            !usage.is_empty(),
            "Buffer must have at least one usage flag"
        );

        // Log for debugging
        if let Some(ref n) = name {
            log::debug!("Creating buffer '{n}': size={size} bytes, usage={usage:?}, memory={memory_usage:?}");
        } else {
            log::debug!("Creating unnamed buffer: size={size} bytes, usage={usage:?}, memory={memory_usage:?}");
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

        let (buffer, allocation) = self
            .vma
            .create_buffer(&buffer_info, &allocation_info)
            .map_err(|e| {
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
        let (image, allocation) = self
            .vma
            .create_image(&image_info, &allocation_info)
            .map_err(|e| crate::AshError::VulkanError(format!("Image creation failed: {e:?}")))?;

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

        let view = match self.device.create_image_view(&view_info, None) {
            Ok(view) => view,
            Err(e) => {
                let mut allocation = allocation;
                self.vma.destroy_image(image, &mut allocation);
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
        if let Some(alloc) = self.buffer_allocations.lock().remove(&buffer) {
            let name = alloc.name.as_deref().unwrap_or("unnamed");
            let age = alloc.created_at.elapsed().as_secs_f32();
            log::debug!("Destroying buffer '{name}' (age: {age:.1}s)");
        } else {
            log::warn!("Destroying untracked buffer: {buffer:?}");
        }

        self.vma.destroy_buffer(buffer, allocation);
    }

    /// Print current buffer allocation statistics
    pub fn print_buffer_stats(&self) {
        let allocations = self.buffer_allocations.lock();
        log::info!("=== Buffer Allocation Statistics ===");
        let total: vk::DeviceSize = allocations.values().map(|b| b.size).sum();

        log::info!("Total buffers: {}", allocations.len());
        log::info!("Total memory: {:.2} MB", total as f32 / 1_000_000.0);

        for alloc in allocations.values() {
            let age = alloc.created_at.elapsed().as_secs_f32();
            let name = alloc.name.as_deref().unwrap_or("unnamed");
            log::info!("  {} ({} bytes, {:.1}s old)", name, alloc.size, age);
        }
    }

    // ============================================================================
    // COMMON PRESET BUILDERS (Convenience Methods)
    // ============================================================================

    /// Create uniform buffer (read-only, constant data)
    pub fn create_uniform_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        crate::vulkan::buffer_builder::BufferBuilder::new(size)
            .uniform_buffer()
            .gpu_only()
            .build(self)
    }

    /// Create staging buffer (CPU → GPU transfer)
    pub fn create_staging_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        crate::vulkan::buffer_builder::BufferBuilder::new(size)
            .transfer_src()
            .cpu_writable()
            .named("Staging Buffer")
            .build(self)
    }

    /// Create indirect draw buffer (GPU-written, GPU-consumed)
    pub fn create_indirect_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        crate::vulkan::buffer_builder::BufferBuilder::new(size)
            .indirect_buffer()
            .storage_buffer()
            .transfer_dst()
            .gpu_only()
            .named("Indirect Draw Buffer")
            .build(self)
    }

    /// Create readback buffer (GPU → CPU transfer)
    pub fn create_readback_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> crate::Result<(vk::Buffer, vk_mem::Allocation)> {
        crate::vulkan::buffer_builder::BufferBuilder::new(size)
            .transfer_dst()
            .cpu_readable()
            .named("Readback Buffer")
            .build(self)
    }

    /// Map an allocation and return the raw pointer.
    ///
    /// # Safety
    /// // SAFETY: Must be matched with unmap_allocation.
    pub unsafe fn map_allocation(
        &self,
        allocation: &mut vk_mem::Allocation,
    ) -> crate::Result<*mut u8> {
        self.vma
            .map_memory(allocation)
            .map_err(|e| crate::AshError::VulkanError(format!("Map memory failed: {e:?}")))
    }

    /// Unmap an allocation.
    ///
    /// # Safety
    /// // SAFETY: Must be called only if mapped.
    pub unsafe fn unmap_allocation(&self, allocation: &mut vk_mem::Allocation) {
        self.vma.unmap_memory(allocation);
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
        let ptr = self.map_allocation(allocation)?;

        Ok(MapGuard {
            vma: &self.vma,
            allocation,
            ptr,
            size,
        })
    }

    /*
    /// Calculate aggregate statistics about memory usage.
    pub fn get_statistics(&self) -> vk_mem::Statistics {
        self.vma.calculate_statistics().unwrap_or_default()
    }

    /// Query current heap budgets.
    /// Requires VK_EXT_memory_budget to be enabled on the device.
    pub fn get_heap_budgets(&self) -> Vec<vk_mem::Statistics> {
        self.vma.get_heap_budgets().unwrap_or_default()
    }
    */
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
        std::slice::from_raw_parts(self.ptr as *const T, count)
    }

    /// Access mapped memory as a mutable slice of a specific type.
    ///
    /// # Safety
    /// // SAFETY: Type T must be compatible with the mapped data.
    pub unsafe fn as_mut_slice_t<T: Copy>(&mut self) -> &mut [T] {
        let count = self.size as usize / std::mem::size_of::<T>();
        std::slice::from_raw_parts_mut(self.ptr as *mut T, count)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_validation_logic() {
        // Test the standalone validation function directly without needing an Allocator instance
        // Test 1: Size 0 should fail
        let res = validate_buffer_params_impl(
            0,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        );
        assert!(res.is_err(), "Size 0 should be rejected");

        // Test 2: GPU-only + Mapped should fail
        let res = validate_buffer_params_impl(
            1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::MAPPED,
        );
        assert!(res.is_err(), "GPU-only + Mapped should be rejected");

        // Test 3: Valid params should pass
        let res = validate_buffer_params_impl(
            1024,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        );
        assert!(res.is_ok(), "Valid params should pass");
    }
}
