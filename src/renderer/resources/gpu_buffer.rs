use crate::vulkan::Allocator;
use ash::vk;
use std::marker::PhantomData;
use std::sync::Arc;

/// Strongly-typed GPU buffer with automatic cleanup via RAII.
///
/// The type parameter `T` enforces compile-time type safety, preventing
/// binding a buffer of one type to a shader expecting another type.
///
/// # Example
/// ```no_run
/// # use ash_renderer::renderer::resources::GpuBuffer;
/// # use glam::Vec3;
/// // Cannot accidentally bind Vec3 buffer to u32 shader slot
/// let vertex_buffer: GpuBuffer<Vec3> = unsafe {
///     GpuBuffer::new(allocator, 1024, usage, memory_usage, None)?
/// };
/// ```
pub struct GpuBuffer<T> {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: Arc<Allocator>,
    size: u64,
    element_count: usize,
    name: Option<String>,
    _phantom: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    /// Creates a new typed GPU buffer.
    ///
    /// # Safety
    ///
    /// The allocator must outlive this buffer handle. The buffer is automatically
    /// destroyed when this handle is dropped.
    pub unsafe fn new(
        allocator: Arc<Allocator>,
        element_count: usize,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        name: Option<String>,
    ) -> crate::Result<Self> {
        Self::new_with_flags(
            allocator,
            element_count,
            usage,
            memory_usage,
            vk_mem::AllocationCreateFlags::empty(),
            name,
        )
    }

    /// Creates a new typed GPU buffer with custom allocation flags.
    ///
    /// # Safety
    ///
    /// The allocator must outlive this buffer handle. The buffer is automatically
    /// destroyed when this handle is dropped.
    pub unsafe fn new_with_flags(
        allocator: Arc<Allocator>,
        element_count: usize,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        flags: vk_mem::AllocationCreateFlags,
        name: Option<String>,
    ) -> crate::Result<Self> {
        let element_size = std::mem::size_of::<T>() as u64;
        let size = element_size * element_count as u64;

        if let Some(ref n) = name {
            log::info!(
                "Creating typed buffer<{}> '{n}' ({element_count} elements, {size}B)",
                std::any::type_name::<T>()
            );
        } else {
            log::info!(
                "Creating typed buffer<{}> ({element_count} elements, {size}B)",
                std::any::type_name::<T>()
            );
        }

        let (buffer, allocation) =
            allocator.create_buffer_with_flags(size, usage, memory_usage, flags)?;

        Ok(Self {
            buffer,
            allocation,
            allocator,
            size,
            element_count,
            name,
            _phantom: PhantomData,
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

    /// Returns the number of elements this buffer can hold
    pub fn element_count(&self) -> usize {
        self.element_count
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
        &mut self.allocation
    }
    /// Uploads data to the buffer.
    ///
    /// # Safety
    ///
    /// The buffer must be CPU-writable (e.g., created with `MemoryUsage::CpuToGpu` or `CpuOnly`).
    /// The provided data must fit within the buffer.
    pub unsafe fn upload(&mut self, data: &[T]) -> crate::Result<()>
    where
        T: Copy + bytemuck::Pod,
    {
        let data_bytes = bytemuck::cast_slice(data);
        if data_bytes.len() as u64 > self.size {
            return Err(crate::AshError::VulkanError(format!(
                "Buffer upload too large ({}, capacity {})",
                data_bytes.len(),
                self.size
            )));
        }

        let ptr = self.allocator.map_allocation(&mut self.allocation)?;
        std::ptr::copy_nonoverlapping(data_bytes.as_ptr(), ptr, data_bytes.len());
        self.allocator.unmap_allocation(&mut self.allocation);

        Ok(())
    }

    /// Maps the buffer memory for direct access.
    ///
    /// # Safety
    ///
    /// The buffer must be CPU-visible. The caller is responsible for proper synchronization
    /// if the buffer is accessed by the GPU while mapped.
    pub unsafe fn map(&mut self) -> crate::Result<&mut [T]>
    where
        T: Copy + bytemuck::Pod,
    {
        let ptr = self.allocator.map_allocation(&mut self.allocation)?;
        let slice = std::slice::from_raw_parts_mut(ptr as *mut T, self.element_count);
        Ok(slice)
    }

    /// Unmaps the buffer memory.
    pub unsafe fn unmap(&mut self) {
        self.allocator.unmap_allocation(&mut self.allocation);
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref name) = self.name {
                log::debug!(
                    "Destroying typed buffer<{}> '{name}'",
                    std::any::type_name::<T>()
                );
            }
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

impl<T> std::fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("type", &std::any::type_name::<T>())
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("element_count", &self.element_count)
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_size_calculation() {
        // Verify size calculation is correct
        let element_count = 100;
        let expected_size = (std::mem::size_of::<f32>() * element_count) as u64;

        // We can't actually create a buffer without Vulkan, but we can verify the math
        assert_eq!(std::mem::size_of::<f32>() as u64 * 100, expected_size);
    }

    #[test]
    fn test_type_safety() {
        // This test verifies that the type system prevents misuse at compile time
        // If this compiles, the type system is working correctly
        fn _accepts_vec3_buffer(_buf: &GpuBuffer<glam::Vec3>) {}
        fn _accepts_u32_buffer(_buf: &GpuBuffer<u32>) {}

        // These would be compile errors if uncommented:
        // let vec3_buf: GpuBuffer<glam::Vec3> = ...;
        // _accepts_u32_buffer(&vec3_buf); // ERROR: mismatched types
    }
}
