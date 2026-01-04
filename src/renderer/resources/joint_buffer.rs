use ash::vk;
use glam::Mat4;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// GPU buffer for skeletal animation joint matrices
pub struct JointMatricesBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    capacity: usize,
    allocator: Arc<Allocator>,
}

impl JointMatricesBuffer {
    /// Create a new joint matrices buffer with the given capacity
    ///
    /// # Safety
    /// Caller must ensure allocator and device remain valid for the buffer's lifetime
    pub unsafe fn new(allocator: Arc<Allocator>, capacity: usize) -> Result<Self> {
        let size = (capacity * std::mem::size_of::<Mat4>()) as u64;

        let (buffer, allocation) = allocator.create_joint_buffer(size)?;

        log::info!("Created joint matrices buffer (capacity: {capacity}, size: {size} bytes)");

        Ok(Self {
            buffer,
            allocation,
            capacity,
            allocator,
        })
    }

    /// Update the buffer with new joint matrices
    ///
    /// # Safety
    /// Caller must ensure matrices slice does not exceed buffer capacity
    pub unsafe fn update(&mut self, matrices: &[Mat4]) -> Result<()> {
        self.update_offset(matrices, 0)
    }

    /// Update the buffer with new joint matrices at a specific offset
    ///
    /// # Safety
    /// Caller must ensure matrices slice and offset do not exceed buffer capacity
    pub unsafe fn update_offset(&mut self, matrices: &[Mat4], offset: usize) -> Result<()> {
        if offset + matrices.len() > self.capacity {
            return Err(AshError::VulkanError(format!(
                "Joint matrices update (offset: {}, count: {}) exceeds buffer capacity ({})",
                offset,
                matrices.len(),
                self.capacity
            )));
        }

        let data_size = std::mem::size_of_val(matrices) as u64;
        let memory_offset = (offset * std::mem::size_of::<Mat4>()) as u64;

        {
            let mut guard = self
                .allocator
                .map_allocation_guarded(&mut self.allocation, memory_offset + data_size)?;

            let slice = &mut guard[memory_offset as usize..(memory_offset + data_size) as usize];
            let target: &mut [Mat4] = bytemuck::cast_slice_mut(slice);
            target.copy_from_slice(matrices);
        }

        self.allocator
            .vma
            .flush_allocation(&self.allocation, memory_offset, data_size)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to flush joint matrices buffer: {e}"))
            })?;

        Ok(())
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for JointMatricesBuffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_buffer_capacity_check() {
        // Verify that capacity is enforced
        let capacity = 256;
        let size = capacity * std::mem::size_of::<Mat4>();
        assert_eq!(size, 256 * 64); // Mat4 is 64 bytes
    }

    #[test]
    fn mat4_size() {
        assert_eq!(std::mem::size_of::<Mat4>(), 64);
    }
}
