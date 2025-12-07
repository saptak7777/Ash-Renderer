use std::sync::Mutex;

use vk_mem::Allocator;

/// Trait for Vulkan resources that can be cleaned up with a device or allocator.
pub trait VulkanResourceCleanup: Send + Sync + 'static {
    /// Clean up the resource using a Vulkan device.
    fn cleanup_with_device(&mut self, device: &ash::Device) -> Result<(), String>;

    /// Clean up the resource using a VMA allocator wrapped in a mutex.
    fn cleanup_with_allocator(&mut self, allocator: &Mutex<Allocator>) -> Result<(), String> {
        match allocator.lock() {
            Ok(mut alloc) => self.cleanup_with_allocator_inner(&mut alloc),
            Err(_) => Err("Failed to acquire allocator lock".to_string()),
        }
    }

    /// Internal hook that gives direct mutable access to the allocator implementation.
    fn cleanup_with_allocator_inner(&mut self, _allocator: &mut Allocator) -> Result<(), String> {
        Ok(())
    }

    /// Returns the resource type used for logging/debugging.
    fn resource_type(&self) -> &'static str;
}

/// Helper trait for cleaning up collections of buffer-like resources.
pub trait BufferCleanup: VulkanResourceCleanup {
    fn cleanup_buffers<T: VulkanResourceCleanup>(
        buffers: &mut Vec<T>,
        allocator: &Mutex<Allocator>,
        resource_name: &str,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for (index, mut buffer) in buffers.drain(..).enumerate() {
            if let Err(e) = buffer.cleanup_with_allocator(allocator) {
                errors.push(format!("Failed to cleanup {resource_name} {index}: {e}"));
            }
        }
        errors
    }
}
