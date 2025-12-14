use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Convenience wrapper around a Vulkan command pool that tracks allocated buffers so they can be
/// freed automatically when the pool is dropped (unless the resource registry owns them).
pub struct CommandPool {
    device: Arc<ash::Device>,
    pool: vk::CommandPool,
    queue_family_index: u32,
    tracked_buffers: Vec<vk::CommandBuffer>,
    manage_buffers: bool,
    managed_by_registry: bool,
}

impl CommandPool {
    pub fn new(device: Arc<ash::Device>, queue_family_index: u32) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );

        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(|e| AshError::VulkanError(format!("Failed to create command pool: {e}")))?
        };

        Ok(Self {
            device,
            pool,
            queue_family_index,
            tracked_buffers: Vec::new(),
            manage_buffers: true,
            managed_by_registry: false,
        })
    }

    pub fn allocate_primary_buffers(&mut self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let buffers = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to allocate command buffers: {e}"))
                })?
        };

        if self.manage_buffers {
            self.tracked_buffers.extend_from_slice(&buffers);
        }

        Ok(buffers)
    }

    pub fn allocate_secondary_buffers(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(count);

        let buffers = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to allocate secondary command buffers: {e}"
                    ))
                })?
        };

        Ok(buffers)
    }

    pub fn begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
                })?
        }
        Ok(())
    }

    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?
        }
        Ok(())
    }

    pub fn reset(&mut self, flags: vk::CommandPoolResetFlags) -> Result<()> {
        unsafe {
            self.device
                .reset_command_pool(self.pool, flags)
                .map_err(|e| AshError::VulkanError(format!("Failed to reset command pool: {e}")))?;
        }

        if flags.contains(vk::CommandPoolResetFlags::RELEASE_RESOURCES) {
            self.tracked_buffers.clear();
        }

        Ok(())
    }

    pub fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn mark_managed_by_registry(&mut self) {
        self.managed_by_registry = true;
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }

        unsafe {
            if self.manage_buffers && !self.tracked_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.pool, &self.tracked_buffers);
            }
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
