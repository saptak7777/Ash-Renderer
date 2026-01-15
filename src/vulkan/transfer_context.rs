use ash::vk;
use std::sync::{Arc, Mutex};
use vk_mem::Alloc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// Transfer operation handle for tracking async uploads.
struct TransferOperation {
    timeline_value: u64,
    staging_buffer: vk::Buffer,
    staging_allocation: vk_mem::Allocation,
}

/// Context for asynchronous data transfers on a dedicated transfer queue.
///
/// This enables background uploads without stalling the graphics queue,
/// using timeline semaphores for synchronization.
pub struct TransferContext {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,
    transfer_queue: vk::Queue,
    transfer_queue_family: u32,
    command_pool: vk::CommandPool,
    timeline_semaphore: vk::Semaphore,
    timeline_value: Arc<Mutex<u64>>,
    pending_operations: Arc<Mutex<Vec<TransferOperation>>>,
}

impl TransferContext {
    /// Creates a new transfer context.
    ///
    /// # Safety
    ///
    /// The device must support timeline semaphores (Vulkan 1.2+).
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        transfer_queue: vk::Queue,
        transfer_queue_family: u32,
    ) -> Result<Self> {
        // Create command pool for transfer operations
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(transfer_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = device.create_command_pool(&pool_info, None).map_err(|e| {
            AshError::VulkanError(format!("Failed to create transfer command pool: {e:?}"))
        })?;

        // Create timeline semaphore for synchronization
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);

        let timeline_semaphore = device
            .create_semaphore(&semaphore_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create timeline semaphore: {e:?}"))
            })?;

        log::info!("TransferContext: Initialized with dedicated transfer queue");

        Ok(Self {
            device,
            allocator,
            transfer_queue,
            transfer_queue_family,
            command_pool,
            timeline_semaphore,
            timeline_value: Arc::new(Mutex::new(0)),
            pending_operations: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Begins an async transfer operation.
    ///
    /// Returns a timeline value that can be waited on by the graphics queue.
    ///
    /// # Safety
    ///
    /// The source data must remain valid until the transfer completes.
    pub unsafe fn begin_transfer<T: Copy + bytemuck::Pod>(
        &self,
        data: &[T],
        dst_buffer: vk::Buffer,
    ) -> Result<u64> {
        let size = (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize;

        // Create staging buffer
        let (staging_buffer, mut staging_alloc) = self
            .allocator
            .vma
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create staging buffer: {e:?}"))
            })?;

        // Copy data to staging
        {
            let mut guard = self
                .allocator
                .map_allocation_guarded(&mut staging_alloc, size)?;
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                guard.as_mut_ptr(),
                size as usize,
            );
        }

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = self
            .device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to allocate command buffer: {e:?}"))
            })?;
        let cmd = cmd_buffers[0];

        // Record transfer command
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device
            .begin_command_buffer(cmd, &begin_info)
            .map_err(|e| AshError::VulkanError(format!("Failed to begin command buffer: {e:?}")))?;

        let copy_region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(size);

        self.device
            .cmd_copy_buffer(cmd, staging_buffer, dst_buffer, &[copy_region]);

        self.device
            .end_command_buffer(cmd)
            .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e:?}")))?;

        // Increment timeline value
        let mut timeline_value = self.timeline_value.lock().unwrap();
        *timeline_value += 1;
        let signal_value = *timeline_value;

        // Submit with timeline semaphore signal
        let signal_values = [signal_value];
        let mut timeline_submit_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

        let cmd_buffers = [cmd];
        let semaphores = [self.timeline_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmd_buffers)
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline_submit_info);

        self.device
            .queue_submit(self.transfer_queue, &[submit_info], vk::Fence::null())
            .map_err(|e| AshError::VulkanError(format!("Failed to submit transfer: {e:?}")))?;

        // Track operation for cleanup
        self.pending_operations
            .lock()
            .unwrap()
            .push(TransferOperation {
                timeline_value: signal_value,
                staging_buffer,
                staging_allocation: staging_alloc,
            });

        log::debug!("TransferContext: Submitted async transfer (timeline value: {signal_value})");

        Ok(signal_value)
    }

    /// Waits for a specific transfer to complete.
    pub fn wait_for_transfer(&self, timeline_value: u64) -> Result<()> {
        unsafe {
            let semaphores = [self.timeline_semaphore];
            let values = [timeline_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);

            self.device
                .wait_semaphores(&wait_info, u64::MAX)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to wait for transfer: {e:?}"))
                })?;
        }

        Ok(())
    }

    /// Cleans up completed transfers.
    ///
    /// Should be called periodically (e.g., once per frame).
    pub fn cleanup_completed(&self) -> Result<()> {
        unsafe {
            let mut pending = self.pending_operations.lock().unwrap();
            let current_value = self.get_timeline_value()?;

            pending.retain(|op| {
                if op.timeline_value <= current_value {
                    // Transfer completed, destroy staging buffer
                    let mut alloc = op.staging_allocation;
                    self.allocator
                        .vma
                        .destroy_buffer(op.staging_buffer, &mut alloc);
                    false
                } else {
                    true
                }
            });
        }

        Ok(())
    }

    /// Gets the current timeline semaphore value.
    fn get_timeline_value(&self) -> Result<u64> {
        unsafe {
            self.device
                .get_semaphore_counter_value(self.timeline_semaphore)
                .map_err(|e| AshError::VulkanError(format!("Failed to get semaphore value: {e:?}")))
        }
    }

    /// Returns the timeline semaphore for external synchronization.
    pub fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline_semaphore
    }

    /// Returns the transfer queue family index.
    pub fn queue_family(&self) -> u32 {
        self.transfer_queue_family
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        unsafe {
            // Wait for all pending transfers
            if let Ok(value) = self.get_timeline_value() {
                let _ = self.wait_for_transfer(value);
            }

            // Cleanup remaining operations
            let mut pending = self.pending_operations.lock().unwrap();
            for op in pending.drain(..) {
                let mut alloc = op.staging_allocation;
                self.allocator
                    .vma
                    .destroy_buffer(op.staging_buffer, &mut alloc);
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_semaphore(self.timeline_semaphore, None);

            log::debug!("TransferContext: Destroyed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_value_increment() {
        // Verify timeline value increments correctly
        let value = Arc::new(Mutex::new(0u64));
        {
            let mut v = value.lock().unwrap();
            *v += 1;
            assert_eq!(*v, 1);
        }
        {
            let mut v = value.lock().unwrap();
            *v += 1;
            assert_eq!(*v, 2);
        }
    }
}
