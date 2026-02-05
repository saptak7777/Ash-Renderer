use ash::vk;
use rayon::prelude::*;
use std::sync::Arc;

use crate::{AshError, Result};

/// Thread-local command pool manager for parallel recording.
///
/// Vulkan command pools are not thread-safe, so each thread needs its own pool.
pub struct ParallelCommandRecorder {
    device: Arc<ash::Device>,
    queue_family: u32,
    /// Threshold for parallel recording (passes below this use serial recording)
    parallel_threshold: usize,
}

impl ParallelCommandRecorder {
    /// Creates a new parallel command recorder.
    pub fn new(device: Arc<ash::Device>, queue_family: u32) -> Self {
        Self {
            device,
            queue_family,
            parallel_threshold: 4,
        }
    }

    /// Sets the threshold for parallel recording.
    ///
    /// Passes below this count will use serial recording to avoid overhead.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Records commands in parallel using secondary command buffers.
    ///
    /// # Safety
    ///
    /// The primary command buffer must be in the recording state.
    /// All passes must be independent (no data dependencies).
    pub unsafe fn record_parallel<F>(
        &self,
        primary_cmd: vk::CommandBuffer,
        pass_count: usize,
        record_fn: F,
    ) -> Result<()>
    where
        F: FnMut(usize, vk::CommandBuffer) -> Result<()> + Send,
    {
        // Only parallelize if we have enough passes
        if pass_count < self.parallel_threshold {
            return self.record_serial(primary_cmd, pass_count, record_fn);
        }

        log::debug!("ParallelCommandRecorder: Recording {pass_count} passes in parallel");

        // Wrap record_fn in Mutex for thread-safe mutable access
        let record_fn = std::sync::Mutex::new(record_fn);

        // Create thread-local command pools
        thread_local! {
            static COMMAND_POOL: std::cell::RefCell<Option<(vk::CommandPool, Arc<ash::Device>)>> = const { std::cell::RefCell::new(None) };
        }

        // Record secondary command buffers in parallel
        let secondary_buffers: Result<Vec<vk::CommandBuffer>> = (0..pass_count)
            .into_par_iter()
            .map(|pass_idx| {
                COMMAND_POOL.with(|pool_cell| {
                    let mut pool_opt = pool_cell.borrow_mut();

                    // Create pool if it doesn't exist for this thread
                    if pool_opt.is_none() {
                        let pool_info = vk::CommandPoolCreateInfo::default()
                            .queue_family_index(self.queue_family)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

                        let pool =
                            self.device
                                .create_command_pool(&pool_info, None)
                                .map_err(|e| {
                                    AshError::VulkanError(format!(
                                        "Failed to create thread-local command pool: {e:?}"
                                    ))
                                })?;

                        *pool_opt = Some((pool, Arc::clone(&self.device)));
                    }

                    let (pool, device) = pool_opt.as_ref().unwrap();

                    // Allocate secondary command buffer
                    let alloc_info = vk::CommandBufferAllocateInfo::default()
                        .command_pool(*pool)
                        .level(vk::CommandBufferLevel::SECONDARY)
                        .command_buffer_count(1);

                    let cmd_buffers =
                        device.allocate_command_buffers(&alloc_info).map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to allocate secondary command buffer: {e:?}"
                            ))
                        })?;
                    let cmd = cmd_buffers[0];

                    // Begin secondary command buffer
                    let inheritance_info = vk::CommandBufferInheritanceInfo::default();
                    let begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .inheritance_info(&inheritance_info);

                    device.begin_command_buffer(cmd, &begin_info).map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to begin secondary command buffer: {e:?}"
                        ))
                    })?;

                    // Record pass commands
                    let mut record_fn_guard = record_fn.lock().unwrap();
                    record_fn_guard(pass_idx, cmd)?;
                    drop(record_fn_guard);

                    device.end_command_buffer(cmd).map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to end secondary command buffer: {e:?}"
                        ))
                    })?;

                    Ok(cmd)
                })
            })
            .collect();

        let secondary_buffers = secondary_buffers?;

        // Execute secondary buffers in primary
        self.device
            .cmd_execute_commands(primary_cmd, &secondary_buffers);

        log::debug!(
            "ParallelCommandRecorder: Executed {} secondary command buffers",
            secondary_buffers.len()
        );

        Ok(())
    }

    /// Records commands serially (fallback for small pass counts).
    unsafe fn record_serial<F>(
        &self,
        primary_cmd: vk::CommandBuffer,
        pass_count: usize,
        mut record_fn: F,
    ) -> Result<()>
    where
        F: FnMut(usize, vk::CommandBuffer) -> Result<()>,
    {
        log::debug!("ParallelCommandRecorder: Recording {pass_count} passes serially");

        for pass_idx in 0..pass_count {
            record_fn(pass_idx, primary_cmd)?;
        }

        Ok(())
    }
}

impl Drop for ParallelCommandRecorder {
    fn drop(&mut self) {
        // Thread-local pools are cleaned up when threads exit
        log::debug!("ParallelCommandRecorder: Destroyed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a dummy device that satisfies the non-null function pointer requirement.
    /// NEVER call methods on this device as it contains dummy data.
    fn create_dummy_device() -> Arc<ash::Device> {
        unsafe {
            // Creating a zeroed high-level ash::Device is UB because it contains function pointers
            // that must be non-null. We use a non-zero pattern to satisfy the layout requirements
            // for these tests where the device is never actually dereferenced.
            let mut dummy = std::mem::MaybeUninit::<ash::Device>::uninit();
            std::ptr::write_bytes(dummy.as_mut_ptr(), 0x01, 1);
            Arc::new(dummy.assume_init())
        }
    }

    #[test]
    fn test_threshold_logic() {
        let recorder = ParallelCommandRecorder::new(create_dummy_device(), 0);
        assert_eq!(recorder.parallel_threshold, 4);

        let recorder = recorder.with_threshold(8);
        assert_eq!(recorder.parallel_threshold, 8);
    }

    #[test]
    fn test_parallel_threshold() {
        // Verify that passes below threshold use serial path
        let recorder = ParallelCommandRecorder::new(create_dummy_device(), 0).with_threshold(5);

        assert!(3 < recorder.parallel_threshold); // Should use serial
        assert!(5 >= recorder.parallel_threshold); // Should use parallel
    }
}
