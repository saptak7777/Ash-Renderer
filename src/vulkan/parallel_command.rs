use ash::vk;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::{AshError, Result};

const SECONDARY_BATCH_SIZE: u32 = 4;

struct WorkerState {
    command_pool: vk::CommandPool,
    available_secondary: Vec<vk::CommandBuffer>,
    /// Buffers queued for reset on next acquire (avoids blocking GPU)
    pending_reset: Vec<vk::CommandBuffer>,
}

impl WorkerState {
    fn new(device: &ash::Device, queue_family_index: u32) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );

        let command_pool = unsafe {
            device.create_command_pool(&pool_info, None).map_err(|e| {
                AshError::VulkanError(format!("Failed to create worker command pool: {e}"))
            })?
        };

        Ok(Self {
            command_pool,
            available_secondary: Vec::new(),
            pending_reset: Vec::new(),
        })
    }
}

/// Manages per-worker command pools and cached secondary command buffers for parallel recording.
pub struct ParallelCommandManager {
    device: Arc<ash::Device>,
    #[allow(dead_code)] // Reserved for future use (e.g., transfer queue)
    queue_family_index: u32,
    primary_pool: vk::CommandPool,
    tracked_primary: Mutex<Vec<vk::CommandBuffer>>,
    workers: Vec<Mutex<WorkerState>>,
}

impl ParallelCommandManager {
    pub fn new(
        device: Arc<ash::Device>,
        queue_family_index: u32,
        worker_count: usize,
    ) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let primary_pool = unsafe {
            device.create_command_pool(&pool_info, None).map_err(|e| {
                AshError::VulkanError(format!("Failed to create primary command pool: {e}"))
            })?
        };

        let mut workers = Vec::with_capacity(worker_count.max(1));
        for _ in 0..worker_count.max(1) {
            workers.push(Mutex::new(WorkerState::new(&device, queue_family_index)?));
        }

        Ok(Self {
            device,
            queue_family_index,
            primary_pool,
            tracked_primary: Mutex::new(Vec::new()),
            workers,
        })
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn allocate_primary_buffers(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.primary_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let buffers = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to allocate primary buffers: {e}"))
                })?
        };

        self.tracked_primary.lock().extend_from_slice(&buffers);
        Ok(buffers)
    }

    pub fn acquire_secondary(&self, worker_index: usize) -> Result<vk::CommandBuffer> {
        let worker = self
            .workers
            .get(worker_index)
            .ok_or_else(|| AshError::VulkanError("Worker index out of range".into()))?;

        // Phase 1: Process pending resets (GPU guaranteed done with these)
        // Phase 2: Try to get a buffer
        // Phase 3: If none, allocate new batch

        let (buffer_to_reset, need_alloc) = {
            let mut state = worker.lock();

            // Process any pending resets first (these were recycled last frame)
            let pending: Vec<_> = state.pending_reset.drain(..).collect();
            state.available_secondary.extend(pending);

            if let Some(buffer) = state.available_secondary.pop() {
                (Some(buffer), false)
            } else {
                (None, true)
            }
        }; // Lock released here

        if let Some(buffer) = buffer_to_reset {
            // Reset outside of lock to avoid contention
            self.reset_command_buffer(buffer)?;
            return Ok(buffer);
        }

        if need_alloc {
            // Re-acquire lock for allocation
            let mut state = worker.lock();
            let buffers = self.allocate_secondary_batch(&mut state, SECONDARY_BATCH_SIZE)?;
            let mut iter = buffers.into_iter();
            let next = iter.next().ok_or_else(|| {
                AshError::VulkanError("Failed to allocate secondary buffer".into())
            })?;
            state.available_secondary.extend(iter);
            drop(state); // Release lock before reset
            self.reset_command_buffer(next)?;
            return Ok(next);
        }

        Err(AshError::VulkanError(
            "Failed to acquire secondary buffer".into(),
        ))
    }

    pub fn recycle_secondary(&self, worker_index: usize, buffer: vk::CommandBuffer) -> Result<()> {
        if buffer == vk::CommandBuffer::null() {
            return Ok(());
        }

        let worker = self
            .workers
            .get(worker_index)
            .ok_or_else(|| AshError::VulkanError("Worker index out of range".into()))?;
        // Queue for deferred reset (GPU may still be using it this frame)
        worker.lock().pending_reset.push(buffer);
        Ok(())
    }

    pub fn destroy_all_pools(&self) -> Result<()> {
        unsafe {
            let buffers = self.tracked_primary.lock().drain(..).collect::<Vec<_>>();
            if !buffers.is_empty() {
                self.device
                    .free_command_buffers(self.primary_pool, &buffers);
            }
            self.device.destroy_command_pool(self.primary_pool, None);
        }

        for worker in &self.workers {
            let mut state = worker.lock();
            unsafe {
                if !state.available_secondary.is_empty() {
                    self.device
                        .free_command_buffers(state.command_pool, &state.available_secondary);
                    state.available_secondary.clear();
                }
                self.device.destroy_command_pool(state.command_pool, None);
                state.command_pool = vk::CommandPool::null();
            }
        }

        Ok(())
    }

    fn allocate_secondary_batch(
        &self,
        state: &mut WorkerState,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(state.command_pool)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(count);

        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to allocate secondary command buffers: {e}"
                    ))
                })
        }
    }

    fn reset_command_buffer(&self, buffer: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to reset secondary command buffer: {e}"))
                })
        }
    }
}

impl Drop for ParallelCommandManager {
    fn drop(&mut self) {
        if let Err(err) = self.destroy_all_pools() {
            log::error!("ParallelCommandManager cleanup failed: {err}");
        }
    }
}
