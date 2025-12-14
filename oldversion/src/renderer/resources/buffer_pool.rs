use crate::vulkan::Allocator;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Buffer allocation metadata
#[derive(Debug, Clone)]
pub struct BufferAllocation {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub offset: u64,
    pub name: Option<String>,
}

/// Efficient buffer pool for reusing allocations
pub struct BufferPool {
    allocator: Arc<Allocator>,
    pools: Mutex<BufferPoolInner>,
}

struct BufferPoolInner {
    available: VecDeque<BufferAllocation>,
    in_use: Vec<BufferAllocation>,
    total_allocated: u64,
}

impl BufferPool {
    /// Creates a new buffer pool
    pub fn new(allocator: Arc<Allocator>) -> Self {
        Self {
            allocator,
            pools: Mutex::new(BufferPoolInner {
                available: VecDeque::new(),
                in_use: Vec::new(),
                total_allocated: 0,
            }),
        }
    }

    /// Allocates a buffer from the pool
    ///
    /// # Safety
    ///
    /// This function calls unsafe Vulkan allocation routines. Caller must ensure:
    /// - `size` is greater than 0
    /// - `usage` flags are valid for the target GPU
    /// - Allocated buffers are returned via `deallocate` before pool destruction
    pub unsafe fn allocate(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        name: Option<String>,
    ) -> crate::Result<BufferAllocation> {
        let mut pools = self.pools.lock().unwrap();

        // Try to find a reusable buffer
        if let Some(mut alloc) = pools.available.pop_front() {
            if alloc.size >= size {
                if let Some(ref n) = name {
                    log::debug!("Reusing buffer '{n}' ({size} bytes)");
                }
                alloc.name = name;
                pools.in_use.push(alloc.clone());
                return Ok(alloc);
            } else {
                // Put it back if too small
                pools.available.push_back(alloc);
            }
        }

        // Allocate new buffer
        if let Some(ref n) = name {
            log::info!("Allocating new buffer '{n}' ({size} bytes)");
        } else {
            log::info!("Allocating new buffer ({size} bytes)");
        }

        let (buffer, _allocation) = self.allocator.create_buffer(size, usage, memory_usage)?;

        pools.total_allocated += size;

        let alloc = BufferAllocation {
            buffer,
            size,
            offset: 0,
            name,
        };

        pools.in_use.push(alloc.clone());
        Ok(alloc)
    }

    /// Returns a buffer to the pool
    pub fn deallocate(&self, buffer: BufferAllocation) {
        let mut pools = self.pools.lock().unwrap();

        // Remove from in_use
        pools.in_use.retain(|b| b.buffer != buffer.buffer);

        // Add to available
        if let Some(ref name) = buffer.name {
            log::debug!("Returning buffer '{name}' to pool");
        }
        pools.available.push_back(buffer);
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, u64) {
        let pools = self.pools.lock().unwrap();
        (
            pools.available.len(),
            pools.in_use.len(),
            pools.total_allocated,
        )
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        if let Ok(pools) = self.pools.lock() {
            log::info!(
                "Buffer pool destroyed: {} available, {} in use, {total_allocated} bytes allocated",
                pools.available.len(),
                pools.in_use.len(),
                total_allocated = pools.total_allocated
            );
        }
    }
}
