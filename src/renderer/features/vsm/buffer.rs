//! VSM Buffer Abstractions

use crate::Result;
use crate::vulkan::Allocator;
use ash::vk;

/// VSM Request Buffer - Stores page requests from the GPU
pub struct VsmRequestBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub size_bytes: u64,
    pub max_requests: u32,
}

pub const ATOMIC_HEADER_SIZE: vk::DeviceSize = 16;

impl VsmRequestBuffer {
    /// Read requests from the buffer
    pub fn read_requests(
        &mut self,
        allocator: &Allocator,
    ) -> Result<(
        Vec<crate::renderer::features::vsm::resources::PageRequest>,
        u32,
    )> {
        use crate::renderer::features::vsm::resources::PageRequest;

        // 1. Map Memory using the engine's guarded wrapper
        let guard =
            unsafe { allocator.map_allocation_guarded(&mut self.allocation, self.size_bytes)? };

        // 2. Read Atomic Header (Index 0: Count, Index 1: Overflow) and Validate
        let ptr = guard.as_ptr() as *const u32;
        let count = unsafe { *ptr };
        let overflow_count = unsafe { *ptr.add(1) };

        // Defensive: GPU corruption or driver bugs could return garbage
        let safe_count = count.min(self.max_requests);
        if safe_count != count {
            log::warn!(
                "VSM: Internal Sync Failure - reported count {} exceeds max_requests {} - clamping",
                count,
                self.max_requests
            );
        }

        let requests = if safe_count > 0 {
            unsafe {
                // 4. Data starts at offset ATOMIC_HEADER_SIZE
                let data_ptr =
                    guard.as_ptr().add(ATOMIC_HEADER_SIZE as usize) as *const PageRequest;
                let slice = std::slice::from_raw_parts(data_ptr, safe_count as usize);
                slice.to_vec()
            }
        } else {
            Vec::new()
        };

        Ok((requests, overflow_count))
    }
}
