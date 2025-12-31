use crate::AshError;
use crate::Result;
use ash::vk;

/// Real-time GPU memory tracking and budgeting system.
/// Prevents ERROR_DEVICE_LOST by enforcing allocation limits.
pub struct VramBudget {
    total_raw_vram: vk::DeviceSize,
    _heap_index: u32,
    _reserved_engine: vk::DeviceSize, // Safety margin (e.g., 20% for G-buffers, swapchain, etc.)
    used_textures: vk::DeviceSize,
    safety_threshold: vk::DeviceSize, // Maximum allowed for textures/dynamic resources (e.g., 80%)
}

impl VramBudget {
    /// Create a new budget tracker based on physical device properties.
    pub fn new(memory_properties: &vk::PhysicalDeviceMemoryProperties) -> Self {
        // Find the largest DEVICE_LOCAL heap
        let mut largest_heap_index = 0;
        let mut largest_size = 0;

        for (i, heap) in memory_properties.memory_heaps.iter().enumerate() {
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) && heap.size > largest_size {
                largest_size = heap.size;
                largest_heap_index = i as u32;
            }
        }

        let total_raw_vram = largest_size;
        let reserved_engine = total_raw_vram / 5; // Reserve 20%
        let safety_threshold = (total_raw_vram / 5) * 4; // Use up to 80%

        log::info!(
            "VRAM Budget initialized for heap {}: Total {}MB, Safety Threshold {}MB",
            largest_heap_index,
            total_raw_vram / 1024 / 1024,
            safety_threshold / 1024 / 1024
        );

        Self {
            total_raw_vram,
            _heap_index: largest_heap_index,
            _reserved_engine: reserved_engine,
            used_textures: 0,
            safety_threshold,
        }
    }

    /// Check if a new allocation of `size` fits within the budget.
    pub fn can_allocate(&self, size: vk::DeviceSize) -> Result<()> {
        if self.used_textures + size > self.safety_threshold {
            let available = self.safety_threshold.saturating_sub(self.used_textures);
            Err(AshError::VramExhausted {
                requested: size,
                available,
                recommendation:
                    "Reduce texture resolution, disable high-res assets, or free GPU resources.",
            })
        } else {
            Ok(())
        }
    }

    /// Record an allocation.
    pub fn allocate(&mut self, size: vk::DeviceSize) {
        self.used_textures += size;
    }

    /// Record a deallocation.
    pub fn deallocate(&mut self, size: vk::DeviceSize) {
        self.used_textures = self.used_textures.saturating_sub(size);
    }

    /// Get current utilization percentage relative to the safety threshold.
    pub fn utilization_percent(&self) -> f32 {
        (self.used_textures as f32 / self.safety_threshold as f32) * 100.0
    }

    /// Get detailed statistics for telemetry.
    pub fn get_stats(&self) -> VramStats {
        VramStats {
            total_vram: self.total_raw_vram,
            used_textures: self.used_textures,
            available_budget: self.safety_threshold.saturating_sub(self.used_textures),
            utilization_percent: self.utilization_percent(),
        }
    }
}

/// Statistics snapshot for a single frame.
#[derive(Debug, Clone, Copy)]
pub struct VramStats {
    pub total_vram: vk::DeviceSize,
    pub used_textures: vk::DeviceSize,
    pub available_budget: vk::DeviceSize,
    pub utilization_percent: f32,
}
