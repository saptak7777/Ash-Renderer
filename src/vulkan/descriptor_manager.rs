use log::info;

use std::sync::Arc;

use crate::renderer::resource_registry::ResourceRegistry;
use crate::Result;

use super::descriptor_allocator::DescriptorAllocator;

const EXTRA_TEXTURE_SETS: u32 = 2048;

/// Manages descriptor allocation for the renderer.
/// In Phase 4+, this serves primarily as an allocator for the BindlessManager.
pub struct DescriptorManager {
    allocator: DescriptorAllocator,
}

impl DescriptorManager {
    pub fn new(
        device: Arc<ash::Device>,
        frame_count: u32,
        resource_registry: Option<Arc<ResourceRegistry>>,
    ) -> Result<Self> {
        info!("Creating simplified descriptor manager for {frame_count} frames");

        let allocator =
            DescriptorAllocator::new(Arc::clone(&device), EXTRA_TEXTURE_SETS, resource_registry)?;

        Ok(Self { allocator })
    }

    pub fn next_frame(&mut self) {
        self.allocator.next_frame();
    }

    /// Get mutable access to the allocator for external allocation (e.g., bindless)
    pub fn allocator_mut(&mut self) -> &mut DescriptorAllocator {
        &mut self.allocator
    }
}
