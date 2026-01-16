use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

use crate::renderer::resources::ImageHandle;
use crate::vulkan::Allocator;

/// Key for bucketing transient resources by their properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResourceKey {
    format: vk::Format,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
    mip_levels: u32,
}

impl ResourceKey {
    fn new(
        format: vk::Format,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
    ) -> Self {
        Self {
            format,
            width: extent.width,
            height: extent.height,
            usage,
            mip_levels,
        }
    }
}

/// A pooled transient resource with lifecycle tracking.
struct TransientResource {
    image: ImageHandle,
    last_used_frame: u64,
}

/// Resource pool for reusing temporary render targets.
///
/// This eliminates per-frame allocation overhead by maintaining a pool
/// of resources bucketed by their properties (format, size, usage).
pub struct ResourcePool {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,
    pools: HashMap<ResourceKey, Vec<TransientResource>>,
    current_frame: u64,
}

impl ResourcePool {
    /// Creates a new resource pool.
    pub fn new(device: Arc<ash::Device>, allocator: Arc<Allocator>) -> Self {
        log::info!("ResourcePool: Initialized");
        Self {
            device,
            allocator,
            pools: HashMap::with_capacity(64), // Pre-allocate for common resource types
            current_frame: 0,
        }
    }

    /// Allocates a transient image, reusing from pool if available.
    ///
    /// # Safety
    ///
    /// The caller must ensure the image is not used after being released back to the pool.
    pub unsafe fn allocate_transient(
        &mut self,
        format: vk::Format,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
    ) -> crate::Result<ImageHandle> {
        let key = ResourceKey::new(format, extent, usage, mip_levels);

        // Try to reuse from pool
        if let Some(pool) = self.pools.get_mut(&key) {
            if let Some(mut resource) = pool.pop() {
                resource.last_used_frame = self.current_frame;
                log::debug!(
                    "ResourcePool: Reused transient image ({}x{}, {:?})",
                    extent.width,
                    extent.height,
                    format
                );
                return Ok(resource.image);
            }
        }

        // Create new resource
        log::debug!(
            "ResourcePool: Creating new transient image ({}x{}, {:?})",
            extent.width,
            extent.height,
            format
        );

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, view, allocation) = self.allocator.create_image_with_view(
            image_info,
            vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            },
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
        )?;

        let image_handle = ImageHandle::new_with_allocation(
            Arc::clone(&self.device),
            image,
            view,
            format,
            extent,
            mip_levels,
            1,
            Some(allocation),
            Some(Arc::clone(&self.allocator)),
            Some(format!(
                "transient_{width}x{height}",
                width = extent.width,
                height = extent.height
            )),
        )?;

        Ok(image_handle)
    }

    /// Releases a transient image back to the pool for reuse.
    pub fn release_transient(&mut self, image: ImageHandle) {
        let key = ResourceKey::new(
            image.format(),
            image.extent(),
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED, // Inferred usage
            image.mip_levels(),
        );

        let resource = TransientResource {
            image,
            last_used_frame: self.current_frame,
        };

        self.pools.entry(key).or_default().push(resource);
    }

    /// Advances to the next frame and cleans up old resources.
    ///
    /// Resources unused for more than `max_age` frames are destroyed.
    pub fn next_frame(&mut self, max_age: u64) {
        self.current_frame += 1;

        // Cleanup old resources
        let current_frame = self.current_frame;
        self.pools.retain(|key, resources| {
            let initial_count = resources.len();
            resources.retain(|r| current_frame - r.last_used_frame < max_age);
            let removed = initial_count - resources.len();
            if removed > 0 {
                log::debug!(
                    "ResourcePool: Cleaned up {} old resources ({}x{}, {:?})",
                    removed,
                    key.width,
                    key.height,
                    key.format
                );
            }
            !resources.is_empty()
        });
    }

    /// Returns the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Returns statistics about the pool.
    pub fn stats(&self) -> PoolStats {
        let total_resources: usize = self.pools.values().map(|v| v.len()).sum();
        let bucket_count = self.pools.len();

        PoolStats {
            total_resources,
            bucket_count,
            current_frame: self.current_frame,
        }
    }
}

/// Statistics about the resource pool.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub total_resources: usize,
    pub bucket_count: usize,
    pub current_frame: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_key_equality() {
        let key1 = ResourceKey::new(
            vk::Format::R8G8B8A8_UNORM,
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
        );

        let key2 = ResourceKey::new(
            vk::Format::R8G8B8A8_UNORM,
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
        );

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_resource_key_inequality() {
        let key1 = ResourceKey::new(
            vk::Format::R8G8B8A8_UNORM,
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
        );

        let key2 = ResourceKey::new(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
        );

        assert_ne!(key1, key2);
    }
}
