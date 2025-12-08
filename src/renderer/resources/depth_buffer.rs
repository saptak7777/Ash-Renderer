use ash::vk;
use std::sync::Arc;

use crate::renderer::resource_registry::{ResourceError, ResourceId, ResourceRegistry};
use crate::vulkan::Allocator;

/// Depth buffer wrapper for Z-fighting prevention
pub struct DepthBuffer {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<vk_mem::Allocation>,
    allocator: Arc<Allocator>,
    device: Arc<ash::Device>,
    format: vk::Format,
    managed_by_registry: bool,
}

impl DepthBuffer {
    /// Creates a new depth buffer with the given dimensions
    ///
    /// # Safety
    ///
    /// Device must remain valid for the lifetime of this buffer.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
    ) -> crate::Result<Self> {
        Self::with_sample_count(
            device,
            allocator,
            width,
            height,
            vk::SampleCountFlags::TYPE_1,
        )
    }

    /// Creates a new depth buffer with the given dimensions and sample count
    ///
    /// # Safety
    ///
    /// Device must remain valid for the lifetime of this buffer.
    pub unsafe fn with_sample_count(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
        sample_count: vk::SampleCountFlags,
    ) -> crate::Result<Self> {
        let format = vk::Format::D32_SFLOAT;

        log::info!("Creating depth buffer ({width}x{height}, samples: {sample_count:?})");

        // Create depth image
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(sample_count)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            allocator.create_image(&image_create_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

        // Create image view
        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = device.create_image_view(&view_create_info, None)?;

        log::info!("Depth buffer created successfully");

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            allocator,
            device,
            format,
            managed_by_registry: false,
        })
    }

    pub fn register_with_registry(
        &mut self,
        registry: &ResourceRegistry,
    ) -> Result<ResourceId, ResourceError> {
        if self.managed_by_registry {
            return Err(ResourceError::InvalidDependency(
                "Depth buffer already registered with registry".to_string(),
            ));
        }

        let allocation = self
            .allocation
            .take()
            .expect("Depth buffer allocation already taken");

        let id = registry.register_depth_buffer(
            self.image,
            self.view,
            allocation,
            Arc::clone(&self.allocator),
        )?;

        self.managed_by_registry = true;
        Ok(id)
    }

    /// Returns the depth image view
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Returns the depth image
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Returns the depth format
    pub fn format(&self) -> vk::Format {
        self.format
    }
}

impl Drop for DepthBuffer {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }
        if let Some(mut allocation) = self.allocation.take() {
            unsafe {
                log::debug!("Destroying depth buffer");
                self.device.destroy_image_view(self.view, None);
                self.allocator
                    .vma
                    .destroy_image(self.image, &mut allocation);
            }
        }
    }
}
