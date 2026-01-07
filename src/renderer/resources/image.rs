use ash::vk;
use std::sync::Arc;

/// Safe image/texture wrapper with automatic cleanup
pub struct ImageHandle {
    image: vk::Image,
    image_view: vk::ImageView,
    device: Arc<ash::Device>,
    allocation: Option<vk_mem::Allocation>,
    allocator: Option<Arc<crate::vulkan::Allocator>>,
    extent: vk::Extent2D,
    format: vk::Format,
    name: Option<String>,
}

impl ImageHandle {
    /// Creates a new image handle.
    ///
    /// # Safety
    ///
    /// The device must remain valid for the lifetime of this handle.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        image: vk::Image,
        image_view: vk::ImageView,
        format: vk::Format,
        extent: vk::Extent2D,
        name: Option<String>,
    ) -> crate::Result<Self> {
        Self::new_with_allocation(device, image, image_view, format, extent, None, None, name)
    }

    /// Creates a new image handle with VMA allocation.
    ///
    /// # Safety
    /// The Vulkan handles must be valid and remain valid for the lifetime of this handle.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_with_allocation(
        device: Arc<ash::Device>,
        image: vk::Image,
        image_view: vk::ImageView,
        format: vk::Format,
        extent: vk::Extent2D,
        allocation: Option<vk_mem::Allocation>,
        allocator: Option<Arc<crate::vulkan::Allocator>>,
        name: Option<String>,
    ) -> crate::Result<Self> {
        if let Some(ref n) = name {
            log::info!("Creating image '{n}' ({}x{})", extent.width, extent.height);
        } else {
            log::info!("Creating image ({}x{})", extent.width, extent.height);
        }

        Ok(Self {
            image,
            image_view,
            device,
            allocation,
            allocator,
            extent,
            format,
            name,
        })
    }

    /// Returns the Vulkan image handle
    pub fn handle(&self) -> vk::Image {
        self.image
    }

    /// Returns the image view
    pub fn view(&self) -> vk::ImageView {
        self.image_view
    }

    /// Returns the image extent (width, height)
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the image format
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Returns the name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Creates a 2D image suitable for a BRDF LUT.
    pub fn create_brdf_lut(
        device: Arc<ash::Device>,
        allocator: Arc<crate::vulkan::Allocator>,
        resolution: u32,
    ) -> crate::Result<Self> {
        let format = vk::Format::R16G16_SFLOAT;
        let extent = vk::Extent2D {
            width: resolution,
            height: resolution,
        };

        let (image, view, allocation) = unsafe {
            allocator.create_image_with_view(
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: resolution,
                        height: resolution,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
                vk::ImageViewType::TYPE_2D,
                vk::ImageAspectFlags::COLOR,
            )?
        };

        unsafe {
            Self::new_with_allocation(
                device,
                image,
                view,
                format,
                extent,
                Some(allocation),
                Some(allocator),
                Some("BRDF_LUT".to_string()),
            )
        }
    }

    /// Creates a cubemap image.
    pub fn create_cubemap(
        device: Arc<ash::Device>,
        allocator: Arc<crate::vulkan::Allocator>,
        resolution: u32,
        mip_levels: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<String>,
    ) -> crate::Result<Self> {
        let extent = vk::Extent2D {
            width: resolution,
            height: resolution,
        };

        let (image, view, allocation) = unsafe {
            allocator.create_image_with_view(
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: resolution,
                        height: resolution,
                        depth: 1,
                    })
                    .mip_levels(mip_levels)
                    .array_layers(6)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
                vk::ImageViewType::CUBE,
                vk::ImageAspectFlags::COLOR,
            )?
        };

        unsafe {
            Self::new_with_allocation(
                device,
                image,
                view,
                format,
                extent,
                Some(allocation),
                Some(allocator),
                name,
            )
        }
    }
}

impl Drop for ImageHandle {
    fn drop(&mut self) {
        unsafe {
            if let Some(ref name) = self.name {
                log::debug!("Destroying image '{name}'");
            }
            if self.image_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.image_view, None);
            }
            if let (Some(mut allocation), Some(allocator)) =
                (self.allocation.take(), &self.allocator)
            {
                allocator.vma.destroy_image(self.image, &mut allocation);
            } else if self.image != vk::Image::null() {
                // Fallback for non-VMA images (e.g. swapchain)
                self.device.destroy_image(self.image, None);
            }
        }
    }
}

impl std::fmt::Debug for ImageHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageHandle")
            .field("image", &self.image)
            .field("image_view", &self.image_view)
            .field("extent", &self.extent)
            .field("format", &self.format)
            .field("name", &self.name)
            .finish()
    }
}
