use ash::vk;
use std::sync::Arc;

/// Safe image/texture wrapper with automatic cleanup
pub struct ImageHandle {
    image: vk::Image,
    image_view: vk::ImageView,
    device: Arc<ash::Device>,
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
        if let Some(ref n) = name {
            log::info!("Creating image '{n}' ({}x{})", extent.width, extent.height);
        } else {
            log::info!("Creating image ({}x{})", extent.width, extent.height);
        }

        Ok(Self {
            image,
            image_view,
            device,
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
            if self.image != vk::Image::null() {
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
