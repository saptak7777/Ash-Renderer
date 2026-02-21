//! RAII wrappers for Vulkan resources
//!
//! These wrappers automatically clean up Vulkan resources when they go out of scope,
//! preventing resource leaks even on early returns or panics.

use ash::{Device, vk};
use std::ops::Deref;
use std::sync::Arc;

/// RAII wrapper for Vulkan ImageView
///
/// Automatically destroys the image view when dropped, preventing resource leaks.
pub struct ScopedImageView {
    view: vk::ImageView,
    device: Arc<Device>,
}

impl ScopedImageView {
    /// Create a new scoped image view
    ///
    /// # Safety
    /// The device must remain valid for the lifetime of this wrapper.
    pub unsafe fn new(
        device: Arc<Device>,
        create_info: &vk::ImageViewCreateInfo,
    ) -> Result<Self, vk::Result> {
        let view = unsafe { device.create_image_view(create_info, None)? };
        Ok(Self { view, device })
    }

    /// Get the raw Vulkan handle
    pub fn handle(&self) -> vk::ImageView {
        self.view
    }
}

impl Deref for ScopedImageView {
    type Target = vk::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl Drop for ScopedImageView {
    fn drop(&mut self) {
        if self.view != vk::ImageView::null() {
            unsafe {
                self.device.destroy_image_view(self.view, None);
            }
        }
    }
}

/// RAII wrapper for Vulkan Buffer
///
/// Automatically destroys the buffer when dropped.
pub struct ScopedBuffer {
    buffer: vk::Buffer,
    device: Arc<Device>,
}

impl ScopedBuffer {
    /// Create a new scoped buffer
    ///
    /// # Safety
    /// The device must remain valid for the lifetime of this wrapper.
    pub unsafe fn new(
        device: Arc<Device>,
        create_info: &vk::BufferCreateInfo,
    ) -> Result<Self, vk::Result> {
        let buffer = unsafe { device.create_buffer(create_info, None)? };
        Ok(Self { buffer, device })
    }

    /// Get the raw Vulkan handle
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Deref for ScopedBuffer {
    type Target = vk::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl Drop for ScopedBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                self.device.destroy_buffer(self.buffer, None);
            }
        }
    }
}

/// RAII wrapper for Vulkan Image
///
/// Automatically destroys the image when dropped.
pub struct ScopedImage {
    image: vk::Image,
    device: Arc<Device>,
}

impl ScopedImage {
    /// Create a new scoped image
    ///
    /// # Safety
    /// The device must remain valid for the lifetime of this wrapper.
    pub unsafe fn new(
        device: Arc<Device>,
        create_info: &vk::ImageCreateInfo,
    ) -> Result<Self, vk::Result> {
        let image = unsafe { device.create_image(create_info, None)? };
        Ok(Self { image, device })
    }

    /// Get the raw Vulkan handle
    pub fn handle(&self) -> vk::Image {
        self.image
    }
}

impl Deref for ScopedImage {
    type Target = vk::Image;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl Drop for ScopedImage {
    fn drop(&mut self) {
        if self.image != vk::Image::null() {
            unsafe {
                self.device.destroy_image(self.image, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_resource_sizes() {
        // Verify that these are near zero-cost (handle + Arc)
        assert_eq!(std::mem::size_of::<ScopedImageView>(), 16);
        assert_eq!(std::mem::size_of::<ScopedBuffer>(), 16);
        assert_eq!(std::mem::size_of::<ScopedImage>(), 16);
    }
}
