use crate::renderer::resources::ImageHandle;
use crate::vulkan::Allocator;
use ash::vk;
use std::sync::Arc;

/// Strongly-typed GPU texture with automatic cleanup via RAII.
///
/// This is a lightweight wrapper around `ImageHandle` that provides
/// additional type safety and a simplified API for common texture operations.
///
/// # Example
/// ```no_run
/// # use ash_renderer::renderer::resources::GpuTexture;
/// let texture = unsafe {
///     GpuTexture::new_2d(
///         device,
///         allocator,
///         1024,
///         1024,
///         vk::Format::R8G8B8A8_SRGB,
///         vk::ImageUsageFlags::SAMPLED,
///         Some("AlbedoMap".to_string()),
///     )?
/// };
/// ```
pub struct GpuTexture {
    inner: ImageHandle,
}

impl GpuTexture {
    /// Creates a new 2D texture.
    ///
    /// # Safety
    ///
    /// The device and allocator must remain valid for the lifetime of this texture.
    pub unsafe fn new_2d(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<String>,
    ) -> crate::Result<Self> {
        Self::new_2d_with_mips(device, allocator, width, height, 1, format, usage, name)
    }

    /// Creates a new 2D texture with mip levels.
    ///
    /// # Safety
    ///
    /// The device and allocator must remain valid for the lifetime of this texture.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_2d_with_mips(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<String>,
    ) -> crate::Result<Self> {
        let extent = vk::Extent2D { width, height };

        let (image, view, allocation) = allocator.create_image_with_view(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(mip_levels)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            },
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
        )?;

        let inner = ImageHandle::new_with_allocation(
            device,
            image,
            view,
            format,
            extent,
            mip_levels,
            1,
            Some(allocation),
            Some(allocator),
            name,
        )?;

        Ok(Self { inner })
    }

    /// Creates a cubemap texture.
    ///
    /// # Safety
    ///
    /// The device and allocator must remain valid for the lifetime of this texture.
    pub unsafe fn new_cubemap(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        resolution: u32,
        mip_levels: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<String>,
    ) -> crate::Result<Self> {
        let inner = ImageHandle::create_cubemap(
            device, allocator, resolution, mip_levels, format, usage, name,
        )?;

        Ok(Self { inner })
    }

    /// Returns the Vulkan image handle
    pub fn image(&self) -> vk::Image {
        self.inner.handle()
    }

    /// Returns the image view
    pub fn view(&self) -> vk::ImageView {
        self.inner.view()
    }

    /// Returns the image extent (width, height)
    pub fn extent(&self) -> vk::Extent2D {
        self.inner.extent()
    }

    /// Returns the image format
    pub fn format(&self) -> vk::Format {
        self.inner.format()
    }

    /// Returns the number of mip levels
    pub fn mip_levels(&self) -> u32 {
        self.inner.mip_levels()
    }

    /// Returns the underlying ImageHandle (for compatibility)
    pub fn inner(&self) -> &ImageHandle {
        &self.inner
    }

    /// Consumes this wrapper and returns the underlying ImageHandle
    pub fn into_inner(self) -> ImageHandle {
        self.inner
    }
}

impl std::fmt::Debug for GpuTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTexture")
            .field("inner", &self.inner)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_type_safety() {
        // This test verifies that the type system prevents misuse at compile time
        fn _accepts_texture(_tex: &GpuTexture) {}

        // GpuTexture provides a simplified, safe API over ImageHandle
        // The actual creation requires Vulkan context, so we just verify the API exists
    }
}
