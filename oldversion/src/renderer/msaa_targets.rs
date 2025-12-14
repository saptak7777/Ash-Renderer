//! MSAA (Multi-Sample Anti-Aliasing) render targets
//!
//! Provides multi-sampled color and depth images that resolve to single-sample
//! framebuffers for anti-aliased rendering.

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// Multi-sample color render target
pub struct MsaaColorTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<vk_mem::Allocation>,
    allocator: Arc<Allocator>,
    device: Arc<ash::Device>,
    format: vk::Format,
    sample_count: vk::SampleCountFlags,
    extent: vk::Extent2D,
}

impl MsaaColorTarget {
    /// Creates a new MSAA color target
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this target.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
        format: vk::Format,
        sample_count: vk::SampleCountFlags,
    ) -> Result<Self> {
        log::info!("Creating MSAA color target ({width}x{height}, samples: {sample_count:?})");

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
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            allocator.create_image(&image_create_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

        let view_create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = device
            .create_image_view(&view_create_info, None)
            .map_err(|e| AshError::VulkanError(format!("MSAA color view creation failed: {e}")))?;

        log::info!("MSAA color target created successfully");

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            allocator,
            device,
            format,
            sample_count,
            extent: vk::Extent2D { width, height },
        })
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn sample_count(&self) -> vk::SampleCountFlags {
        self.sample_count
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for MsaaColorTarget {
    fn drop(&mut self) {
        if let Some(mut allocation) = self.allocation.take() {
            unsafe {
                log::debug!("Destroying MSAA color target");
                self.device.destroy_image_view(self.view, None);
                self.allocator
                    .vma
                    .destroy_image(self.image, &mut allocation);
            }
        }
    }
}

/// Multi-sample depth render target
pub struct MsaaDepthTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<vk_mem::Allocation>,
    allocator: Arc<Allocator>,
    device: Arc<ash::Device>,
    format: vk::Format,
    sample_count: vk::SampleCountFlags,
}

impl MsaaDepthTarget {
    /// Creates a new MSAA depth target
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this target.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
        sample_count: vk::SampleCountFlags,
    ) -> Result<Self> {
        let format = vk::Format::D32_SFLOAT;

        log::info!("Creating MSAA depth target ({width}x{height}, samples: {sample_count:?})");

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
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            allocator.create_image(&image_create_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

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

        let view = device
            .create_image_view(&view_create_info, None)
            .map_err(|e| AshError::VulkanError(format!("MSAA depth view creation failed: {e}")))?;

        log::info!("MSAA depth target created successfully");

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            allocator,
            device,
            format,
            sample_count,
        })
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn sample_count(&self) -> vk::SampleCountFlags {
        self.sample_count
    }
}

impl Drop for MsaaDepthTarget {
    fn drop(&mut self) {
        if let Some(mut allocation) = self.allocation.take() {
            unsafe {
                log::debug!("Destroying MSAA depth target");
                self.device.destroy_image_view(self.view, None);
                self.allocator
                    .vma
                    .destroy_image(self.image, &mut allocation);
            }
        }
    }
}

/// Helper to check if a sample count is supported by the device
pub fn get_max_usable_sample_count(
    physical_device_properties: &vk::PhysicalDeviceProperties,
) -> vk::SampleCountFlags {
    let counts = physical_device_properties
        .limits
        .framebuffer_color_sample_counts
        & physical_device_properties
            .limits
            .framebuffer_depth_sample_counts;

    if counts.contains(vk::SampleCountFlags::TYPE_8) {
        vk::SampleCountFlags::TYPE_8
    } else if counts.contains(vk::SampleCountFlags::TYPE_4) {
        vk::SampleCountFlags::TYPE_4
    } else if counts.contains(vk::SampleCountFlags::TYPE_2) {
        vk::SampleCountFlags::TYPE_2
    } else {
        vk::SampleCountFlags::TYPE_1
    }
}

/// Clamp requested sample count to device-supported maximum
pub fn clamp_sample_count(
    requested: vk::SampleCountFlags,
    physical_device_properties: &vk::PhysicalDeviceProperties,
) -> vk::SampleCountFlags {
    let max = get_max_usable_sample_count(physical_device_properties);

    // Compare sample counts numerically
    let requested_val = match requested {
        vk::SampleCountFlags::TYPE_8 => 8,
        vk::SampleCountFlags::TYPE_4 => 4,
        vk::SampleCountFlags::TYPE_2 => 2,
        _ => 1,
    };

    let max_val = match max {
        vk::SampleCountFlags::TYPE_8 => 8,
        vk::SampleCountFlags::TYPE_4 => 4,
        vk::SampleCountFlags::TYPE_2 => 2,
        _ => 1,
    };

    if requested_val <= max_val {
        requested
    } else {
        log::warn!("Requested MSAA {requested:?} exceeds device max {max:?}, clamping");
        max
    }
}
