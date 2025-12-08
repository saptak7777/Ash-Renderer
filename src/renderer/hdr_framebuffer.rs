//! HDR (High Dynamic Range) Framebuffer
//!
//! Provides a 16-bit float framebuffer for HDR rendering before tonemapping.

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// HDR render target for pre-tonemapping scene rendering
pub struct HdrFramebuffer {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<vk_mem::Allocation>,
    allocator: Arc<Allocator>,
    device: Arc<ash::Device>,
    format: vk::Format,
    extent: vk::Extent2D,
    sampler: vk::Sampler,
}

impl HdrFramebuffer {
    /// Creates a new HDR framebuffer with 16-bit float format
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this framebuffer.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let format = vk::Format::R16G16B16A16_SFLOAT;

        log::info!("Creating HDR framebuffer ({width}x{height}, R16G16B16A16_SFLOAT)");

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
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
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
            .map_err(|e| AshError::VulkanError(format!("HDR view creation failed: {e}")))?;

        // Create sampler for reading HDR buffer in post-processing
        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        let sampler = device
            .create_sampler(&sampler_create_info, None)
            .map_err(|e| AshError::VulkanError(format!("HDR sampler creation failed: {e}")))?;

        log::info!("HDR framebuffer created successfully");

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            allocator,
            device,
            format,
            extent: vk::Extent2D { width, height },
            sampler,
        })
    }

    /// Returns the HDR image view
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// Returns the HDR image
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Returns the HDR format
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Returns the extent
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Returns the sampler for reading the HDR buffer
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Returns descriptor image info for binding to post-process shader
    pub fn descriptor_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: self.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}

impl Drop for HdrFramebuffer {
    fn drop(&mut self) {
        if let Some(mut allocation) = self.allocation.take() {
            unsafe {
                log::debug!("Destroying HDR framebuffer");
                self.device.destroy_sampler(self.sampler, None);
                self.device.destroy_image_view(self.view, None);
                self.allocator
                    .vma
                    .destroy_image(self.image, &mut allocation);
            }
        }
    }
}
