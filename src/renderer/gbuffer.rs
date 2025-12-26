//! G-Buffer (Geometry Buffer)
//!
//! Stores surface properties like Normals and Albedo in separate textures
//! for use in screen-space post-processing and indirect lighting.

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// G-Buffer textures for deferred or hybrid rendering
pub struct GBuffer {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,

    // Normal buffer (R16G16B16A16_SFLOAT for high precision normals)
    normal_image: vk::Image,
    normal_view: vk::ImageView,
    normal_allocation: Option<vk_mem::Allocation>,

    // Albedo buffer (R8G8B8A8_UNORM)
    albedo_image: vk::Image,
    albedo_view: vk::ImageView,
    albedo_allocation: Option<vk_mem::Allocation>,

    // Motion buffer (R16G16_SFLOAT)
    motion_image: vk::Image,
    motion_view: vk::ImageView,
    motion_allocation: Option<vk_mem::Allocation>,

    extent: vk::Extent2D,
}

impl GBuffer {
    /// Create a new G-Buffer with the specified dimensions
    ///
    /// # Safety
    /// This function creates Vulkan resources. The caller must ensure that the
    /// device and allocator are valid and that the G-Buffer is destroyed before
    /// they are dropped.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width, height };

        // 1. Create Normal Buffer
        let normal_format = vk::Format::R16G16B16A16_SFLOAT;
        let (normal_image, normal_allocation) = Self::create_image(
            &allocator,
            width,
            height,
            normal_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?;

        let normal_view = Self::create_view(&device, normal_image, normal_format)?;

        // 2. Create Albedo Buffer
        let albedo_format = vk::Format::R8G8B8A8_UNORM;
        let (albedo_image, albedo_allocation) = Self::create_image(
            &allocator,
            width,
            height,
            albedo_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?;

        let albedo_view = Self::create_view(&device, albedo_image, albedo_format)?;

        // 3. Create Motion Buffer
        let motion_format = vk::Format::R16G16_SFLOAT;
        let (motion_image, motion_allocation) = Self::create_image(
            &allocator,
            width,
            height,
            motion_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?;

        let motion_view = Self::create_view(&device, motion_image, motion_format)?;

        Ok(Self {
            device,
            allocator,
            normal_image,
            normal_view,
            normal_allocation: Some(normal_allocation),
            albedo_image,
            albedo_view,
            albedo_allocation: Some(albedo_allocation),
            motion_image,
            motion_view,
            motion_allocation: Some(motion_allocation),
            extent,
        })
    }

    unsafe fn create_image(
        allocator: &Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk_mem::Allocation)> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        allocator
            .create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)
            .map_err(|e| AshError::VulkanError(format!("G-Buffer image creation failed: {e:?}")))
    }

    unsafe fn create_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> Result<vk::ImageView> {
        let view_info = vk::ImageViewCreateInfo::default()
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

        device
            .create_image_view(&view_info, None)
            .map_err(|e| AshError::VulkanError(format!("G-Buffer view creation failed: {e:?}")))
    }

    pub fn normal_view(&self) -> vk::ImageView {
        self.normal_view
    }

    pub fn albedo_view(&self) -> vk::ImageView {
        self.albedo_view
    }

    pub fn motion_view(&self) -> vk::ImageView {
        self.motion_view
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for GBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.normal_view, None);
            self.device.destroy_image_view(self.albedo_view, None);
            self.device.destroy_image_view(self.motion_view, None);

            if let Some(mut alloc) = self.normal_allocation.take() {
                self.allocator
                    .vma
                    .destroy_image(self.normal_image, &mut alloc);
            }
            if let Some(mut alloc) = self.albedo_allocation.take() {
                self.allocator
                    .vma
                    .destroy_image(self.albedo_image, &mut alloc);
            }
            if let Some(mut alloc) = self.motion_allocation.take() {
                self.allocator
                    .vma
                    .destroy_image(self.motion_image, &mut alloc);
            }
        }
    }
}
