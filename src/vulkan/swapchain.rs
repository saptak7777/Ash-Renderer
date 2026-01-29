use crate::vulkan::utils::find_memory_type;
use ash::{khr::swapchain, vk};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::{AshError, Result};

pub struct SwapchainWrapper {
    pub swapchain_loader: Option<swapchain::Device>,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    device: Arc<ash::Device>,
    image_views_managed_by_registry: bool,
    headless: bool,
    headless_image_index: AtomicU32,
    headless_memory: Vec<vk::DeviceMemory>,
}

impl SwapchainWrapper {
    pub fn is_headless(&self) -> bool {
        self.headless
    }
    ///
    /// # Safety
    ///
    /// This function creates Vulkan swapchain and surface. Caller must ensure:
    /// - `vk_device` references a valid initialized Vulkan device
    /// - The window used to create the VulkanInstance remains valid
    /// - Only one swapchain exists per window at a time
    pub unsafe fn new(
        vk_device: &crate::vulkan::VulkanDevice,
        headless: bool,
        preferred_extent: vk::Extent2D,
    ) -> Result<Self> {
        let (swapchain_loader, swapchain, images, image_views, format, extent, headless_memory) =
            if !headless {
                let swapchain_loader =
                    swapchain::Device::new(vk_device.instance.instance(), &vk_device.device);
                let (swapchain, images, image_views, format, extent) =
                    Self::build_swapchain(vk_device, &swapchain_loader, vk::SwapchainKHR::null())?;
                (
                    Some(swapchain_loader),
                    swapchain,
                    images,
                    image_views,
                    format,
                    extent,
                    Vec::new(),
                )
            } else {
                // Headless mode: Create offscreen images
                let extent = preferred_extent;
                let format = vk::Format::R8G8B8A8_UNORM; // Standard format
                let image_count = 3; // Triple buffering simulation

                let mut images = Vec::new();
                let mut image_views = Vec::new();
                let mut memories = Vec::new();

                for _ in 0..image_count {
                    let create_info = vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_SRC,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED);

                    let image = vk_device
                        .device
                        .create_image(&create_info, None)
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to create headless image: {e}"))
                        })?;
                    images.push(image);

                    let mem_req = vk_device.device.get_image_memory_requirements(image);
                    let mem_type_index = find_memory_type(
                        &vk_device.memory_properties,
                        mem_req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .ok_or(AshError::VulkanError(
                        "No suitable memory for headless image".to_string(),
                    ))?;

                    let alloc_info = vk::MemoryAllocateInfo::default()
                        .allocation_size(mem_req.size)
                        .memory_type_index(mem_type_index);

                    let memory = vk_device
                        .device
                        .allocate_memory(&alloc_info, None)
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to allocate headless memory: {e}"
                            ))
                        })?;
                    memories.push(memory);

                    vk_device
                        .device
                        .bind_image_memory(image, memory, 0)
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to bind headless memory: {e}"))
                        })?;

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

                    let view = vk_device
                        .device
                        .create_image_view(&view_info, None)
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to create headless view: {e}"))
                        })?;
                    image_views.push(view);
                }

                log::info!(
                    "Created headless swapchain with {} images ({}x{})",
                    image_count,
                    extent.width,
                    extent.height
                );

                (
                    None,
                    vk::SwapchainKHR::null(),
                    images,
                    image_views,
                    format,
                    extent,
                    memories,
                )
            };

        Ok(Self {
            swapchain_loader,
            swapchain,
            images,
            image_views,
            format,
            extent,
            device: Arc::clone(&vk_device.device),
            image_views_managed_by_registry: false,
            headless,
            headless_image_index: AtomicU32::new(0),
            headless_memory,
        })
    }

    #[allow(clippy::type_complexity)]
    unsafe fn build_swapchain(
        vk_device: &crate::vulkan::VulkanDevice,
        swapchain_loader: &swapchain::Device,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<(
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        vk::Format,
        vk::Extent2D,
    )> {
        let surface_loader = vk_device.instance.surface_loader();
        let surface = vk_device.instance.surface();

        let surface_support = surface_loader
            .get_physical_device_surface_support(
                vk_device.physical_device,
                vk_device.graphics_queue_family,
                surface,
            )
            .map_err(|e| AshError::SwapchainCreationFailed(format!("{e:?}")))?;

        if !surface_support {
            return Err(AshError::SwapchainCreationFailed(
                "Surface not supported by queue family".to_string(),
            ));
        }

        let capabilities = surface_loader
            .get_physical_device_surface_capabilities(vk_device.physical_device, surface)
            .map_err(|e| AshError::SwapchainCreationFailed(format!("{e:?}")))?;

        let formats = surface_loader
            .get_physical_device_surface_formats(vk_device.physical_device, surface)
            .map_err(|e| AshError::SwapchainCreationFailed(format!("{e:?}")))?;

        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
            .format;

        let image_count = if capabilities.max_image_count > 0 {
            capabilities
                .min_image_count
                .max(2)
                .min(capabilities.max_image_count)
        } else {
            capabilities.min_image_count.max(2)
        };

        let extent = capabilities.current_extent;

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .map_err(|e| AshError::SwapchainCreationFailed(format!("{e:?}")))?;

        log::info!("Swapchain created with {image_count} images");

        let images = swapchain_loader
            .get_swapchain_images(swapchain)
            .map_err(|e| AshError::SwapchainCreationFailed(format!("{e:?}")))?;

        let mut image_views = Vec::new();
        for &image in &images {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = vk_device
                .device
                .create_image_view(&create_info, None)
                .map_err(|e| AshError::SwapchainCreationFailed(format!("{e:?}")))?;

            image_views.push(view);
        }

        Ok((swapchain, images, image_views, format, extent))
    }

    /// Recreates the swapchain, typically after window resize.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - All rendering to the old swapchain images has completed
    /// - No frames are in-flight using the old swapchain
    /// - `vk_device` is the same device used to create this swapchain
    pub unsafe fn recreate(
        &mut self,
        vk_device: &crate::vulkan::VulkanDevice,
    ) -> Result<vk::SwapchainKHR> {
        let old_swapchain = self.swapchain;

        if self.headless {
            log::warn!(
                "Recreating headless swapchain not fully implemented - keeping existing images"
            );
            return Ok(vk::SwapchainKHR::null());
        }

        let loader = self.swapchain_loader.as_ref().unwrap();
        let (swapchain, images, image_views, format, extent) =
            Self::build_swapchain(vk_device, loader, self.swapchain)?;

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;
        self.format = format;
        self.extent = extent;

        Ok(old_swapchain)
    }

    /// Acquires the next image from the swapchain for rendering.
    ///
    /// # Safety
    ///
    /// This function acquires a swapchain image. Caller must ensure:
    /// - `semaphore` is a valid Vulkan semaphore
    /// - The semaphore is not currently in use
    /// - The returned image index is used before acquiring the next one
    pub unsafe fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<u32> {
        if self.headless {
            let index = self.headless_image_index.fetch_add(1, Ordering::Relaxed)
                % (self.images.len() as u32);
            return Ok(index);
        }

        match self.swapchain_loader.as_ref().unwrap().acquire_next_image(
            self.swapchain,
            u64::MAX,
            semaphore,
            vk::Fence::null(),
        ) {
            Ok((index, _)) => Ok(index),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => Err(
                AshError::SwapchainOutOfDate("acquire_next_image".to_string()),
            ),
            Err(e) => Err(AshError::FrameAcquisitionFailed(format!("{e:?}"))),
        }
    }

    /// Presents a rendered image to the window.
    ///
    /// # Safety
    ///
    /// This function presents to the swapchain. Caller must ensure:
    /// - `queue` is a valid present queue
    /// - `image_index` was acquired from `acquire_next_image`
    /// - `wait_semaphore` is a valid semaphore that signals render completion
    /// - Rendering to the image is complete before calling present
    pub unsafe fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<()> {
        if self.headless {
            return Ok(());
        }

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match self
            .swapchain_loader
            .as_ref()
            .unwrap()
            .queue_present(queue, &present_info)
        {
            Ok(_) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                Err(AshError::SwapchainOutOfDate("present".to_string()))
            }
            Err(e) => Err(AshError::VulkanError(format!("Present failed: {e:?}"))),
        }
    }

    /// Marks that image views are tracked elsewhere (e.g., ResourceRegistry).
    pub fn mark_image_views_managed_by_registry(&mut self) {
        self.image_views_managed_by_registry = true;
    }

    /// Destroys an old swapchain handle.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The swapchain handle is no longer in use
    /// - All images from this swapchain have been released
    /// - No commands referencing this swapchain are in-flight
    pub unsafe fn destroy_swapchain_handle(&self, handle: vk::SwapchainKHR) {
        if self.headless || handle == vk::SwapchainKHR::null() {
            return;
        }
        self.swapchain_loader
            .as_ref()
            .unwrap()
            .destroy_swapchain(handle, None);
    }
}

impl Drop for SwapchainWrapper {
    fn drop(&mut self) {
        unsafe {
            if !self.headless {
                if let Some(loader) = &self.swapchain_loader {
                    loader.destroy_swapchain(self.swapchain, None);
                }
            } else {
                for &image in &self.images {
                    self.device.destroy_image(image, None);
                }
                for &mem in &self.headless_memory {
                    self.device.free_memory(mem, None);
                }
            }

            if !self.image_views_managed_by_registry {
                for &view in &self.image_views {
                    self.device.destroy_image_view(view, None);
                }
            }
        }
        log::info!("Swapchain destroyed");
    }
}
