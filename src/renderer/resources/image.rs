use ash::vk;
use std::sync::Arc;

/// Parameters for creating an image handle.
pub struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub layers: u32,
    pub name: Option<String>,
}

/// Safe image/texture wrapper with automatic cleanup
pub struct ImageHandle {
    image: vk::Image,
    image_view: vk::ImageView,
    device: Arc<ash::Device>,
    allocation: Option<vk_mem::Allocation>,
    allocator: Option<Arc<crate::vulkan::Allocator>>,
    extent: vk::Extent2D,
    format: vk::Format,
    mip_levels: u32,
    layers: u32,
    name: Option<String>,
}

impl ImageHandle {
    /// Creates a new image handle.
    ///
    /// # Safety
    /// The Vulkan device must remain valid for the lifetime of this handle. The provided image and view must be valid handles.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        image: vk::Image,
        image_view: vk::ImageView,
        info: ImageCreateInfo,
    ) -> crate::Result<Self> {
        unsafe { Self::new_with_allocation(device, image, image_view, None, None, info) }
    }

    /// Creates a new image handle with VMA allocation.
    ///
    /// # Safety
    /// All provided Vulkan handles (device, image, view, allocator) must be valid and remain active. The allocation must correspond to the provided image.
    pub unsafe fn new_with_allocation(
        device: Arc<ash::Device>,
        image: vk::Image,
        image_view: vk::ImageView,
        allocation: Option<vk_mem::Allocation>,
        allocator: Option<Arc<crate::vulkan::Allocator>>,
        info: ImageCreateInfo,
    ) -> crate::Result<Self> {
        if let Some(ref n) = info.name {
            log::info!(
                "Creating image '{n}' ({width}x{height})",
                width = info.width,
                height = info.height
            );
        } else {
            log::info!(
                "Creating image ({width}x{height})",
                width = info.width,
                height = info.height
            );
        }

        Ok(Self {
            image,
            image_view,
            device,
            allocation,
            allocator,
            extent: vk::Extent2D {
                width: info.width,
                height: info.height,
            },
            format: info.format,
            mip_levels: info.mip_levels,
            layers: info.layers,
            name: info.name,
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

    /// Returns the number of mip levels
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Returns the number of array layers
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Returns the allocator if set
    pub fn allocator(&self) -> Option<Arc<crate::vulkan::Allocator>> {
        self.allocator.as_ref().map(Arc::clone)
    }

    /// Returns the device reference
    pub fn device(&self) -> Arc<ash::Device> {
        Arc::clone(&self.device)
    }

    /// Reads the image data back to a CPU-accessible buffer.
    pub fn read_to_buffer(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> crate::Result<Vec<u8>> {
        let (allocator, name) = match (&self.allocator, &self.name) {
            (Some(a), Some(n)) => (a, n.as_str()),
            (Some(a), None) => (a, "unnamed_image"),
            _ => {
                return Err(crate::AshError::VulkanError(
                    "Cannot read back image without allocator".to_string(),
                ));
            }
        };

        let format_size = match self.format {
            vk::Format::R32G32B32A32_SFLOAT => 16,
            vk::Format::R16G16B16A16_SFLOAT => 8,
            vk::Format::R16G16_SFLOAT => 4,
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => 4,
            _ => {
                return Err(crate::AshError::VulkanError(format!(
                    "Unsupported readback format: {:?}",
                    self.format
                )));
            }
        };

        // Calculate total size across all mips and layers
        let mut total_size = 0;
        for mip in 0..self.mip_levels {
            let mip_w = (self.extent.width >> mip).max(1);
            let mip_h = (self.extent.height >> mip).max(1);
            total_size += mip_w * mip_h * format_size * self.layers;
        }

        let (readback_buffer, mut readback_alloc) =
            crate::vulkan::buffer_builder::BufferBuilder::new(total_size as u64)
                .transfer_dst()
                .cpu_readable()
                .named("Readback Buffer")
                .build(allocator)?;

        crate::vulkan::utils::execute_single_use(
            self.device.as_ref(),
            command_pool,
            queue,
            |cmd| {
                // Transition to TRANSFER_SRC_OPTIMAL (Sync2)
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: self.mip_levels,
                        base_array_layer: 0,
                        layer_count: self.layers,
                    });

                let image_barriers = [barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
                unsafe {
                    self.device.cmd_pipeline_barrier2(cmd, &dep_info);
                }
                let mut buffer_offset = 0;
                for mip in 0..self.mip_levels {
                    let mip_w = (self.extent.width >> mip).max(1);
                    let mip_h = (self.extent.height >> mip).max(1);

                    let region = vk::BufferImageCopy {
                        buffer_offset,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: mip,
                            base_array_layer: 0,
                            layer_count: self.layers,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: mip_w,
                            height: mip_h,
                            depth: 1,
                        },
                    };

                    unsafe {
                        self.device.cmd_copy_image_to_buffer(
                            cmd,
                            self.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            readback_buffer,
                            &[region],
                        );
                    }

                    buffer_offset +=
                        mip_w as u64 * mip_h as u64 * format_size as u64 * self.layers as u64;
                }

                // Transition back to SHADER_READ_ONLY_OPTIMAL (Sync2)
                let barrier_restore = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: self.mip_levels,
                        base_array_layer: 0,
                        layer_count: self.layers,
                    });

                let image_barriers_restore = [barrier_restore];
                let dep_info_restore =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers_restore);
                unsafe {
                    self.device.cmd_pipeline_barrier2(cmd, &dep_info_restore);
                }
            },
        )?;

        let data = unsafe {
            let guard = allocator.map_allocation_guarded(&mut readback_alloc, total_size as u64)?;
            guard.to_vec()
        };

        unsafe {
            allocator
                .vma
                .destroy_buffer(readback_buffer, &mut readback_alloc);
        }

        log::info!("Read back image '{name}' ({} bytes)", data.len());
        Ok(data)
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
                Some(allocation),
                Some(allocator),
                ImageCreateInfo {
                    width: extent.width,
                    height: extent.height,
                    format,
                    mip_levels: 1,
                    layers: 1,
                    name: Some("BRDF_LUT".to_string()),
                },
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
                Some(allocation),
                Some(allocator),
                ImageCreateInfo {
                    width: extent.width,
                    height: extent.height,
                    format,
                    mip_levels,
                    layers: 6,
                    name,
                },
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
