use ash::vk;
use std::sync::Arc;

use crate::{vulkan, AshError, Result};

/// CPU-side texture data ready for GPU upload (RGBA8)
#[derive(Clone, Debug)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl TextureData {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        let expected = width as usize * height as usize * 4;
        if pixels.len() != expected {
            return Err(AshError::VulkanError(format!(
                "Texture pixel data size mismatch: expected {expected} bytes, got {}",
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn solid_color(color: [u8; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            pixels: Vec::from(color),
        }
    }

    pub fn solid_color_uint(color: [u8; 4]) -> Self {
        // For R32_UINT format, we need 4 bytes per pixel
        Self {
            width: 1,
            height: 1,
            pixels: Vec::from(color),
        }
    }
}

/// GPU texture with image, view, and sampler
pub struct Texture {
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: vk_mem::Allocation,
    allocator: Arc<vulkan::Allocator>,
    device: Arc<ash::Device>,
}

impl Texture {
    /// # Safety
    /// Caller must ensure the provided Vulkan handles remain valid for the lifetime of the texture.
    pub unsafe fn from_data(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        data: &TextureData,
        format: vk::Format,
        name: Option<&str>,
    ) -> Result<Self> {
        let image_size = data.pixels.len() as vk::DeviceSize;
        if image_size == 0 {
            return Err(crate::AshError::VulkanError(
                "Cannot create texture from empty pixel data".to_string(),
            ));
        }

        let is_compressed = matches!(
            format,
            vk::Format::BC7_UNORM_BLOCK
                | vk::Format::BC7_SRGB_BLOCK
                | vk::Format::BC5_UNORM_BLOCK
                | vk::Format::BC5_SNORM_BLOCK
        );

        let mip_levels = if is_compressed {
            1
        } else {
            (data.width.max(data.height) as f32).log2().floor() as u32 + 1
        };

        // Staging buffer
        let (staging_buffer, mut staging_alloc) = allocator.create_buffer_with_flags(
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )?;

        {
            let mut guard =
                unsafe { allocator.map_allocation_guarded(&mut staging_alloc, image_size)? };
            guard.copy_from_slice(&data.pixels);
        }

        allocator
            .vma
            .flush_allocation(&staging_alloc, 0, image_size)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to flush texture staging buffer: {e}"))
            })?;
        // unmap handled by guard drop

        // Create image with mipmaps and proper usage
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: data.width,
                height: data.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            allocator.create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

        // Execute upload and mipmap generation
        vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
            // Transition Mip 0 to TRANSFER_DST_OPTIMAL
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels, // Transition all initially
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: data.width,
                    height: data.height,
                    depth: 1,
                },
            };

            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Generate Mipmaps
            let mut mip_width = data.width as i32;
            let mut mip_height = data.height as i32;

            for i in 1..mip_levels {
                let next_width = if mip_width > 1 { mip_width / 2 } else { 1 };
                let next_height = if mip_height > 1 { mip_height / 2 } else { 1 };

                // Transition i-1 to TRANSFER_SRC
                let barrier_src = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_src],
                );

                let blit = vk::ImageBlit {
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ],
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: next_width,
                            y: next_height,
                            z: 1,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };

                device.cmd_blit_image(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );

                // Transition i-1 to SHADER_READ_ONLY
                let barrier_done = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_done],
                );

                mip_width = next_width;
                mip_height = next_height;
            }

            // Transition last mip
            let barrier_last = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip_levels - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_last],
            );
        })?;

        // Cleanup staging buffer
        allocator
            .vma
            .destroy_buffer(staging_buffer, &mut staging_alloc);

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = device.create_image_view(&view_info, None)?;

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true) // Enable Anisotropy
            .max_anisotropy(16.0) // High Quality
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(mip_levels as f32); // Full Range

        let sampler = device.create_sampler(&sampler_info, None)?;

        if let Some(label) = name {
            log::info!(
                "Created texture '{label}' ({}x{}, {} mips)",
                data.width,
                data.height,
                mip_levels
            );
        } else {
            log::info!(
                "Created texture ({}x{}, {} mips)",
                data.width,
                data.height,
                mip_levels
            );
        }

        Ok(Self {
            image,
            view: image_view,
            sampler,
            allocation,
            allocator,
            device,
        })
    }

    /// Creates a 2D texture from raw bytes and format.
    ///
    /// # Safety
    /// Caller must ensure Vulkan handles are valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_raw_data(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        raw_data: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        mip_levels: u32,
        name: Option<&str>,
    ) -> Result<Self> {
        let image_size = raw_data.len() as vk::DeviceSize;

        let (staging_buffer, mut staging_alloc) = allocator.create_buffer_with_flags(
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )?;

        unsafe {
            let mut guard = allocator.map_allocation_guarded(&mut staging_alloc, image_size)?;
            guard.copy_from_slice(raw_data);
        }

        allocator
            .vma
            .flush_allocation(&staging_alloc, 0, image_size)
            .map_err(|e| AshError::VulkanError(format!("Flush failed: {e}")))?;

        // Image creation
        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            allocator.create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

        // Single move
        vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
            let barrier_start = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_start],
            );

            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };

            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            let barrier_end = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_end],
            );
        })?;

        allocator
            .vma
            .destroy_buffer(staging_buffer, &mut staging_alloc);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = device.create_image_view(&view_info, None)?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);

        let sampler = device.create_sampler(&sampler_info, None)?;

        if let Some(label) = name {
            log::info!("Created raw texture '{label}' ({width}x{height})");
        }

        Ok(Self {
            image,
            view,
            sampler,
            allocation,
            allocator,
            device,
        })
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Creates a texture from a list of pre-generated mip levels.
    ///
    /// Useful for compressed textures where GPU-side mipmap generation (blitting) is not supported.
    ///
    /// # Safety
    /// Caller must ensure Vulkan handles are valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_mips(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mips: &[Vec<u8>],
        width: u32,
        height: u32,
        format: vk::Format,
        name: Option<&str>,
    ) -> Result<Self> {
        if mips.is_empty() {
            return Err(crate::AshError::VulkanError(
                "Cannot create texture from empty mip chain".to_string(),
            ));
        }

        let total_size: usize = mips.iter().map(|mip| mip.len()).sum();
        let mip_levels = mips.len() as u32;

        // Staging buffer
        let (staging_buffer, mut staging_alloc) = allocator.create_buffer_with_flags(
            total_size as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )?;

        {
            let mut guard = unsafe {
                allocator
                    .map_allocation_guarded(&mut staging_alloc, total_size as vk::DeviceSize)?
            };

            let mut offset = 0;
            for mip in mips {
                guard[offset..offset + mip.len()].copy_from_slice(mip);
                offset += mip.len();
            }
        }

        allocator
            .vma
            .flush_allocation(&staging_alloc, 0, total_size as vk::DeviceSize)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to flush texture staging buffer: {e}"))
            })?;

        // Create image
        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            allocator.create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

        // Upload
        vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
            // Transition all mips to TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy regions
            let mut regions = Vec::with_capacity(mip_levels as usize);
            let mut buffer_offset = 0;
            let mut mip_width = width;
            let mut mip_height = height;

            for i in 0..mip_levels {
                regions.push(vk::BufferImageCopy {
                    buffer_offset,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: mip_width,
                        height: mip_height,
                        depth: 1,
                    },
                });

                buffer_offset += mips[i as usize].len() as vk::DeviceSize;

                if mip_width > 1 {
                    mip_width /= 2;
                }
                if mip_height > 1 {
                    mip_height /= 2;
                }
            }

            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );

            // Transition to SHADER_READ_ONLY
            let barrier_done = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_done],
            );
        })?;

        // Cleanup staging
        allocator
            .vma
            .destroy_buffer(staging_buffer, &mut staging_alloc);

        // View & Sampler
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_view = device.create_image_view(&view_info, None)?;

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);

        let sampler = device.create_sampler(&sampler_info, None)?;

        if let Some(label) = name {
            log::info!(
                "Created texture '{label}' ({width}x{height}, {mip_levels} mips, compressed)"
            );
        } else {
            log::info!("Created texture ({width}x{height}, {mip_levels} mips, compressed)");
        }

        Ok(Self {
            image,
            view: image_view,
            sampler,
            allocation,
            allocator,
            device,
        })
    }
    pub fn create_default_white(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let white = TextureData::solid_color([255, 255, 255, 255]);
        unsafe {
            Self::from_data(
                allocator,
                device,
                command_pool,
                queue,
                &white,
                vk::Format::R8G8B8A8_UNORM,
                Some("DefaultWhite"),
            )
        }
    }

    pub fn create_default_black(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let black = TextureData::solid_color([0, 0, 0, 255]);
        unsafe {
            Self::from_data(
                allocator,
                device,
                command_pool,
                queue,
                &black,
                vk::Format::R8G8B8A8_UNORM,
                Some("DefaultBlack"),
            )
        }
    }

    pub fn create_default_cube_black(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let resolution = 1;
        let format = vk::Format::R8G8B8A8_UNORM;
        let pixel_size = 4;
        let total_size = resolution * resolution * pixel_size * 6; // 6 faces

        // Create staging buffer (all zeros for black)
        // Create staging buffer (all zeros for black)
        let (staging_buffer, mut staging_alloc) = unsafe {
            allocator.create_buffer_with_flags(
                total_size as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            )?
        };

        {
            let mut guard = unsafe {
                allocator
                    .map_allocation_guarded(&mut staging_alloc, total_size as vk::DeviceSize)?
            };
            // Dark grey initialize (provides subtle ambient fallback)
            for i in 0..6 {
                let offset = i * 4;
                guard[offset] = 30; // R
                guard[offset + 1] = 30; // G
                guard[offset + 2] = 30; // B
                guard[offset + 3] = 255; // A
            }
        }

        allocator
            .vma
            .flush_allocation(&staging_alloc, 0, total_size as vk::DeviceSize)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to flush default cube staging: {e}"))
            })?;

        // Create Cubemap Image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(6)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            unsafe { allocator.create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)? };

        // Upload
        vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
            // Transition to TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });

            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            // Copy 6 faces
            let mut regions = Vec::with_capacity(6);
            for i in 0..6 {
                regions.push(vk::BufferImageCopy {
                    buffer_offset: (i * 4) as vk::DeviceSize,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: i as u32,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: resolution,
                        height: resolution,
                        depth: 1,
                    },
                });
            }

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
            }

            // Transition to SHADER_READ_ONLY
            let barrier_end = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });

            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_end],
                );
            }
        })?;

        // Cleanup Staging
        unsafe {
            allocator
                .vma
                .destroy_buffer(staging_buffer, &mut staging_alloc);
        }

        // View
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });

        let view = unsafe { device.create_image_view(&view_info, None)? };

        // Sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(1.0); // 1 Mip

        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        log::info!("Created default black cubemap (1x1)");

        Ok(Self {
            image,
            view,
            sampler,
            allocation,
            allocator,
            device,
        })
    }

    /// Creates a 1x1 R32_UINT texture with NEAREST filtering for VSM page table default
    /// CRITICAL: Integer textures (usampler2D) MUST use NEAREST filtering, not LINEAR
    pub fn create_vsm_default_uint(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        let data = TextureData::solid_color_uint([0xFF, 0xFF, 0xFF, 0xFF]);

        unsafe {
            let image_size = data.pixels.len() as vk::DeviceSize;

            // Staging buffer
            let (staging_buffer, mut staging_alloc) = allocator.create_buffer_with_flags(
                image_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            )?;

            {
                let mut guard = allocator.map_allocation_guarded(&mut staging_alloc, image_size)?;
                guard.copy_from_slice(&data.pixels);
            }

            allocator
                .vma
                .flush_allocation(&staging_alloc, 0, image_size)
                .map_err(|e| AshError::VulkanError(format!("Failed to flush: {e}")))?;

            // Create R32_UINT image (no mipmaps for integer formats)
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R32_UINT)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let (image, allocation) =
                allocator.create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)?;

            // Upload
            vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );

                let region = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                };

                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );

                let barrier_final = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_final],
                );
            })?;

            allocator
                .vma
                .destroy_buffer(staging_buffer, &mut staging_alloc);

            // Create image view
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_UINT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = device.create_image_view(&view_info, None)?;

            // CRITICAL: Create NEAREST sampler for integer textures
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .anisotropy_enable(false)
                .min_lod(0.0)
                .max_lod(0.0);

            let sampler = device.create_sampler(&sampler_info, None)?;

            log::info!("Created VSM default UINT texture (R32_UINT, NEAREST filtering)");

            Ok(Self {
                image,
                view,
                sampler,
                allocation,
                allocator,
                device,
            })
        }
    }

    pub fn create_procedural_skybox(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        resolution: u32,
    ) -> Result<Self> {
        let format = vk::Format::R8G8B8A8_UNORM;
        let pixel_size = 4;
        let total_size = (resolution * resolution * pixel_size * 6) as u64;

        // Create staging buffer
        let (staging_buffer, mut staging_alloc) = unsafe {
            allocator.create_buffer_with_flags(
                total_size as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            )?
        };

        {
            let mut guard = unsafe {
                allocator
                    .map_allocation_guarded(&mut staging_alloc, total_size as vk::DeviceSize)?
            };

            // Colors for gradient
            let zenith_color = glam::Vec3::new(0.05, 0.05, 0.1); // Deep Space Blue (Top)
            let horizon_color = glam::Vec3::new(0.3, 0.4, 0.6); // Desaturated Blue (Horizon)
            let nadir_color = glam::Vec3::new(0.02, 0.02, 0.03); // Very Dark Blue (Bottom)

            // Iterate over 6 faces of the cubemap
            for face in 0..6 {
                for y in 0..resolution {
                    for x in 0..resolution {
                        // Calculate UV coordinates in [-1, 1] range
                        let u = (x as f32 / resolution as f32) * 2.0 - 1.0;
                        let v = (y as f32 / resolution as f32) * 2.0 - 1.0;

                        // Calculate direction vector based on face index
                        let dir = match face {
                            0 => glam::Vec3::new(1.0, -v, -u),  // +X
                            1 => glam::Vec3::new(-1.0, -v, u),  // -X
                            2 => glam::Vec3::new(u, 1.0, v),    // +Y (Top)
                            3 => glam::Vec3::new(u, -1.0, -v),  // -Y (Bottom)
                            4 => glam::Vec3::new(u, -v, 1.0),   // +Z
                            5 => glam::Vec3::new(-u, -v, -1.0), // -Z
                            _ => glam::Vec3::ZERO,
                        }
                        .normalize();

                        // Calculate gradient color based on Y component
                        let color = if dir.y > 0.0 {
                            zenith_color.lerp(horizon_color, 1.0 - dir.y.powf(0.5))
                        } else {
                            horizon_color.lerp(nadir_color, (-dir.y).powf(0.5))
                        };

                        // Gamma correct (approximate linear to sRGB conversion for storage)
                        let srgb = color.powf(1.0 / 2.2);

                        let r = (srgb.x * 255.0).clamp(0.0, 255.0) as u8;
                        let g = (srgb.y * 255.0).clamp(0.0, 255.0) as u8;
                        let b = (srgb.z * 255.0).clamp(0.0, 255.0) as u8;

                        let pixel_index = ((face as u64 * resolution as u64 * resolution as u64)
                            + (y as u64 * resolution as u64)
                            + x as u64) as usize;
                        let offset = pixel_index * 4;

                        guard[offset] = r;
                        guard[offset + 1] = g;
                        guard[offset + 2] = b;
                        guard[offset + 3] = 255;
                    }
                }
            }
        }

        allocator
            .vma
            .flush_allocation(&staging_alloc, 0, total_size as vk::DeviceSize)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to flush procedural skybox staging: {e}"))
            })?;

        // Create Cubemap Image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(6)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation) =
            unsafe { allocator.create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)? };

        // Upload
        vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
            // Transition to TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });

            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            // Copy 6 faces
            let mut regions = Vec::with_capacity(6);
            for i in 0..6 {
                regions.push(vk::BufferImageCopy {
                    buffer_offset: (i as u64 * resolution as u64 * resolution as u64 * 4)
                        as vk::DeviceSize,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: i as u32,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: resolution,
                        height: resolution,
                        depth: 1,
                    },
                });
            }

            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
            }

            // Transition to SHADER_READ_ONLY
            let barrier_end = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });

            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_end],
                );
            }
        })?;

        // Cleanup Staging
        unsafe {
            allocator
                .vma
                .destroy_buffer(staging_buffer, &mut staging_alloc);
        }

        // View
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            });

        let view = unsafe { device.create_image_view(&view_info, None)? };

        // Sampler (Linear for smooth skybox)
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        log::info!("Created procedural skybox ({resolution}x{resolution})");

        Ok(Self {
            image,
            view,
            sampler,
            allocation,
            allocator,
            device,
        })
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_image_view(self.view, None);
            self.allocator
                .vma
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}
