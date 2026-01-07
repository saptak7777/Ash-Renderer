use std::sync::Arc;

use ash::vk;

use bytemuck::{Pod, Zeroable};

use crate::{vulkan, AshError, Result};

/// Header for the custom .ash_tex binary asset format
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AshTexHeader {
    /// Magic bytes "ASHT"
    pub magic: [u8; 4],
    /// Format version (currently 1)
    pub version: u32,
    /// Texture width in pixels
    pub width: u32,
    /// Texture height in pixels
    pub height: u32,
    /// Vulkan format (u32 cast)
    pub format: u32,
    /// Number of mip levels
    pub mip_levels: u32,
    /// Compression type (0=None, 1=BC7, 2=BC5)
    pub compression: u8,
    /// Padding to align struct to 4 bytes
    pub _padding: [u8; 3],
}

impl Default for AshTexHeader {
    fn default() -> Self {
        Self {
            magic: *b"ASHT",
            version: 1,
            width: 0,
            height: 0,
            format: 0,
            mip_levels: 0,
            compression: 0,
            _padding: [0; 3],
        }
    }
}

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

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Loads a cooked .ash_tex binary asset from disk.
    ///
    /// This method is much faster than loading standard images as it bypasses
    /// compression and mipmap generation.
    /// # Safety
    /// Caller must ensure Vulkan handles (device, queue, command_pool) are valid.
    pub unsafe fn load_from_ash_tex(
        path: &std::path::Path,
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        name: Option<&str>,
    ) -> Result<Self> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to open asset {path:?}: {e}"))
        })?;

        // Read Header
        let mut header = AshTexHeader::default();
        let header_slice = bytemuck::bytes_of_mut(&mut header);
        file.read_exact(header_slice).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to read asset header {path:?}: {e}"))
        })?;

        // Validation
        if &header.magic != b"ASHT" {
            return Err(crate::AshError::VulkanError(format!(
                "Invalid asset magic in {path:?}"
            )));
        }
        if header.version != 1 {
            return Err(crate::AshError::VulkanError(format!(
                "Unsupported asset version {} in {path:?}",
                header.version
            )));
        }

        // Read Mips
        let mut mips = Vec::with_capacity(header.mip_levels as usize);
        let mut width = header.width;
        let mut height = header.height;
        let compression = header.compression;

        for _ in 0..header.mip_levels {
            let size = if compression == 0 {
                // Uncompressed RGBA8
                (width * height * 4) as usize
            } else {
                // BC7 or BC5 (16 bytes per 4x4 block -> 1 byte per pixel effective)
                // But we need to handle block alignment
                let blocks_w = width.div_ceil(4);
                let blocks_h = height.div_ceil(4);
                (blocks_w * blocks_h * 16) as usize
            };

            let mut buffer = vec![0u8; size];
            file.read_exact(&mut buffer).map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to read mip data in {path:?}: {e}"))
            })?;
            mips.push(buffer);

            if width > 1 {
                width /= 2;
            }
            if height > 1 {
                height /= 2;
            }
        }

        // Reconstruct format
        let format = vk::Format::from_raw(header.format as i32);

        Self::from_mips(
            allocator,
            device,
            command_pool,
            queue,
            &mips,
            header.width,
            header.height,
            format,
            name,
        )
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
