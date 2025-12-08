use std::sync::Arc;

use ash::vk;

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
        let image_size = (data.width as usize * data.height as usize * 4) as vk::DeviceSize;

        let mip_levels = (data.width.max(data.height) as f32).log2().floor() as u32 + 1;

        // Staging buffer
        let (staging_buffer, mut staging_alloc) = allocator.create_buffer(
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
        )?;

        let staging_ptr = allocator.vma.map_memory(&mut staging_alloc).map_err(|e| {
            AshError::VulkanError(format!("Failed to map texture staging buffer: {e}"))
        })?;
        std::ptr::copy_nonoverlapping(data.pixels.as_ptr(), staging_ptr, data.pixels.len());
        allocator
            .vma
            .flush_allocation(&staging_alloc, 0, image_size)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to flush texture staging buffer: {e}"))
            })?;
        allocator.vma.unmap_memory(&mut staging_alloc);

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
        execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
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

fn execute_single_use<F>(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    recorder: F,
) -> Result<()>
where
    F: FnOnce(vk::CommandBuffer),
{
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
    let command_buffer = command_buffers[0];

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        recorder(command_buffer);

        device.end_command_buffer(command_buffer)?;

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &command_buffers);
    }

    Ok(())
}
