use crate::renderer::resources::ImageHandle;
use crate::{AshError, Result};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IblAssetHeader {
    pub magic: [u8; 4],        // "AIBL" (Ash IBL)
    pub version: u32,          // Format version
    pub cubemap_size: u32,     // e.g., 512
    pub irradiance_size: u32,  // e.g., 32
    pub prefiltered_size: u32, // e.g., 128
    pub prefiltered_mips: u32, // Number of mip levels
    pub format: u32,           // ash::vk::Format as u32 (usually R32G32B32A32_SFLOAT)
    pub _padding: [u32; 9],    // 8-byte alignment (Total 64 bytes)
}

impl Default for IblAssetHeader {
    fn default() -> Self {
        Self {
            magic: *b"AIBL",
            version: 1,
            cubemap_size: 0,
            irradiance_size: 0,
            prefiltered_size: 0,
            prefiltered_mips: 0,
            format: 0,
            _padding: [0; 9],
        }
    }
}

/// Uploads IBL data from raw slices to the GPU.
///
/// This supports Zero-Copy loading as the data slices can come directly from a memory-mapped file.
#[allow(clippy::too_many_arguments)]
pub fn upload_ibl(
    device: Arc<ash::Device>,
    allocator: Arc<crate::vulkan::Allocator>,
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
    header: &IblAssetHeader,
    cubemap_data: &[u8],
    irradiance_data: &[u8],
    prefiltered_data: &[u8],
) -> Result<(ImageHandle, ImageHandle, ImageHandle)> {
    log::info!(
        "Creating cubemap image ({}x{})...",
        header.cubemap_size,
        header.cubemap_size
    );
    // 1. Create Cubemap
    let cubemap = ImageHandle::create_cubemap(
        Arc::clone(&device),
        Arc::clone(&allocator),
        header.cubemap_size,
        1,
        ash::vk::Format::from_raw(header.format as i32),
        ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
        Some("BakedVolumeCubemap".to_string()),
    )?;

    log::info!(
        "Creating irradiance image ({}x{})...",
        header.irradiance_size,
        header.irradiance_size
    );
    // 2. Create Irradiance
    let irradiance = ImageHandle::create_cubemap(
        Arc::clone(&device),
        Arc::clone(&allocator),
        header.irradiance_size,
        1,
        ash::vk::Format::from_raw(header.format as i32),
        ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
        Some("BakedIrradianceMap".to_string()),
    )?;

    log::info!(
        "Creating prefiltered image ({}x{}, {} mips)...",
        header.prefiltered_size,
        header.prefiltered_size,
        header.prefiltered_mips
    );
    // 3. Create Prefiltered
    let prefiltered = ImageHandle::create_cubemap(
        Arc::clone(&device),
        Arc::clone(&allocator),
        header.prefiltered_size,
        header.prefiltered_mips,
        ash::vk::Format::from_raw(header.format as i32),
        ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
        Some("BakedPrefilteredMap".to_string()),
    )?;

    // 4. Upload Data
    log::info!("Uploading IBL data...");

    upload_data_to_image(&cubemap, cubemap_data, command_pool, queue)?;
    upload_data_to_image(&irradiance, irradiance_data, command_pool, queue)?;
    upload_data_to_image(&prefiltered, prefiltered_data, command_pool, queue)?;

    log::info!("All IBL data uploaded successfully");

    Ok((cubemap, irradiance, prefiltered))
}

fn upload_data_to_image(
    image: &ImageHandle,
    data: &[u8],
    command_pool: ash::vk::CommandPool,
    queue: ash::vk::Queue,
) -> Result<()> {
    let allocator = image
        .allocator()
        .ok_or_else(|| AshError::VulkanError("No allocator for image".to_string()))?;
    let device = image.device();
    let size = data.len() as u64;

    let (staging_buffer, mut staging_alloc) = unsafe {
        allocator.create_buffer_with_flags(
            size,
            ash::vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
        )?
    };

    unsafe {
        let mut guard = allocator.map_allocation_guarded(&mut staging_alloc, size)?;
        guard.copy_from_slice(data);
    }

    crate::vulkan::utils::execute_single_use(device.as_ref(), command_pool, queue, |cmd| {
        // Transition to TRANSFER_DST_OPTIMAL
        let barrier = ash::vk::ImageMemoryBarrier::default()
            .old_layout(ash::vk::ImageLayout::UNDEFINED)
            .new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(ash::vk::AccessFlags::empty())
            .dst_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
            .image(image.handle())
            .subresource_range(ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: image.mip_levels(),
                base_array_layer: 0,
                layer_count: image.layers(),
            });

        unsafe {
            device.as_ref().cmd_pipeline_barrier(
                cmd,
                ash::vk::PipelineStageFlags::TOP_OF_PIPE,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // Copy regions
        let mut buffer_offset = 0;
        let mut regions = Vec::new();
        let format_size = match image.format() {
            ash::vk::Format::R32G32B32A32_SFLOAT => 16,
            ash::vk::Format::R16G16B16A16_SFLOAT => 8,
            ash::vk::Format::R16G16_SFLOAT => 4,
            ash::vk::Format::R8G8B8A8_UNORM | ash::vk::Format::R8G8B8A8_SRGB => 4,
            _ => {
                log::error!("Unsupported upload format: {:?}", image.format());
                4 // Fallback
            }
        };

        for mip in 0..image.mip_levels() {
            let mip_w = (image.extent().width >> mip).max(1);
            let mip_h = (image.extent().height >> mip).max(1);

            regions.push(ash::vk::BufferImageCopy {
                buffer_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: ash::vk::ImageSubresourceLayers {
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: 0,
                    layer_count: image.layers(),
                },
                image_offset: ash::vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: ash::vk::Extent3D {
                    width: mip_w,
                    height: mip_h,
                    depth: 1,
                },
            });

            buffer_offset +=
                mip_w as u64 * mip_h as u64 * format_size as u64 * image.layers() as u64;
        }

        unsafe {
            device.as_ref().cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image.handle(),
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }

        // Transition to SHADER_READ_ONLY_OPTIMAL
        let barrier_read = ash::vk::ImageMemoryBarrier::default()
            .old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(ash::vk::AccessFlags::SHADER_READ)
            .image(image.handle())
            .subresource_range(ash::vk::ImageSubresourceRange {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: image.mip_levels(),
                base_array_layer: 0,
                layer_count: image.layers(),
            });

        unsafe {
            device.as_ref().cmd_pipeline_barrier(
                cmd,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
                ash::vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_read],
            );
        }
    })?;

    unsafe {
        allocator
            .vma
            .destroy_buffer(staging_buffer, &mut staging_alloc);
    }

    Ok(())
}
