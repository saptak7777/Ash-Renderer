use crate::AshError;
use crate::Result;
use std::io::{Read, Write};
use std::path::Path;
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
    pub _padding: [u32; 8],    // Future expansion
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
            _padding: [0; 8],
        }
    }
}

pub struct IblAsset {
    pub header: IblAssetHeader,
    pub cubemap_data: Vec<u8>,
    pub irradiance_data: Vec<u8>,
    pub prefiltered_data: Vec<u8>,
    pub brdf_lut_data: Vec<u8>,
}

impl IblAsset {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = std::fs::File::create(path).map_err(|e| {
            AshError::VulkanError(format!("Failed to create IBL asset file: {e}"))
        })?;

        // Write header
        file.write_all(bytemuck::bytes_of(&self.header))
            .map_err(|e| AshError::VulkanError(format!("Failed to write IBL header: {e}")))?;

        // Write lengths first for easy reading
        let lengths = [
            self.cubemap_data.len() as u64,
            self.irradiance_data.len() as u64,
            self.prefiltered_data.len() as u64,
            self.brdf_lut_data.len() as u64,
        ];
        file.write_all(bytemuck::cast_slice(&lengths))
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to write IBL data lengths: {e}"))
            })?;

        // Write data
        file.write_all(&self.cubemap_data)?;
        file.write_all(&self.irradiance_data)?;
        file.write_all(&self.prefiltered_data)?;
        file.write_all(&self.brdf_lut_data)?;

        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        log::info!("Loading IBL asset from: {:?}", path.as_ref());
        let mut file = std::fs::File::open(path)
            .map_err(|e| AshError::VulkanError(format!("Failed to open IBL asset file: {e}")))?;

        let mut header = IblAssetHeader::default();
        file.read_exact(bytemuck::bytes_of_mut(&mut header))
            .map_err(|e| AshError::VulkanError(format!("Failed to read IBL header: {e}")))?;

        if &header.magic != b"AIBL" {
            return Err(AshError::VulkanError("Invalid IBL asset magic".to_string()));
        }

        log::info!("Reading data lengths...");
        let mut lengths = [0u64; 4];
        file.read_exact(bytemuck::cast_slice_mut(&mut lengths))
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to read IBL data lengths: {e}"))
            })?;

        log::info!("Allocating cubemap buffer ({} bytes)...", lengths[0]);
        let mut cubemap_data = vec![0u8; lengths[0] as usize];
        file.read_exact(&mut cubemap_data)?;

        log::info!("Allocating irradiance buffer ({} bytes)...", lengths[1]);
        let mut irradiance_data = vec![0u8; lengths[1] as usize];
        file.read_exact(&mut irradiance_data)?;

        let mut prefiltered_data = vec![0u8; lengths[2] as usize];
        file.read_exact(&mut prefiltered_data)?;

        let mut brdf_lut_data = vec![0u8; lengths[3] as usize];
        file.read_exact(&mut brdf_lut_data)?;

        log::info!("IBL asset loaded successfully");
        Ok(Self {
            header,
            cubemap_data,
            irradiance_data,
            prefiltered_data,
            brdf_lut_data,
        })
    }

    /// Uploads the pre-baked IBL data to the GPU.
    pub fn upload_to_gpu(
        &self,
        device: Arc<ash::Device>,
        allocator: Arc<crate::vulkan::Allocator>,
        command_pool: ash::vk::CommandPool,
        queue: ash::vk::Queue,
    ) -> Result<(
        crate::renderer::resources::ImageHandle,
        crate::renderer::resources::ImageHandle,
        crate::renderer::resources::ImageHandle,
    )> {
        use crate::renderer::resources::ImageHandle;

        log::info!(
            "Creating cubemap image ({}x{})...",
            self.header.cubemap_size,
            self.header.cubemap_size
        );
        // 1. Create Cubemap
        let cubemap = ImageHandle::create_cubemap(
            Arc::clone(&device),
            Arc::clone(&allocator),
            self.header.cubemap_size,
            1,
            ash::vk::Format::from_raw(self.header.format as i32),
            ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
            Some("BakedVolumeCubemap".to_string()),
        )?;

        log::info!(
            "Creating irradiance image ({}x{})...",
            self.header.irradiance_size,
            self.header.irradiance_size
        );
        // 2. Create Irradiance
        let irradiance = ImageHandle::create_cubemap(
            Arc::clone(&device),
            Arc::clone(&allocator),
            self.header.irradiance_size,
            1,
            ash::vk::Format::from_raw(self.header.format as i32),
            ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
            Some("BakedIrradianceMap".to_string()),
        )?;

        log::info!(
            "Creating prefiltered image ({}x{}, {} mips)...",
            self.header.prefiltered_size,
            self.header.prefiltered_size,
            self.header.prefiltered_mips
        );
        // 3. Create Prefiltered
        let prefiltered = ImageHandle::create_cubemap(
            Arc::clone(&device),
            Arc::clone(&allocator),
            self.header.prefiltered_size,
            self.header.prefiltered_mips,
            ash::vk::Format::from_raw(self.header.format as i32),
            ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST,
            Some("BakedPrefilteredMap".to_string()),
        )?;

        // 4. Upload Data (Single submission for everything)
        // Note: For simplicity, we use the existing from_raw_data logic or implement a faster version here.
        // Since we have the raw bytes, we can use a single staging buffer.

        log::info!(
            "Uploading cubemap data ({} bytes)...",
            self.cubemap_data.len()
        );
        self.upload_data_to_image(&cubemap, &self.cubemap_data, command_pool, queue)?;
        log::info!(
            "Uploading irradiance data ({} bytes)...",
            self.irradiance_data.len()
        );
        self.upload_data_to_image(&irradiance, &self.irradiance_data, command_pool, queue)?;
        log::info!(
            "Uploading prefiltered data ({} bytes)...",
            self.prefiltered_data.len()
        );
        self.upload_data_to_image(&prefiltered, &self.prefiltered_data, command_pool, queue)?;
        log::info!("All IBL data uploaded successfully");

        Ok((cubemap, irradiance, prefiltered))
    }

    fn upload_data_to_image(
        &self,
        image: &crate::renderer::resources::ImageHandle,
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
}
