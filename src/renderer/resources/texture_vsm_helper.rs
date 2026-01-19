// Helper to create a 1x1 R32_UINT texture with NEAREST filtering for VSM page table
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
