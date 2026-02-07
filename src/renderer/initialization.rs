use crate::renderer::init_types::*;
use crate::renderer::model_renderer::{DRAW_PUSH_FRAGMENT_BYTES, DRAW_PUSH_VERTEX_BYTES};
use crate::renderer::resource_registry::ResourceRegistry;
use crate::renderer::resources::{
    self,
    uniform::{StorageBuffer, UniformBuffer},
    Texture, TextureData,
};
use crate::renderer::types::*;
use crate::renderer::Material;
use crate::vulkan;
use crate::AshError;
use crate::Result;
use ash::vk;

use std::sync::Arc;
use std::thread;

pub unsafe fn create_swapchain_data(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    extent: vk::Extent2D,
) -> Result<SwapchainData> {
    let swapchain = vulkan::SwapchainWrapper::new(device, device.headless, extent)?;

    let depth_buffer = resources::DepthBuffer::new(
        Arc::clone(&device.device),
        Arc::clone(alloc),
        swapchain.extent.width,
        swapchain.extent.height,
    )?;

    let mut render_pass = vulkan::RenderPass::builder(Arc::clone(&device.device))
        .with_swapchain_color(
            swapchain.format,
            if device.headless {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::PRESENT_SRC_KHR
            },
        )
        .with_depth_attachment(
            depth_buffer.format(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        .build()?;
    render_pass.mark_managed_by_registry();

    let mut framebuffers = Vec::new();
    for &image_view in &swapchain.image_views {
        let attachments = [image_view, depth_buffer.view()];
        let mut framebuffer = vulkan::Framebuffer::new(
            Arc::clone(&device.device),
            render_pass.handle(),
            &attachments,
            swapchain.extent,
        )?;
        framebuffer.mark_managed_by_registry();
        framebuffers.push(framebuffer);
    }

    Ok(SwapchainData {
        swapchain,
        render_pass: render_pass.handle(),
        framebuffers,
        depth_buffer,
    })
}

pub unsafe fn create_frame_resources(
    device: &vulkan::VulkanDevice,
    swapchain_image_count: usize,
) -> Result<FrameData> {
    let worker_count = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let command_manager = vulkan::CommandBufferManager::new(
        Arc::clone(&device.device),
        device.graphics_queue_family,
        worker_count,
    )?;

    let command_buffers = command_manager.allocate_primary_buffers(swapchain_image_count as u32)?;

    let mut frame_syncs = Vec::with_capacity(swapchain_image_count);
    for _ in 0..swapchain_image_count {
        let sync = vulkan::FrameSync::new(Arc::clone(&device.device))?;
        frame_syncs.push(sync);
    }

    Ok(FrameData {
        command_buffers,
        frame_syncs,
        command_manager,
    })
}

pub unsafe fn create_main_pipeline(
    device: &vulkan::VulkanDevice,
    resources: &Arc<ResourceRegistry>,
    swapchain_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    render_pass_id: crate::renderer::resource_registry::ResourceId,
    set_layouts: &[vk::DescriptorSetLayout],
    pipeline_cfg: &PipelineConfig,
    depth_format: vk::Format,
    pipeline_cache: vk::PipelineCache,
) -> Result<(
    vulkan::PipelineLayout,
    crate::renderer::resource_registry::ResourceId,
    vulkan::Pipeline,
    crate::renderer::resource_registry::ResourceId,
)> {
    let push_constant_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: DRAW_PUSH_VERTEX_BYTES + DRAW_PUSH_FRAGMENT_BYTES,
    }];

    let mut pipeline_layout_builder = vulkan::PipelineLayout::builder(Arc::clone(&device.device));
    for layout in set_layouts {
        pipeline_layout_builder = pipeline_layout_builder.add_set_layout(*layout);
    }
    for range in &push_constant_ranges {
        pipeline_layout_builder = pipeline_layout_builder.add_push_constant(*range);
    }
    let mut pipeline_layout_wrapper = pipeline_layout_builder.build()?;
    pipeline_layout_wrapper.mark_managed_by_registry();
    let pipeline_layout_handle = pipeline_layout_wrapper.handle();

    let mut pipeline_builder = vulkan::Pipeline::builder(Arc::clone(&device.device))
        .with_layout(pipeline_layout_handle)
        .with_render_pass(render_pass)
        .with_extent(swapchain_extent)
        .with_pipeline_cache(pipeline_cache)
        .with_depth_format(depth_format)
        .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, true)
        .with_cull_mode(vk::CullModeFlags::NONE)
        .with_front_face(vk::FrontFace::CLOCKWISE)
        .with_multisampling(pipeline_cfg.multisample_config())
        .add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/vert.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?
        .add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

    for specialization in &pipeline_cfg.specialization_constants {
        pipeline_builder = pipeline_builder.with_specialization_bytes(
            specialization.stage,
            specialization.constant_id,
            specialization.bytes(),
        );
    }

    let mut pipeline_wrapper = pipeline_builder.build()?;
    pipeline_wrapper.mark_managed_by_registry();
    let pipeline_handle = pipeline_wrapper.pipeline;

    // Register resources
    let pipeline_layout_id = resources.register_pipeline_layout(pipeline_layout_handle)?;
    let pipeline_id =
        resources.register_pipeline(pipeline_handle, &[pipeline_layout_id, render_pass_id])?;

    Ok((
        pipeline_layout_wrapper,
        pipeline_layout_id,
        pipeline_wrapper,
        pipeline_id,
    ))
}

pub unsafe fn init_resources(
    alloc: &Arc<vulkan::Allocator>,
    device: &vulkan::VulkanDevice,
    command_pool: vk::CommandPool,
    frame_count: usize,
    aspect: f32,
) -> Result<RendererResources> {
    // Initialize uniform buffers
    let mut uniform_buffers = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        let mut buffer =
            // SAFETY: We provide a valid allocator and device. The buffer size is determined strictly by `UniformBuffer::new` logic.
            UniformBuffer::new(Arc::clone(alloc), Arc::clone(&device.device))?;
        {
            let matrices = buffer.matrices_mut();
            matrices.set_view(
                glam::Vec3::new(0.0, 2.0, 5.0),
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            );
            matrices.set_projection(std::f32::consts::PI / 4.0, aspect, 0.5, 1000.0);
        }
        buffer.update()?;
        uniform_buffers.push(buffer);
    }

    // Create default texture
    let default_texture_data = TextureData::solid_color([255, 255, 255, 255]);
    let default_texture = Texture::from_data(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        command_pool,
        device.graphics_queue,
        &default_texture_data,
        vk::Format::R8G8B8A8_SRGB,
        Some("default_texture"),
    )?;

    // Create black texture for IBL fallback (provides some ambient light when IBL not loaded)
    let black_texture_data = TextureData::solid_color([0, 0, 0, 255]);
    let black_texture = Texture::from_data(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        command_pool,
        device.graphics_queue,
        &black_texture_data,
        vk::Format::R8G8B8A8_SRGB,
        Some("black_texture"),
    )?;

    // Create white texture for Occlusion Culling fallback (Standard-Z Far Plane = 1.0)
    let white_texture_data = TextureData::solid_color([255, 255, 255, 255]);
    let white_texture = Texture::from_data(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        command_pool,
        device.graphics_queue,
        &white_texture_data,
        vk::Format::R8G8B8A8_UNORM, // Use UNORM for precise 1.0 mapping
        Some("white_texture"),
    )?;

    // Create procedural skybox
    let default_skybox = Texture::create_procedural_skybox(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        command_pool,
        device.graphics_queue,
        512,
    )?;

    // Create default cube black
    let default_cube_black = Texture::create_default_cube_black(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        command_pool,
        device.graphics_queue,
    )?;

    // Initialize material storage buffer (Bindless-ready)
    let max_materials = 1024;
    let mut material_storage_buffer = StorageBuffer::<resources::uniform::MaterialUniform>::new(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        max_materials,
        "material_storage_buffer",
    )?;

    // Populate with default material at index 0
    let default_mat = Material::default();
    let mut initial_materials = vec![resources::uniform::MaterialUniform::default(); max_materials];

    let mut first_mat = resources::uniform::MaterialUniform::default();
    first_mat.set_base_color_factor(glam::Vec4::from_array(default_mat.color));
    first_mat.set_emissive_factor(glam::Vec4::from_array(default_mat.emissive));
    first_mat.set_metallic_roughness(default_mat.metallic, default_mat.roughness);
    first_mat.set_occlusion_strength(default_mat.occlusion_strength);
    first_mat.set_normal_scale(default_mat.normal_scale);
    first_mat.set_alpha_cutoff(default_mat.alpha_cutoff);
    initial_materials[0] = first_mat;

    material_storage_buffer.update(&initial_materials)?;

    // Initialize instance buffers for GPU culling/instancing
    let mut instance_buffers = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        let buffer = resources::InstanceBuffer::new(
            Arc::clone(alloc),
            Arc::clone(&device.device),
            crate::renderer::vcgs::MAX_CULLABLE_OBJECTS,
        )?;
        instance_buffers.push(buffer);
    }

    let post_sampler = device
        .device
        .create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )
        .map_err(|e| AshError::VulkanError(format!("Failed to create post_sampler: {e}")))?;

    // Phase 19: Transient Transform Arena
    let transform_arena_size = 1024 * 1024; // 1MB
    let (transform_arena, transform_arena_alloc) = alloc.create_buffer_with_flags_and_name(
        transform_arena_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk_mem::MemoryUsage::AutoPreferHost,
        vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            | vk_mem::AllocationCreateFlags::MAPPED,
        Some("Transform Arena (Phase 19)".to_string()),
    )?;

    Ok(RendererResources {
        uniform_buffers,
        default_texture,
        black_texture,
        white_texture,
        default_skybox,
        default_cube_black,
        material_storage_buffer,
        instance_buffers,
        transform_arena,
        transform_arena_alloc,
        post_sampler,
    })
}

pub unsafe fn create_frame_syncs_internal(
    device: &Arc<ash::Device>,
    count: usize,
) -> Result<Vec<vulkan::FrameSync>> {
    let mut frame_syncs = Vec::with_capacity(count);
    for _ in 0..count {
        let sync = vulkan::FrameSync::new(Arc::clone(device))?;
        frame_syncs.push(sync);
    }
    Ok(frame_syncs)
}
