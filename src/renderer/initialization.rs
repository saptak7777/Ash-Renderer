use crate::renderer::init_types::*;
use crate::renderer::model_renderer::{DRAW_PUSH_FRAGMENT_BYTES, DRAW_PUSH_VERTEX_BYTES};
use crate::renderer::resource_registry::{ResourceId, ResourceRegistry};
use crate::renderer::resources::material::MAX_MATERIALS;
use crate::renderer::resources::{
    self,
    uniform::{StorageBuffer, UniformBuffer},
    Texture, TextureData,
};
use crate::renderer::types::*;
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

pub unsafe fn init_swapchain<S: vulkan::SurfaceProvider>(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    resources: &Arc<ResourceRegistry>,
    surface_provider: &S,
) -> Result<SwapchainDataWithIds> {
    let (width, height) = surface_provider.physical_size();
    let extent = vk::Extent2D { width, height };

    log::info!("Creating Swapchain & Frame Resources");
    let mut data = create_swapchain_data(device, alloc, extent)?;

    let mut swapchain_image_view_ids = Vec::with_capacity(data.swapchain.image_views.len());
    for &view in &data.swapchain.image_views {
        let id = resources.register_image_view(view)?;
        swapchain_image_view_ids.push(id);
    }
    data.swapchain.mark_image_views_managed_by_registry();

    let depth_buffer_id = data.depth_buffer.register_with_registry(resources)?;
    let render_pass_id = resources.register_render_pass(data.render_pass)?;

    let mut framebuffer_ids = Vec::with_capacity(data.framebuffers.len());
    for (idx, fb) in data.framebuffers.iter_mut().enumerate() {
        let id = resources.register_framebuffer(
            fb.handle(),
            &[
                render_pass_id,
                depth_buffer_id,
                swapchain_image_view_ids[idx],
            ],
        )?;
        fb.mark_managed_by_registry();
        framebuffer_ids.push(id);
    }

    Ok(SwapchainDataWithIds {
        data,
        swapchain_image_view_ids,
        depth_buffer_id,
        render_pass_id,
        framebuffer_ids,
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

pub unsafe fn init_pipelines(
    device: &vulkan::VulkanDevice,
    resources: &Arc<ResourceRegistry>,
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    render_pass_id: ResourceId,
    set_layouts: &[vk::DescriptorSetLayout],
    pipeline_cfg: &PipelineConfig,
    depth_format: vk::Format,
    pipeline_cache: vk::PipelineCache,
) -> Result<PipelineData> {
    let (layout, layout_id, pipeline, pipeline_id) = create_main_pipeline(
        device,
        resources,
        extent,
        render_pass,
        render_pass_id,
        set_layouts,
        pipeline_cfg,
        depth_format,
        pipeline_cache,
    )?;

    Ok(PipelineData {
        layout,
        layout_id,
        pipeline,
        pipeline_id,
    })
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
    let mut material_storage_buffer = StorageBuffer::<resources::uniform::MaterialUniform>::new(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        MAX_MATERIALS as usize,
        "material_storage_buffer",
    )?;

    // Reserve Index 0 as the "Magenta Error Material" (AAA Pattern)
    // This ensures that any mesh missing a material index shows up bright magenta.
    let error_mat = resources::uniform::MaterialUniform {
        base_color_factor: glam::Vec4::new(1.0, 0.0, 1.0, 1.0), // Bright Magenta
        emissive_factor: glam::Vec4::new(1.0, 0.0, 1.0, 1.0),   // Glowing Magenta
        parameters: glam::Vec4::new(0.0, 1.0, 1.0, 1.0),        // metallic 0, roughness 1
        ..resources::uniform::MaterialUniform::default()
    };

    // Initialize the buffer with error material at index 0
    unsafe {
        material_storage_buffer.write_element_at(0, &error_mat)?;
    }

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

    Ok(RendererResources {
        uniform_buffers,
        default_texture,
        black_texture,
        white_texture,
        default_skybox,
        default_cube_black,
        material_storage_buffer,
        instance_buffers,
        post_sampler,
    })
}

pub unsafe fn init_core_infrastructure(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    instance: &Arc<vulkan::VulkanInstance>,
    resources: &Arc<ResourceRegistry>,
    upload_command_pool: vk::CommandPool,
    frame_count: usize,
    aspect: f32,
) -> Result<CoreInfrastructure> {
    log::info!("Initializing Core Infrastructure...");

    let buffer_pool = Arc::new(resources::BufferPool::new(Arc::clone(alloc)));

    let geometry_buffer = Arc::new(resources::DualHeapGeometryBuffer::new(
        Arc::clone(&device.device),
        Arc::clone(alloc),
        256, // 256MB for vertices
        128, // 128MB for indices
    )?);

    let model_renderer = crate::renderer::model_renderer::ModelRenderer::new(
        Arc::clone(alloc),
        Arc::clone(&device.device),
        Arc::clone(&geometry_buffer),
    );

    let mut descriptor_allocator = vulkan::DescriptorAllocator::new(
        Arc::clone(&device.device),
        2048,
        Some(Arc::clone(resources)),
    )?;

    let bindless_manager = crate::vulkan::BindlessManager::new(
        instance.instance(),
        device.physical_device,
        Arc::clone(&device.device),
        &mut descriptor_allocator,
        crate::vulkan::BindlessManager::DEFAULT_MAX_TEXTURES,
        crate::vulkan::BindlessManager::DEFAULT_MAX_PAGE_TABLES,
        crate::vulkan::BindlessManager::DEFAULT_MAX_CUBEMAPS,
        crate::vulkan::BindlessManager::DEFAULT_MAX_STORAGE_IMAGES,
        crate::vulkan::BindlessManager::DEFAULT_MAX_BUFFERS,
    )?;

    let renderer_resources =
        init_resources(alloc, device, upload_command_pool, frame_count, aspect)?;

    Ok(CoreInfrastructure {
        buffer_pool,
        geometry_buffer,
        model_renderer,
        bindless_manager,
        descriptor_allocator,
        renderer_resources,
    })
}

pub unsafe fn init_rendering_passes(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    resources: &Arc<ResourceRegistry>,
    bindless_manager: &mut crate::vulkan::BindlessManager,
    renderer_resources: &RendererResources,
    swapchain_extent: vk::Extent2D,
    depth_format: vk::Format,
    depth_view: vk::ImageView,
    pipeline_cache: vk::PipelineCache,
    render_pass_handle: vk::RenderPass,
    multisample_config: vulkan::MultisampleConfig,
    set_layouts: &[vk::DescriptorSetLayout],
    model_renderer: &crate::renderer::model_renderer::ModelRenderer,
    upload_command_pool: vk::CommandPool,
) -> Result<RenderingPasses> {
    log::info!("Initializing Rendering Passes...");

    let gbuffer = crate::renderer::GBuffer::new(
        Arc::clone(&device.device),
        Arc::clone(alloc),
        swapchain_extent.width,
        swapchain_extent.height,
    )?;

    let mut hiz_pass = crate::renderer::passes::hiz::HiZPass::new(Arc::clone(&device.device));
    hiz_pass.init(
        alloc,
        device,
        swapchain_extent.width,
        swapchain_extent.height,
    )?;

    let mut indirect_draw_pass =
        crate::renderer::vcgs::IndirectDrawPass::new(Arc::clone(&device.device));
    indirect_draw_pass.init(
        alloc,
        device,
        bindless_manager,
        crate::renderer::vcgs::MAX_INDIRECT_OBJECTS,
    )?;

    // Register GBuffer indices
    let mut gbuffer_indices = GBufferIndices::default();
    gbuffer_indices.motion_index = bindless_manager.add_sampled_image(
        gbuffer.motion_view(),
        renderer_resources.default_texture.sampler(),
    )?;

    gbuffer_indices.depth_index = bindless_manager
        .add_sampled_image(depth_view, renderer_resources.default_texture.sampler())?;

    // Skybox Initialization
    let skybox_index = bindless_manager.add_cubemap(
        renderer_resources.default_skybox.view(),
        renderer_resources.default_skybox.sampler(),
    )?;

    let skybox_mesh = {
        let mut mesh = crate::renderer::Mesh::create_cube();
        for v in &mut mesh.vertices {
            v.position[0] *= 500.0;
            v.position[1] *= 500.0;
            v.position[2] *= 500.0;
        }
        model_renderer.upload_mesh_data(&mesh, upload_command_pool, device.graphics_queue)?
    };

    let skybox_pass = crate::renderer::passes::SkyboxPass::new(
        device,
        resources,
        render_pass_handle,
        swapchain_extent,
        pipeline_cache,
        depth_format,
        multisample_config,
        set_layouts,
        skybox_mesh,
        skybox_index,
    )?;

    Ok(RenderingPasses {
        gbuffer,
        gbuffer_indices,
        hiz_pass,
        indirect_draw_pass,
        skybox_pass,
    })
}

pub unsafe fn init_lighting_system(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    bindless_manager: &mut crate::vulkan::BindlessManager,
    upload_command_pool: vk::CommandPool,
    frame_count: u32,
    extent: vk::Extent2D,
) -> Result<LightingSystem> {
    log::info!("Initializing Lighting System...");

    let global_cluster_buffer = crate::renderer::resources::GlobalClusterBuffer::new(
        Arc::clone(&device.device),
        Arc::clone(alloc),
        64, // 64MB capacity
    )?;

    let mut forward_plus = crate::renderer::ForwardPlusIntegration::new(
        Arc::clone(&device.device),
        alloc,
        frame_count,
    )?;
    forward_plus.init(alloc);
    forward_plus.on_resize(extent.width, extent.height);

    let shadow_system = match crate::renderer::features::ShadowSystem::new(
        Arc::clone(&device.device),
        Arc::clone(alloc),
        bindless_manager,
        upload_command_pool,
        device.graphics_queue,
        crate::renderer::features::default_vsm_config(),
        frame_count,
    ) {
        Ok(system) => Some(system),
        Err(e) => {
            log::error!("Failed to initialize Shadow System: {e}");
            None
        }
    };

    Ok(LightingSystem {
        forward_plus,
        shadow_system,
        global_cluster_buffer,
    })
}

pub fn init_post_processing(
    device: &Arc<ash::Device>,
    frame_count: usize,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<PostProcessingSystem> {
    log::info!("Initializing Post-Processing System...");

    let post_process = crate::renderer::systems::post_process::PostProcessSystem::new(
        Arc::clone(device),
        frame_count,
        extent,
        format,
    )?;

    Ok(PostProcessingSystem { post_process })
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

pub unsafe fn init_render_queue(device: &vulkan::VulkanDevice) -> Result<RenderQueueData> {
    let queue = crate::renderer::queue::RenderQueue::new(
        Arc::clone(&device.device),
        device.graphics_queue,
        device.present_queue,
        device.graphics_queue_family,
    )?;

    Ok(RenderQueueData { queue })
}
