use crate::AshError;
use crate::Result;
use crate::renderer::init_types::*;
use crate::renderer::model_renderer::{DRAW_PUSH_FRAGMENT_BYTES, DRAW_PUSH_VERTEX_BYTES};
use crate::renderer::resource_registry::ResourceRegistry;
use crate::renderer::resources::material::MAX_MATERIALS;
use crate::renderer::resources::{
    self, Texture, TextureData,
    uniform::{StorageBuffer, UniformBuffer},
};
use crate::renderer::{TextureInitContext, types::*};
use crate::vulkan;
use ash::vk;

use crate::renderer::passes::SkyboxInitContext;
use crate::renderer::passes::hiz::HiZPass;
use crate::renderer::vcgs::IndirectDrawPass;
use std::sync::{Arc, RwLock};
use std::thread;

pub type PassResult = Result<(Arc<RwLock<HiZPass>>, Arc<RwLock<IndirectDrawPass>>)>;

/// Configuration for main pipeline creation.
pub struct MainPipelineCreateDesc<'a> {
    pub device: &'a vulkan::VulkanDevice,
    pub resources: &'a Arc<ResourceRegistry>,
    pub swapchain_extent: vk::Extent2D,
    pub color_formats: &'a [vk::Format],
    pub set_layouts: &'a [vk::DescriptorSetLayout],
    pub pipeline_cfg: &'a PipelineConfig,
    pub depth_format: vk::Format,
    pub pipeline_cache: vk::PipelineCache,
}

/// Creates swapchain data including the swapchain wrapper and depth buffer.
///
/// # Safety
/// The caller must ensure that the device and allocator are valid and that the extent matches the surface capabilities.
pub unsafe fn create_swapchain_data(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
) -> Result<SwapchainData> {
    let swapchain =
        unsafe { vulkan::SwapchainWrapper::new(device, device.headless, extent, present_mode)? };

    let depth_buffer = unsafe {
        resources::DepthBuffer::new(
            Arc::clone(&device.device),
            Arc::clone(alloc),
            swapchain.extent.width,
            swapchain.extent.height,
        )?
    };

    Ok(SwapchainData {
        swapchain,
        depth_buffer,
    })
}

/// Initializes the swapchain and registers image views and depth buffers.
///
/// # Safety
/// The caller must ensure that the device, allocator, and resources registry are valid.
pub unsafe fn init_swapchain<S: vulkan::SurfaceProvider>(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    resources: &Arc<ResourceRegistry>,
    surface_provider: &S,
    present_mode: vk::PresentModeKHR,
) -> Result<SwapchainDataWithIds> {
    let (width, height) = surface_provider.physical_size();
    let extent = vk::Extent2D { width, height };

    log::info!("Creating Swapchain & Frame Resources");
    let mut data = unsafe { create_swapchain_data(device, alloc, extent, present_mode)? };

    let mut swapchain_image_view_ids = Vec::with_capacity(data.swapchain.image_views.len());
    for &view in &data.swapchain.image_views {
        let id = resources.register_image_view(view)?;
        swapchain_image_view_ids.push(id);
    }
    data.swapchain.mark_image_views_managed_by_registry();

    let depth_buffer_id = data.depth_buffer.register_with_registry(resources)?;

    Ok(SwapchainDataWithIds {
        data,
        swapchain_image_view_ids,
        depth_buffer_id,
    })
}

/// Creates per-frame synchronization objects and command pools.
///
/// # Safety
/// The caller must ensure that the device handle is valid and that the max frames in flight
/// is greater than zero.
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

/// Creates the main rendering pipeline and layout.
///
/// # Safety
/// The provided Vulkan device, resources,    /// Creates the main rendering pipeline for the forward pass.
///
/// # Safety
/// The caller must ensure that the device and render pass are valid.
pub unsafe fn create_main_pipeline(
    desc: MainPipelineCreateDesc,
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

    let mut pipeline_layout_builder =
        vulkan::PipelineLayout::builder(Arc::clone(&desc.device.device));
    for layout in desc.set_layouts {
        pipeline_layout_builder = pipeline_layout_builder.add_set_layout(*layout);
    }
    for range in &push_constant_ranges {
        pipeline_layout_builder = pipeline_layout_builder.add_push_constant(*range);
    }
    let mut pipeline_layout_wrapper = pipeline_layout_builder.build()?;
    pipeline_layout_wrapper.mark_managed_by_registry();
    let pipeline_layout_handle = pipeline_layout_wrapper.handle();

    let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/forward.vert.spv"));
    let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/forward.frag.spv"));

    let mut pipeline_builder = vulkan::Pipeline::builder(Arc::clone(&desc.device.device))
        .with_layout(pipeline_layout_handle)
        .with_dynamic_rendering(desc.color_formats, Some(desc.depth_format), None)
        .with_extent(desc.swapchain_extent)
        .with_pipeline_cache(desc.pipeline_cache)
        .with_depth_format(desc.depth_format)
        .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, true)
        .with_cull_mode(vk::CullModeFlags::BACK)
        .with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .with_multisampling(desc.pipeline_cfg.multisample_config())
        .add_shader_from_bytes(vert_code, vk::ShaderStageFlags::VERTEX, "main")?
        .add_shader_from_bytes(frag_code, vk::ShaderStageFlags::FRAGMENT, "main")?;

    for specialization in &desc.pipeline_cfg.specialization_constants {
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
    let pipeline_layout_id = desc
        .resources
        .register_pipeline_layout(pipeline_layout_handle)?;
    let pipeline_id = desc
        .resources
        .register_pipeline(pipeline_handle, &[pipeline_layout_id])?;

    Ok((
        pipeline_layout_wrapper,
        pipeline_layout_id,
        pipeline_wrapper,
        pipeline_id,
    ))
}

/// Configuration for pipeline initialization.
pub struct PipelineInitInfo<'a> {
    pub device: &'a vulkan::VulkanDevice,
    pub resources: &'a Arc<ResourceRegistry>,
    pub extent: vk::Extent2D,
    pub color_formats: &'a [vk::Format],
    pub set_layouts: &'a [vk::DescriptorSetLayout],
    pub pipeline_cfg: &'a PipelineConfig,
    pub depth_format: vk::Format,
    pub pipeline_cache: vk::PipelineCache,
}

/// Initializes all pipelines for the renderer.
///
/// # Safety
/// All provided Vulkan handles (device, resources, etc.) must be valid and appropriately synchronized.
pub unsafe fn init_pipelines(info: PipelineInitInfo<'_>) -> Result<PipelineData> {
    let PipelineInitInfo {
        device,
        resources,
        extent,
        color_formats,
        set_layouts,
        pipeline_cfg,
        depth_format,
        pipeline_cache,
    } = info;

    let (layout, layout_id, pipeline, pipeline_id) = unsafe {
        create_main_pipeline(MainPipelineCreateDesc {
            device,
            resources,
            swapchain_extent: extent,
            color_formats,
            set_layouts,
            pipeline_cfg,
            depth_format,
            pipeline_cache,
        })?
    };

    Ok(PipelineData {
        layout,
        layout_id,
        pipeline,
        pipeline_id,
    })
}

/// Initializes core rendering resources like uniform buffers and default textures.
///
/// # Safety
/// The caller must ensure that the allocator, device, and command pool are valid and that
/// the queue can be used for transfer operations.
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
            unsafe { UniformBuffer::new(Arc::clone(alloc), Arc::clone(&device.device))? };
        {
            let matrices = buffer.matrices_mut();
            matrices.set_view(
                glam::Vec3::new(0.0, 2.0, 5.0),
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            );
            matrices.set_projection(std::f32::consts::PI / 4.0, aspect, 0.5, 1000.0);
        }
        unsafe { buffer.update()? };
        uniform_buffers.push(buffer);
    }

    let tex_ctx = TextureInitContext {
        allocator: Arc::clone(alloc),
        device: Arc::clone(&device.device),
        command_pool,
        queue: device.graphics_queue,
    };

    // Create default texture
    let default_texture_data = TextureData::solid_color([255, 255, 255, 255]);
    let default_texture = unsafe {
        Texture::from_data(
            &tex_ctx,
            &default_texture_data,
            vk::Format::R8G8B8A8_SRGB,
            Some("default_texture"),
        )?
    };

    // Create black texture for IBL fallback (provides some ambient light when IBL not loaded)
    let black_texture_data = TextureData::solid_color([0, 0, 0, 255]);
    let black_texture = unsafe {
        Texture::from_data(
            &tex_ctx,
            &black_texture_data,
            vk::Format::R8G8B8A8_SRGB,
            Some("black_texture"),
        )?
    };

    // Create white texture for Occlusion Culling fallback (Standard-Z Far Plane = 1.0)
    let white_texture_data = TextureData::solid_color([255, 255, 255, 255]);
    let white_texture = unsafe {
        Texture::from_data(
            &tex_ctx,
            &white_texture_data,
            vk::Format::R8G8B8A8_UNORM, // Use UNORM for precise 1.0 mapping
            Some("white_texture"),
        )?
    };

    // Create procedural skybox
    let default_skybox = Texture::create_procedural_skybox(&tex_ctx, 512)?;

    // Create default cube black
    let default_cube_black = Texture::create_default_cube_black(&tex_ctx)?;

    // Create black dummy textures for IBL fallbacks
    let dummy_black_cube = Texture::create_default_cube_black(&tex_ctx)?;

    let dummy_black_2d = Texture::create_default_black(&tex_ctx)?;

    // Initialize material storage buffer (Bindless-ready)
    let mut material_storage_buffer = unsafe {
        StorageBuffer::<resources::uniform::MaterialUniform>::new(
            Arc::clone(alloc),
            Arc::clone(&device.device),
            MAX_MATERIALS as usize,
            "material_storage_buffer",
        )?
    };

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

    let post_sampler = unsafe {
        device.device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )
    }
    .map_err(|e| AshError::VulkanError(format!("Failed to create post_sampler: {e}")))?;

    Ok(RendererResources {
        uniform_buffers,
        default_texture,
        black_texture,
        white_texture,
        default_skybox,
        default_cube_black,
        dummy_black_cube,
        dummy_black_2d,
        material_storage_buffer,
        post_sampler,
    })
}

/// Initializes all core renderer systems and infrastructure.
///
/// # Safety
/// The caller must ensure that the Vulkan instance and device are fully initialized and valid.
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

    let geometry_buffer = Arc::new(unsafe {
        resources::DualHeapGeometryBuffer::new(
            Arc::clone(&device.device),
            Arc::clone(alloc),
            256, // 256MB for vertices
            128, // 128MB for indices
        )?
    });

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
        crate::vulkan::BindlessConfig::default(),
    )?;

    let renderer_resources =
        unsafe { init_resources(alloc, device, upload_command_pool, frame_count, aspect)? };

    Ok(CoreInfrastructure {
        buffer_pool,
        geometry_buffer,
        model_renderer,
        bindless_manager,
        descriptor_allocator,
        renderer_resources,
    })
}

pub struct RenderingPassesConfig<'a> {
    pub device: &'a vulkan::VulkanDevice,
    pub alloc: &'a Arc<vulkan::Allocator>,
    pub resources: &'a Arc<ResourceRegistry>,
    pub bindless_manager: &'a mut crate::vulkan::BindlessManager,
    pub renderer_resources: &'a RendererResources,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub depth_format: vk::Format,
    pub depth_view: vk::ImageView,
    pub pipeline_cache: vk::PipelineCache,
    pub multisample_config: vulkan::MultisampleConfig,
    pub set_layouts: &'a [vk::DescriptorSetLayout],
    pub model_renderer: &'a crate::renderer::model_renderer::ModelRenderer,
    pub upload_command_pool: vk::CommandPool,
}

/// Initializes the rendering pass chain based on the provided configuration.
///
/// # Safety
/// The caller must ensure all dependencies (GBuffer, resource registry, etc.) are valid.
pub unsafe fn init_rendering_passes(cfg: RenderingPassesConfig) -> Result<RenderingPasses> {
    let RenderingPassesConfig {
        device,
        alloc,
        resources,
        bindless_manager,
        renderer_resources,
        swapchain_format,
        swapchain_extent,
        depth_format,
        depth_view,
        pipeline_cache,
        multisample_config,
        set_layouts,
        model_renderer,
        upload_command_pool,
    } = cfg;
    log::info!("Initializing Rendering Passes...");

    let gbuffer = unsafe {
        crate::renderer::GBuffer::new(
            Arc::clone(&device.device),
            Arc::clone(alloc),
            swapchain_extent.width,
            swapchain_extent.height,
        )?
    };

    let mut hiz_pass = crate::renderer::passes::hiz::HiZPass::new(Arc::clone(&device.device));
    unsafe {
        hiz_pass.init(
            alloc,
            device,
            swapchain_extent.width,
            swapchain_extent.height,
        )?
    };

    let mut indirect_draw_pass =
        crate::renderer::vcgs::IndirectDrawPass::new(Arc::clone(&device.device));
    unsafe {
        indirect_draw_pass.init(
            alloc,
            device,
            bindless_manager,
            crate::renderer::vcgs::MAX_INDIRECT_OBJECTS,
        )?
    };

    // Register GBuffer indices
    let gbuffer_indices = GBufferIndices {
        motion_index: bindless_manager.add_sampled_image(
            gbuffer.motion_view(),
            renderer_resources.default_texture.sampler(),
        )?,
        depth_index: bindless_manager
            .add_sampled_image(depth_view, renderer_resources.default_texture.sampler())?,
    };

    // ── Skybox Data: Cubemap binding + oversized cube mesh ─────────────────
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

    // ── Construct Skybox Pass (logical, no GPU work) ─────────────
    let mut skybox_pass = crate::renderer::passes::SkyboxPass::new(skybox_mesh, skybox_index);

    // ── Initialize Skybox GPU Resources ───────────────────────────
    unsafe {
        skybox_pass.init(SkyboxInitContext {
            device,
            resources,
            color_format: swapchain_format,
            extent: swapchain_extent,
            pipeline_cache,
            depth_format,
            multisample_config,
            set_layouts,
        })?
    };

    Ok(RenderingPasses {
        gbuffer: Some(gbuffer),
        gbuffer_indices,
        hiz_pass: Some(hiz_pass),
        indirect_draw_pass: Some(indirect_draw_pass),
        skybox_pass: Some(skybox_pass),
    })
}

/// Initializes the lighting and shadow management systems.
///
/// # Safety
/// The caller must ensure that the device and resource registry are valid.
pub unsafe fn init_lighting_system(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    _bindless_manager: &mut crate::vulkan::BindlessManager,
    _upload_command_pool: vk::CommandPool,
    frame_count: u32,
    extent: vk::Extent2D,
) -> Result<LightingSystem> {
    log::info!("Initializing Lighting System...");

    let global_cluster_buffer = unsafe {
        crate::renderer::resources::GlobalClusterBuffer::new(
            Arc::clone(&device.device),
            Arc::clone(alloc),
            64, // 64MB capacity
        )?
    };

    let mut forward_plus = unsafe {
        crate::renderer::ForwardPlusIntegration::new(
            Arc::clone(&device.device),
            alloc,
            frame_count,
        )?
    };
    unsafe { forward_plus.init(alloc) };
    forward_plus.on_resize(extent.width, extent.height);

    // VSM is now handled directly by Renderer and registers itself.
    // ShadowSystem is being removed.

    Ok(LightingSystem {
        forward_plus,
        global_cluster_buffer: Arc::new(global_cluster_buffer),
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

/// Internal helper to create frame synchronization primitives.
///
/// # Safety
/// The caller must ensure that the device is valid.
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

/// Initializes the command submission queue and associated data.
///
/// # Safety
/// The caller must ensure that the Vulkan device is valid.
pub unsafe fn init_render_queue(device: &vulkan::VulkanDevice) -> Result<RenderQueueData> {
    let queue = crate::renderer::queue::RenderQueue::new(
        Arc::clone(&device.device),
        device.graphics_queue,
        device.present_queue,
        device.graphics_queue_family,
    )?;

    Ok(RenderQueueData { queue })
}

/// Initializes the occlusion culling compute system.
///
/// # Safety
/// The caller must ensure that the device and resource registry are valid.
pub unsafe fn initialize_occlusion_culling(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    bindless_manager: &mut crate::vulkan::BindlessManager,
    _black_texture: &crate::renderer::resources::Texture,
    extent: vk::Extent2D,
) -> PassResult {
    // 1. Create Hi-Z pass
    let mut hiz = HiZPass::new(Arc::clone(&device.device));
    unsafe { hiz.init(alloc, device, extent.width, extent.height)? };

    // 2. Create Indirect Draw pass
    let mut indirect = IndirectDrawPass::new(Arc::clone(&device.device));
    unsafe {
        indirect.init(
            alloc,
            device,
            bindless_manager,
            crate::renderer::vcgs::MAX_INDIRECT_OBJECTS,
        )?
    };

    // Phase 5: Hi-Z is now a flat BDA buffer — no image view or sampler to wire up.
    // The culling pipeline reads the buffer directly via the device address in push constants.
    // (hiz.hiz_buffer_addr() is queried at dispatch time in execute_culling.)

    Ok((Arc::new(RwLock::new(hiz)), Arc::new(RwLock::new(indirect))))
}
