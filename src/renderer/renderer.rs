use crate::{
    renderer::{
        diagnostics::{
            DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler,
        },
        features::{
            AutoRotateFeature, FeatureFrameContext, FeatureManager, FeatureRenderContext,
            ShadowFeature,
        },
        forward_plus_integration::ForwardPlusIntegration,
        fullscreen_pass, hdr_framebuffer,
        hiz_pass::HiZPass,
        indirect_draw::IndirectDrawPass,
        model_renderer::{MaterialPushConstants, MeshPushConstants, ModelRenderer},
        occlusion_culling::{CullBoundingBox, OcclusionCulling},
        resource_registry::{ResourceId, ResourceRegistry},
        resources,
        resources::uniform::{MaterialBuffer, UniformBuffer},
        ssgi_pass::{SsgiPass, SsgiQuality},
        temporal_upscaling::{VsrPass, VsrQuality},
        DepthBuffer, GBuffer, Material, Mesh, PipelineCache, Texture, TextureData, Transform,
    },
    vulkan, AshError, Result,
};

use ash::vk;
use bytemuck::Pod;
use glam::{Mat4, Vec3, Vec4};
use parking_lot::Mutex;
use resources::BufferPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::renderer::resources::mesh::{MaterialDescriptor, MeshDescriptor};

#[derive(Clone, Copy, Debug, Default)]
pub enum MsaaPreset {
    #[default]
    Off,
    X2,
    X4,
    X8,
}

/// A render command specifying a mesh, material, and transform to render.
#[derive(Clone, Debug)]
pub struct RenderCommand {
    /// Handle identifying the mesh to render
    pub mesh_handle: u32,
    /// Handle identifying the material to use
    pub material_handle: u32,
    /// Transform matrix for positioning the mesh in world space
    pub transform: Mat4,
}

fn compute_worker_index(worker_count: usize, frame_index: usize) -> usize {
    if worker_count == 0 {
        0
    } else {
        frame_index % worker_count
    }
}

fn validate_worker_resources(
    worker_count: usize,
    descriptor_count: usize,
    buffer_count: usize,
) -> Result<()> {
    if worker_count == 0 {
        return Ok(());
    }

    if descriptor_count != worker_count {
        return Err(AshError::VulkanError(format!(
            "material descriptor count ({descriptor_count}) must match worker count ({worker_count})"
        )));
    }

    if buffer_count != worker_count {
        return Err(AshError::VulkanError(format!(
            "material buffer count ({buffer_count}) must match worker count ({worker_count})"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_worker_index, validate_worker_resources};

    #[test]
    fn worker_index_zero_workers() {
        assert_eq!(compute_worker_index(0, 0), 0);
        assert_eq!(compute_worker_index(0, 5), 0);
    }

    #[test]
    fn worker_index_wraps() {
        assert_eq!(compute_worker_index(4, 0), 0);
        assert_eq!(compute_worker_index(4, 3), 3);
        assert_eq!(compute_worker_index(4, 4), 0);
        assert_eq!(compute_worker_index(4, 7), 3);
    }

    #[test]
    fn validate_worker_resources_ok() {
        assert!(validate_worker_resources(0, 0, 0).is_ok());
        assert!(validate_worker_resources(2, 2, 2).is_ok());
    }

    #[test]
    fn validate_worker_resources_errors_on_mismatch() {
        assert!(validate_worker_resources(2, 1, 2).is_err());
        assert!(validate_worker_resources(2, 2, 1).is_err());
    }
}

impl MsaaPreset {
    fn sample_count(self) -> vk::SampleCountFlags {
        match self {
            MsaaPreset::Off => vk::SampleCountFlags::TYPE_1,
            MsaaPreset::X2 => vk::SampleCountFlags::TYPE_2,
            MsaaPreset::X4 => vk::SampleCountFlags::TYPE_4,
            MsaaPreset::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpecializationOverride {
    pub stage: vk::ShaderStageFlags,
    pub constant_id: u32,
    data: Vec<u8>,
}

impl SpecializationOverride {
    pub fn from_value<T: Pod>(stage: vk::ShaderStageFlags, constant_id: u32, value: &T) -> Self {
        Self {
            stage,
            constant_id,
            data: bytemuck::bytes_of(value).to_vec(),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub msaa: MsaaPreset,
    pub enable_sample_shading: bool,
    pub min_sample_shading: f32,
    pub watch_shaders: bool,
    pub specialization_constants: Vec<SpecializationOverride>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            msaa: MsaaPreset::Off,
            enable_sample_shading: false,
            min_sample_shading: 0.0,
            watch_shaders: false,
            specialization_constants: Vec::new(),
        }
    }
}

impl PipelineConfig {
    fn multisample_config(&self) -> vulkan::MultisampleConfig {
        vulkan::MultisampleConfig {
            sample_count: self.msaa.sample_count(),
            enable_sample_shading: self.enable_sample_shading,
            min_sample_shading: self.min_sample_shading,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RendererConfig {
    pub pipeline: PipelineConfig,
}

/// Main rendering system.
pub struct Renderer {
    // Resources dependent on allocator/device - dropped in reverse order.
    buffer_pool: Arc<BufferPool>,
    resources: Arc<ResourceRegistry>,
    features: FeatureManager,
    _pipeline_cache: PipelineCache,
    cmds: vulkan::CommandBufferManager,
    worker_count: usize,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_syncs: Vec<vulkan::FrameSync>,
    current_frame: usize,
    _default_texture: Texture,
    model_renderer: ModelRenderer,
    draw_items: Vec<DrawItem>,
    swapchain: Option<vulkan::SwapchainWrapper>,
    render_pass: Option<vulkan::RenderPass>,
    render_pass_id: Option<ResourceId>,
    pipeline: Option<vulkan::Pipeline>,
    pipeline_id: Option<ResourceId>,
    depth_buffer: Option<DepthBuffer>,
    uniform_buffers: Vec<UniformBuffer>,
    material_buffers: Vec<Mutex<MaterialBuffer>>,
    pipeline_layout: Option<vulkan::PipelineLayout>,
    pipeline_layout_id: Option<ResourceId>,
    descriptors: Option<vulkan::DescriptorManager>,
    framebuffers: Vec<vulkan::Framebuffer>,
    framebuffer_ids: Vec<ResourceId>,
    start_time: Instant,
    pub mesh: Option<Mesh>,
    material: Material,
    transform: Transform,
    mesh_registry: HashMap<u32, String>,
    mesh_indices_registry: HashMap<String, ([i32; 4], i32)>,
    mesh_texture_flags: HashMap<String, TexturePresenceFlags>,
    material_registry: HashMap<u32, Material>,
    swapchain_image_view_ids: Vec<ResourceId>,
    depth_buffer_id: Option<ResourceId>,
    frame_sync_ids: Vec<(ResourceId, ResourceId, ResourceId)>,
    old_swapchain_handles: Vec<vk::SwapchainKHR>,
    swapchain_cleanup_pending: bool,
    resize_pending: bool,
    pending_extent: Option<vk::Extent2D>,
    // Post-processing support
    msaa_preset: MsaaPreset,
    hdr_framebuffer: Option<hdr_framebuffer::HdrFramebuffer>,
    fullscreen_pass: Option<fullscreen_pass::FullscreenPass>,
    tonemapping_enabled: bool,
    tonemapping_exposure: f32,
    tonemapping_gamma: f32,
    bloom_enabled: bool,
    bloom_intensity: f32,
    // Diagnostics
    diagnostics: DiagnosticsState,
    frame_profiler: FrameProfiler,
    gpu_profiler: Option<GpuProfiler>,
    diagnostics_overlay: DiagnosticsOverlay,
    // Shadows
    shadow_feature: ShadowFeature,
    shadow_pipeline: Option<vulkan::Pipeline>,
    shadow_pipeline_layout: Option<vulkan::PipelineLayout>,
    // Bindless textures
    bindless_manager: Option<vulkan::BindlessManager>,
    // Forward+ lighting
    forward_plus: Option<ForwardPlusIntegration>,
    // GPU-driven occlusion culling (Hi-Z + Indirect Draw)
    hiz_pass: Option<HiZPass>,
    indirect_draw_pass: Option<IndirectDrawPass>,
    occlusion_culling: OcclusionCulling,
    // Temporal Super-Resolution
    vsr_pass: Option<VsrPass>,
    // Screen-Space Global Illumination
    ssgi_pass: Option<SsgiPass>,
    // G-Buffer for Normals and Motion Vectors
    gbuffer: Option<GBuffer>,
    // Post-processing descriptors
    post_descriptor_pool: vk::DescriptorPool,
    post_descriptor_sets: Vec<vk::DescriptorSet>,
    post_pipeline: Option<vk::Pipeline>,
    post_framebuffers: Vec<vulkan::Framebuffer>,
    // Allocator and device are dropped last as they are the foundation for the resources above.
    alloc: Arc<vulkan::Allocator>,
    device: vulkan::VulkanDevice,
}

#[derive(Clone)]
struct DrawItem {
    key: String,
    transform: Mat4,
    material: Material,
    texture_flags: TexturePresenceFlags,
    texture_indices: [i32; 4], // base, normal, mr, occ
    emissive_index: i32,
}

#[derive(Copy, Clone, Default, Debug)]
struct TexturePresenceFlags {
    base_color: bool,
    normal: bool,
    metallic_roughness: bool,
    occlusion: bool,
    emissive: bool,
}

impl TexturePresenceFlags {
    pub fn from_mesh(mesh: &Mesh) -> Self {
        Self {
            base_color: mesh.texture.is_some(),
            normal: mesh.normal_texture.is_some(),
            metallic_roughness: mesh.metallic_roughness_texture.is_some(),
            occlusion: mesh.occlusion_texture.is_some(),
            emissive: mesh.emissive_texture.is_some(),
        }
    }
}

impl Renderer {
    /// Initializes the renderer.
    pub fn new<S: vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        unsafe {
            let instance = Arc::new(vulkan::VulkanInstance::new(
                surface_provider,
                cfg!(debug_assertions),
            )?);
            let device = vulkan::VulkanDevice::new(Arc::clone(&instance))?;
            let alloc = Arc::new(vulkan::Allocator::new(&device)?);
            let resources = Arc::new(ResourceRegistry::new(Arc::clone(&device.device)));
            let mut features = FeatureManager::new();
            features.set_device(Arc::clone(&device.device));
            features.add_feature(AutoRotateFeature::new());

            // Initialize Shadow Feature
            let mut shadow_feature = ShadowFeature::new();
            if shadow_feature.is_active() || shadow_feature.config.enabled {
                let shadow_map = crate::renderer::shadow_map::ShadowMap::new(
                    Arc::clone(&device.device),
                    device.memory_properties,
                    shadow_feature.config.clone(),
                )?;
                shadow_feature.set_shadow_map(shadow_map);
            }
            let pipeline_cache = PipelineCache::new(Arc::clone(&device.device))?;
            let renderer_config = RendererConfig::default();
            let pipeline_cfg = &renderer_config.pipeline;
            let buffer_pool = Arc::new(BufferPool::new(Arc::clone(&alloc)));
            let mut swapchain = vulkan::SwapchainWrapper::new(&device)?;
            let mut swapchain_image_view_ids = Vec::with_capacity(swapchain.image_views.len());
            for &image_view in &swapchain.image_views {
                let image_view_id = resources.register_image_view(image_view).map_err(|e| {
                    AshError::VulkanError(format!("Failed to register swapchain image view: {e}"))
                })?;
                swapchain_image_view_ids.push(image_view_id);
            }
            swapchain.mark_image_views_managed_by_registry();

            let mut depth_buffer = DepthBuffer::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                swapchain.extent.width,
                swapchain.extent.height,
            )?;
            let depth_buffer_id = depth_buffer
                .register_with_registry(&resources)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register depth buffer: {e}"))
                })?;

            let mut render_pass = vulkan::RenderPass::builder(Arc::clone(&device.device))
                .with_swapchain_color(swapchain.format)
                .with_depth_attachment(
                    depth_buffer.format(),
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                )
                .build()?;
            let render_pass_id = resources
                .register_render_pass(render_pass.handle())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register render pass: {e}"))
                })?;
            render_pass.mark_managed_by_registry();

            let mut framebuffers = Vec::new();
            let mut framebuffer_ids = Vec::new();
            for (index, &image_view) in swapchain.image_views.iter().enumerate() {
                let attachments = [image_view, depth_buffer.view()];
                let framebuffer = vulkan::Framebuffer::new(
                    Arc::clone(&device.device),
                    render_pass.handle(),
                    &attachments,
                    swapchain.extent,
                )?;
                let framebuffer_id = resources
                    .register_framebuffer(
                        framebuffer.handle(),
                        &[
                            render_pass_id,
                            depth_buffer_id,
                            swapchain_image_view_ids[index],
                        ],
                    )
                    .map_err(|e| {
                        AshError::VulkanError(format!("Failed to register framebuffer: {e}"))
                    })?;
                let mut framebuffer = framebuffer;
                framebuffer.mark_managed_by_registry();
                framebuffers.push(framebuffer);
                framebuffer_ids.push(framebuffer_id);
            }

            let worker_count = thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);

            let command_manager = vulkan::CommandBufferManager::new(
                Arc::clone(&device.device),
                device.graphics_queue_family,
                worker_count,
            )?;

            let command_buffers =
                command_manager.allocate_primary_buffers(framebuffers.len() as u32)?;

            let mut frame_syncs = Vec::with_capacity(framebuffers.len());
            let mut frame_sync_ids = Vec::with_capacity(framebuffers.len());
            for _ in 0..framebuffers.len() {
                let mut sync = vulkan::FrameSync::new(Arc::clone(&device.device))?;
                let image_available_id = resources
                    .register_semaphore(sync.image_available)
                    .map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to register image-available semaphore: {e}"
                        ))
                    })?;
                let render_finished_id = resources
                    .register_semaphore(sync.render_finished)
                    .map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to register render-finished semaphore: {e}"
                        ))
                    })?;
                let fence_id = resources.register_fence(sync.in_flight).map_err(|e| {
                    AshError::VulkanError(format!("Failed to register in-flight fence: {e}"))
                })?;
                sync.mark_managed_by_registry();
                frame_syncs.push(sync);
                frame_sync_ids.push((image_available_id, render_finished_id, fence_id));
            }

            resources
                .register_command_pool(command_manager.upload_command_pool_handle())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register command pool: {e}"))
                })?;
            command_manager.mark_pool_managed_by_registry();

            let mut model_renderer =
                ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device.device));

            // Initialize uniform buffers with double buffering.
            let mut uniform_buffers = Vec::with_capacity(framebuffers.len());
            let aspect = swapchain.extent.width as f32 / swapchain.extent.height as f32;

            for _ in 0..framebuffers.len() {
                let mut buffer =
                    UniformBuffer::new(Arc::clone(&alloc), Arc::clone(&device.device))?;

                {
                    let matrices = buffer.matrices_mut();
                    matrices.set_view(
                        glam::Vec3::new(0.0, 2.0, 5.0),
                        glam::Vec3::new(0.0, 0.0, 0.0),
                        glam::Vec3::new(0.0, 1.0, 0.0),
                    );
                    // Use 0.5 near plane by default
                    matrices.set_projection(std::f32::consts::PI / 4.0, aspect, 0.5, 1000.0);
                }
                buffer.update()?;
                uniform_buffers.push(buffer);
            }

            // Create descriptor manager and pipeline layout
            let default_texture_data = TextureData::solid_color([255, 255, 255, 255]);
            let default_texture = Texture::from_data(
                Arc::clone(&alloc),
                Arc::clone(&device.device),
                command_manager.upload_command_pool_handle(),
                device.graphics_queue,
                &default_texture_data,
                vk::Format::R8G8B8A8_SRGB,
                Some("default_texture"),
            )?;

            let material = Material::default();

            let mut material_buffers = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                let mut material_buffer =
                    MaterialBuffer::new(Arc::clone(&alloc), Arc::clone(&device.device))?;
                {
                    let uniform = material_buffer.uniform_mut();
                    uniform.set_base_color_factor(Vec4::from_array(material.color));
                    uniform.set_emissive_factor(Vec4::from_array(material.emissive));
                    uniform.set_metallic_roughness(material.metallic, material.roughness);
                    uniform.set_occlusion_strength(material.occlusion_strength);
                    uniform.set_normal_scale(material.normal_scale);
                    uniform.set_normal_scale(material.normal_scale);
                    // uniform.set_texture_flags(...) removed

                    uniform.set_alpha_cutoff(0.1);
                }
                material_buffer.update()?;
                material_buffers.push(Mutex::new(material_buffer));
            }

            let mut descriptor_manager = vulkan::DescriptorManager::new(
                Arc::clone(&device.device),
                framebuffers.len() as u32,
                worker_count as u32,
                Some(Arc::clone(&resources)),
            )?;

            let mut bindless_manager = crate::vulkan::BindlessManager::new(
                Arc::clone(&device.device),
                descriptor_manager.allocator_mut(),
                1024 * 4,
            )?;

            let buffer_size =
                std::mem::size_of::<crate::renderer::resources::uniform::MvpMatrices>()
                    as vk::DeviceSize;
            for set_index in 0..descriptor_manager.frame_set_count() {
                if let Some(ubo) = uniform_buffers.get(set_index) {
                    descriptor_manager.bind_frame_uniform(set_index, ubo.buffer, buffer_size)?;
                }
            }

            let material_size = std::mem::size_of::<
                crate::renderer::resources::uniform::MaterialUniform,
            >() as vk::DeviceSize;
            for (worker_index, buffer) in material_buffers.iter().enumerate() {
                let buffer = buffer.lock();
                descriptor_manager.bind_material_uniform(
                    worker_index as u32,
                    buffer.buffer,
                    material_size,
                )?;
            }

            // Default texture binding removed

            // Register default texture as a fallback for all slots.
            let default_tex_index = bindless_manager
                .add_sampled_image(default_texture.view(), default_texture.sampler())
                .unwrap_or(0); // Fallback to index 0 if registration fails.
            log::info!("Registered default texture at bindless index {default_tex_index}");

            // Bindless architecture - legacy texture binding is bypassed.
            // descriptor_manager.bind_material_textures(...) removed

            validate_worker_resources(
                worker_count,
                descriptor_manager.material_set_count(),
                material_buffers.len(),
            )?;

            // Forward+ lighting integration
            let mut forward_plus =
                ForwardPlusIntegration::new(Arc::clone(&device.device), &alloc.vma)?;
            forward_plus.init(&alloc.vma);
            forward_plus.on_resize(swapchain.extent.width, swapchain.extent.height);

            let set_layouts = [
                descriptor_manager.frame_layout(),
                descriptor_manager.material_layout(),
                bindless_manager.layout(), // Set 2: Bindless textures
                descriptor_manager.shadow_layout(), // Set 3: Shadow map sampler
                forward_plus.layout(),     // Set 4: Forward+ lights
            ];
            let mesh_push_size = std::mem::size_of::<MeshPushConstants>() as u32;
            let material_push_size = std::mem::size_of::<MaterialPushConstants>() as u32;
            let push_constant_ranges = [
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: mesh_push_size,
                },
                vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    offset: mesh_push_size,
                    size: material_push_size,
                },
            ];

            let mut pipeline_layout_builder =
                vulkan::PipelineLayout::builder(Arc::clone(&device.device));
            for layout in &set_layouts {
                pipeline_layout_builder = pipeline_layout_builder.add_set_layout(*layout);
            }
            for range in &push_constant_ranges {
                pipeline_layout_builder = pipeline_layout_builder.add_push_constant(*range);
            }
            let mut pipeline_layout = pipeline_layout_builder.build()?;
            let pipeline_layout_id = resources
                .register_pipeline_layout(pipeline_layout.handle())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register pipeline layout: {e}"))
                })?;
            pipeline_layout.mark_managed_by_registry();

            let mut pipeline_builder = vulkan::Pipeline::builder(Arc::clone(&device.device))
                .with_layout(pipeline_layout.handle())
                .with_render_pass(render_pass.handle())
                .with_extent(swapchain.extent)
                .with_pipeline_cache(pipeline_cache.handle())
                .with_depth_format(depth_buffer.format())
                .with_cull_mode(vk::CullModeFlags::BACK)
                .with_multisampling(pipeline_cfg.multisample_config());

            for specialization in &pipeline_cfg.specialization_constants {
                pipeline_builder = pipeline_builder.with_specialization_bytes(
                    specialization.stage,
                    specialization.constant_id,
                    specialization.bytes(),
                );
            }

            pipeline_builder = pipeline_builder
                .add_shader_from_bytes(
                    include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv")),
                    vk::ShaderStageFlags::VERTEX,
                    "main",
                )?
                .add_shader_from_bytes(
                    include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv")),
                    vk::ShaderStageFlags::FRAGMENT,
                    "main",
                )?;

            let mut pipeline = pipeline_builder.build()?;
            let pipeline_id = resources
                .register_pipeline(pipeline.pipeline, &[pipeline_layout_id, render_pass_id])
                .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;
            pipeline.mark_managed_by_registry();

            // Create Shadow Pipeline
            let (shadow_pipeline, shadow_pipeline_layout) =
                if let Some(shadow_map) = shadow_feature.shadow_map() {
                    let shadow_push_range = vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::VERTEX,
                        offset: 0,
                        size: 128, // mat4 lightSpace + mat4 model
                    };

                    let shadow_push_range_frag = vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        offset: 128,
                        size: 4, // int base_color_index
                    };

                    let shadow_pipeline_layout =
                        vulkan::PipelineLayout::builder(Arc::clone(&device.device))
                            .add_push_constant(shadow_push_range)
                            .add_push_constant(shadow_push_range_frag)
                            .add_set_layout(bindless_manager.layout()) // Set 2: Bindless textures
                            .build()?;

                    let shadow_builder = vulkan::Pipeline::builder(Arc::clone(&device.device))
                        .with_layout(shadow_pipeline_layout.handle())
                        .with_render_pass(shadow_map.render_pass)
                        .with_extent(vk::Extent2D {
                            width: shadow_map.res,
                            height: shadow_map.res,
                        })
                        .with_pipeline_cache(pipeline_cache.handle())
                        .with_depth_format(vk::Format::D32_SFLOAT)
                        .with_cull_mode(vk::CullModeFlags::FRONT)
                        .add_shader_from_bytes(
                            include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv")),
                            vk::ShaderStageFlags::VERTEX,
                            "main",
                        )?
                        .add_shader_from_bytes(
                            include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv")),
                            vk::ShaderStageFlags::FRAGMENT,
                            "main",
                        )?;

                    let shadow_pipeline = shadow_builder.build()?;
                    (Some(shadow_pipeline), Some(shadow_pipeline_layout))
                } else {
                    (None, None)
                };

            let mut mesh = Mesh::create_cube();
            log::trace!("Ensuring cube mesh textures...");
            mesh.ensure_texture(
                Arc::clone(&alloc),
                Arc::clone(&device.device),
                command_manager.upload_command_pool_handle(),
                device.graphics_queue,
            )?;
            log::trace!("Cube mesh textures ready, registering with model renderer...");
            model_renderer.ensure_mesh(
                &mesh.name,
                &mesh,
                command_manager.upload_command_pool_handle(),
                device.graphics_queue,
            )?;
            log::trace!("Cube mesh registered successfully");

            let material = Material::default();
            let transform = Transform::identity();
            let transform_matrix = transform.model_matrix();
            let mut mesh_registry = HashMap::new();
            mesh_registry.insert(0, mesh.name.clone());
            let mut material_registry = HashMap::new();
            material_registry.insert(0, material.clone());
            let mut mesh_texture_flags = HashMap::new();

            let initial_flags = TexturePresenceFlags::from_mesh(&mesh);

            // Register mesh textures with bindless manager
            if let Some(tex) = mesh.texture.as_ref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.texture_index = Some(idx);
            }
            if let Some(tex) = mesh.normal_texture.as_ref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.normal_texture_index = Some(idx);
            }
            if let Some(tex) = mesh.metallic_roughness_texture.as_ref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.metallic_roughness_texture_index = Some(idx);
            }
            if let Some(tex) = mesh.occlusion_texture.as_ref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.occlusion_texture_index = Some(idx);
            }
            if let Some(tex) = mesh.emissive_texture.as_ref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.emissive_texture_index = Some(idx);
            }

            // Legacy map usage is removed in favor of bindless.
            mesh_texture_flags.insert(mesh.name.clone(), initial_flags);
            let start_time = Instant::now();

            let swapchain_extent = swapchain.extent;

            let gbuffer = GBuffer::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                swapchain_extent.width,
                swapchain_extent.height,
            )?;

            Ok(Self {
                buffer_pool,
                resources,
                features,
                _pipeline_cache: pipeline_cache,
                cmds: command_manager,
                worker_count,
                command_buffers,
                frame_syncs,
                current_frame: 0,
                _default_texture: default_texture,
                model_renderer,
                draw_items: vec![DrawItem {
                    key: mesh.name.clone(),
                    transform: transform_matrix,
                    material: material.clone(),
                    texture_flags: initial_flags,
                    texture_indices: [
                        mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                        mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                        mesh.metallic_roughness_texture_index
                            .map(|i| i as i32)
                            .unwrap_or(-1),
                        mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
                    ],
                    emissive_index: mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1),
                }],
                swapchain: Some(swapchain),
                render_pass: Some(render_pass),
                render_pass_id: Some(render_pass_id),
                pipeline: Some(pipeline),
                pipeline_id: Some(pipeline_id),
                depth_buffer: Some(depth_buffer),
                mesh: Some(mesh),
                material,
                transform,
                uniform_buffers,
                material_buffers,
                pipeline_layout: Some(pipeline_layout),
                pipeline_layout_id: Some(pipeline_layout_id),
                descriptors: Some(descriptor_manager),
                framebuffers,
                framebuffer_ids,
                start_time,
                alloc,
                device,
                mesh_registry,
                mesh_indices_registry: HashMap::new(),
                mesh_texture_flags,
                material_registry,
                swapchain_image_view_ids,
                depth_buffer_id: Some(depth_buffer_id),
                frame_sync_ids,
                old_swapchain_handles: Vec::new(),
                swapchain_cleanup_pending: false,
                resize_pending: false,
                pending_extent: Some(swapchain_extent),
                // Post-processing defaults
                msaa_preset: MsaaPreset::Off,
                hdr_framebuffer: None,
                fullscreen_pass: None,
                tonemapping_enabled: true,
                tonemapping_exposure: 1.0,
                tonemapping_gamma: 2.2,
                bloom_enabled: false,
                bloom_intensity: 0.5,
                // Diagnostics
                diagnostics: DiagnosticsState::default(),
                frame_profiler: FrameProfiler::new(),
                gpu_profiler: None, // Initialized when diagnostics are enabled.
                diagnostics_overlay: DiagnosticsOverlay::new(),
                shadow_feature,
                shadow_pipeline,
                shadow_pipeline_layout,
                bindless_manager: Some(bindless_manager),
                forward_plus: Some(forward_plus),
                hiz_pass: None,
                indirect_draw_pass: None,
                occlusion_culling: OcclusionCulling::new(),
                vsr_pass: None,
                ssgi_pass: None,
                gbuffer: Some(gbuffer),
                post_descriptor_pool: vk::DescriptorPool::null(),
                post_descriptor_sets: Vec::new(),
                post_pipeline: None,
                post_framebuffers: Vec::new(),
            })
        }
    }

    fn worker_index_for_frame(&self, frame_index: usize) -> usize {
        compute_worker_index(self.worker_count, frame_index)
    }

    fn render_post_processing(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        if !self.tonemapping_enabled
            || self.fullscreen_pass.is_none()
            || self.post_pipeline.is_none()
            || self.post_framebuffers.is_empty()
            || self.post_descriptor_sets.is_empty()
        {
            return Ok(());
        }

        let pass = self.fullscreen_pass.as_ref().unwrap();
        let pipeline = self.post_pipeline.unwrap();
        let framebuffer = &self.post_framebuffers[image_index];
        let descriptor_set = self.post_descriptor_sets[image_index];
        let extent = self.swapchain.as_ref().unwrap().extent;

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(pass.render_pass())
            .framebuffer(framebuffer.handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            self.device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            self.device.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pass.pipeline_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let push_constants = fullscreen_pass::PostProcessPushConstants {
                exposure: self.tonemapping_exposure,
                gamma: self.tonemapping_gamma,
                bloom_intensity: if self.bloom_enabled {
                    self.bloom_intensity
                } else {
                    0.0
                },
                _padding: 0.0,
            };

            self.device.device.cmd_push_constants(
                command_buffer,
                pass.pipeline_layout(),
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            // Draw 3 vertices for a single fullscreen triangle
            self.device.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            self.device.device.cmd_end_render_pass(command_buffer);
        }

        Ok(())
    }
    /// Set mesh to render
    pub fn set_mesh(&mut self, mut mesh: Mesh) {
        unsafe {
            let upload_pool = self.cmds.upload_command_pool_handle();
            let key = mesh.name.clone();
            if let Err(e) = self.model_renderer.ensure_mesh(
                &key,
                &mesh,
                upload_pool,
                self.device.graphics_queue,
            ) {
                log::error!("Failed to upload mesh via ModelRenderer: {e}");
                return;
            }

            if let Err(e) = mesh.ensure_texture(
                Arc::clone(&self.alloc),
                Arc::clone(&self.device.device),
                upload_pool,
                self.device.graphics_queue,
            ) {
                log::error!("Failed to ensure mesh texture: {e}");
            }

            // Register textures with the bindless manager.
            if let Some(bindless_manager) = self.bindless_manager.as_mut() {
                if let Some(tex) = mesh.texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register base_color texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.normal_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.normal_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register normal texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.metallic_roughness_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.metallic_roughness_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register metallic_roughness texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.occlusion_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.occlusion_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register occlusion texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.emissive_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.emissive_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register emissive texture: {e}"),
                    }
                }
            }

            let flags = TexturePresenceFlags::from_mesh(&mesh);
            self.mesh_texture_flags.clear();
            self.mesh_texture_flags.insert(key.clone(), flags);

            let indices = [
                mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.metallic_roughness_texture_index
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
            ];
            let emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

            self.mesh_indices_registry
                .insert(key.clone(), (indices, emissive_index));

            self.draw_items.clear();
            self.draw_items.push(DrawItem {
                key: key.clone(),
                transform: self.transform.model_matrix(),
                material: self.material.clone(),
                texture_flags: flags,
                texture_indices: indices,
                emissive_index,
            });

            self.mesh_registry.clear();
            self.mesh_registry.insert(0, key.clone());
            self.material_registry.insert(0, self.material.clone());
            self.mesh = Some(mesh);
        }
    }

    pub fn register_mesh_handle(&mut self, handle: u32, mesh: &mut Mesh) -> Result<()> {
        unsafe {
            let key = mesh.name.clone();
            let upload_pool = self.cmds.upload_command_pool_handle();
            mesh.ensure_texture(
                Arc::clone(&self.alloc),
                Arc::clone(&self.device.device),
                upload_pool,
                self.device.graphics_queue,
            )?;

            self.model_renderer
                .ensure_mesh(&key, mesh, upload_pool, self.device.graphics_queue)?;

            // Register textures with bindless manager
            if let Some(bindless_manager) = self.bindless_manager.as_mut() {
                if let Some(tex) = mesh.texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register base_color texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.normal_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.normal_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register normal texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.metallic_roughness_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.metallic_roughness_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register metallic_roughness texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.occlusion_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.occlusion_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register occlusion texture: {e}"),
                    }
                }
                if let Some(tex) = mesh.emissive_texture.as_ref() {
                    match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
                        Ok(idx) => mesh.emissive_texture_index = Some(idx),
                        Err(e) => log::error!("Failed to register emissive texture: {e}"),
                    }
                }
            }

            let flags = TexturePresenceFlags::from_mesh(mesh);

            // Store indices for bindless texture mapping.
            let indices = [
                mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.metallic_roughness_texture_index
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
            ];
            let emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

            self.mesh_indices_registry
                .insert(key.clone(), (indices, emissive_index));
            self.mesh_texture_flags.insert(key.clone(), flags);

            self.mesh_registry.insert(handle, key);
        }

        Ok(())
    }

    pub fn register_material_handle(&mut self, handle: u32, material: &Material) {
        self.material_registry.insert(handle, material.clone());
    }

    /// Registers mesh data described by a [`MeshDescriptor`] with the renderer and returns the
    /// internal key used for lookup.
    pub fn register_mesh_descriptor(
        &mut self,
        handle: u32,
        descriptor: &MeshDescriptor,
    ) -> Result<String> {
        let mut mesh = Mesh::from_descriptor(descriptor);
        let key = mesh.name.clone();

        self.register_mesh_handle(handle, &mut mesh)?;

        Ok(key)
    }

    /// Converts a material descriptor into a renderer material and registers it.
    pub fn register_material_descriptor(
        &mut self,
        handle: u32,
        descriptor: &MaterialDescriptor,
    ) -> Material {
        let material = descriptor.material.clone();
        self.register_material_handle(handle, &material);
        material
    }

    /// Submit render commands for the current frame.
    ///
    /// Each `RenderCommand` specifies a mesh handle, material handle, and transform.
    pub fn submit_render_commands(&mut self, commands: &[RenderCommand]) {
        self.draw_items.clear();

        for command in commands {
            if let Some(mesh_key) = self.mesh_registry.get(&command.mesh_handle) {
                if let Some(material) = self.material_registry.get(&command.material_handle) {
                    let texture_flags = self
                        .mesh_texture_flags
                        .get(mesh_key)
                        .copied()
                        .unwrap_or_default();

                    let (indices, emissive) = self
                        .mesh_indices_registry
                        .get(mesh_key)
                        .cloned()
                        .unwrap_or(([-1, -1, -1, -1], -1));

                    self.draw_items.push(DrawItem {
                        key: mesh_key.clone(),
                        transform: command.transform,
                        material: material.clone(),
                        texture_flags,
                        texture_indices: indices,
                        emissive_index: emissive,
                    });
                }
            }
        }

        // Single mesh fallback
        if self.draw_items.is_empty() {
            if let Some(mesh) = self.mesh.as_ref() {
                let texture_flags = self
                    .mesh_texture_flags
                    .get(&mesh.name)
                    .copied()
                    .unwrap_or_default();
                self.draw_items.push(DrawItem {
                    key: mesh.name.clone(),
                    transform: self.transform.model_matrix(),
                    material: self.material.clone(),
                    texture_flags,
                    texture_indices: [
                        mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                        mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                        mesh.metallic_roughness_texture_index
                            .map(|i| i as i32)
                            .unwrap_or(-1),
                        mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
                    ],
                    emissive_index: mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1),
                });
            }
        }
    }

    pub fn request_swapchain_resize(&mut self, new_extent: vk::Extent2D) {
        self.pending_extent = Some(new_extent);
        if !self.resize_pending {
            log::info!(
                "Swapchain resize requested: {}x{}",
                new_extent.width,
                new_extent.height
            );
            // Synchronize with device to prevent resource conflicts during resize.
            unsafe {
                let _ = self.device.device.device_wait_idle();
            }
        }
        self.resize_pending = true;
    }

    fn resize_if_needed(&mut self) -> Result<()> {
        if !self.resize_pending {
            return Ok(());
        }

        if let Some(extent) = self.pending_extent {
            if extent.width == 0 || extent.height == 0 {
                // Window minimized; await valid swapchain extent.
                return Ok(());
            }
        }

        log::info!("Recreating swapchain and dependent resources");

        self.wait_for_inflight_frames()?;

        self.recreate_swapchain_resources()?;

        self.resize_pending = false;
        if let Some(swapchain) = self.swapchain.as_ref() {
            self.pending_extent = Some(swapchain.extent);
        }

        Ok(())
    }

    fn wait_for_inflight_frames(&self) -> Result<()> {
        for sync in &self.frame_syncs {
            sync.wait()?;
        }
        Ok(())
    }

    fn defer_old_swapchain(&mut self, handle: vk::SwapchainKHR) {
        if handle == vk::SwapchainKHR::null() {
            return;
        }
        self.old_swapchain_handles.push(handle);
        self.swapchain_cleanup_pending = true;
    }

    fn flush_old_swapchains(&mut self) {
        if self.old_swapchain_handles.is_empty() {
            self.swapchain_cleanup_pending = false;
            return;
        }

        if let Some(ref swapchain) = self.swapchain {
            for handle in self.old_swapchain_handles.drain(..) {
                unsafe {
                    swapchain.destroy_swapchain_handle(handle);
                }
            }
        } else {
            self.old_swapchain_handles.clear();
        }

        self.swapchain_cleanup_pending = false;
    }

    fn recreate_swapchain_resources(&mut self) -> Result<()> {
        // Paranoid Validation for enterprise reliability.
        // We cannot proceed with swapchain recreation if surfaces are zero-dimensioned.
        if let Some(extent) = self.pending_extent {
            if extent.width == 0 || extent.height == 0 {
                return Err(AshError::VulkanError(
                    "Cannot recreate swapchain with zero dimensions".into(),
                ));
            }
        }

        let old_swapchain = unsafe {
            if let Some(ref mut swapchain) = self.swapchain {
                Some(swapchain.recreate(&self.device)?)
            } else {
                self.swapchain = Some(vulkan::SwapchainWrapper::new(&self.device)?);
                None
            }
        };

        if let Some(handle) = old_swapchain {
            self.defer_old_swapchain(handle);
        }

        let (swapchain_extent, swapchain_format, image_views, image_count) = {
            let swapchain = self.swapchain.as_ref().ok_or_else(|| {
                AshError::VulkanError("Swapchain unavailable after recreation".into())
            })?;
            (
                swapchain.extent,
                swapchain.format,
                swapchain.image_views.clone(),
                swapchain.images.len(),
            )
        };

        // Cleanup resources in dependency order (dependents first).
        // 1. Destroy pipeline.
        self.cleanup_pipeline();
        // 2. Destroy framebuffers.
        self.cleanup_framebuffers();
        // 3. Destroy render pass.
        self.cleanup_render_pass();
        // 4. Update image views.
        self.update_image_views(&image_views)?;
        // 5. Recreate depth buffer.
        self.recreate_depth_buffer(swapchain_extent)?;
        // 5b. Recreate G-Buffer.
        self.recreate_gbuffer(swapchain_extent)?;
        // 5c. Recreate SSGI pass
        self.recreate_ssgi_pass(swapchain_extent)?;
        // 6. Create new render pass and framebuffers.
        self.create_render_pass_and_framebuffers(swapchain_extent, swapchain_format, &image_views)?;

        self.recreate_frame_syncs(self.framebuffers.len())?;
        self.recreate_command_buffers()?;
        self.recreate_uniform_buffers(self.framebuffers.len())?;
        self.recreate_vsr_pass(self.swapchain.as_ref().unwrap().extent)?;
        self.recreate_descriptor_sets()?;
        // 7. Recreate pipeline.
        self.recreate_pipeline()?;

        log::info!("Swapchain recreation complete ({image_count} images)");
        Ok(())
    }

    fn cleanup_framebuffers(&mut self) {
        for (framebuffer, id) in self
            .framebuffers
            .drain(..)
            .zip(self.framebuffer_ids.drain(..))
        {
            drop(framebuffer);
            if let Err(e) = self.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup framebuffer {id}: {e}");
            }
        }
    }

    fn cleanup_render_pass(&mut self) {
        if let Some(render_pass_id) = self.render_pass_id.take() {
            if let Err(e) = self.resources.cleanup_resource(render_pass_id) {
                log::warn!("Failed to cleanup render pass: {e}");
            }
        }
    }

    fn cleanup_pipeline(&mut self) {
        if let Some(pipeline_id) = self.pipeline_id.take() {
            if let Err(e) = self.resources.cleanup_resource(pipeline_id) {
                log::warn!("Failed to cleanup pipeline: {e}");
            }
        }
        self.pipeline = None;
    }

    fn recreate_pipeline(&mut self) -> Result<()> {
        log::info!("Recompiling pipeline due to shader change...");
        let layout = self.pipeline_layout.as_ref().unwrap().handle();
        let render_pass = self.render_pass.as_ref().unwrap().handle();
        let extent = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain missing".into()))?
            .extent;
        let cache = self._pipeline_cache.handle();
        let depth_format = self
            .depth_buffer
            .as_ref()
            .ok_or(AshError::VulkanError("Depth buffer missing".into()))?
            .format();

        let multisample_config = vulkan::MultisampleConfig {
            sample_count: self.msaa_preset.sample_count(),
            enable_sample_shading: false,
            min_sample_shading: 0.0,
        };

        let mut builder = vulkan::Pipeline::builder(Arc::clone(&self.device.device))
            .with_layout(layout)
            .with_render_pass(render_pass)
            .with_extent(extent)
            .with_pipeline_cache(cache)
            .with_depth_format(depth_format)
            .with_cull_mode(vk::CullModeFlags::BACK)
            .with_multisampling(multisample_config);

        if self.gbuffer.is_some() {
            let blend_attachments = vec![
                // Index 0: Swapchain Color (with blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::TRUE,
                    src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                    dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                    color_blend_op: vk::BlendOp::ADD,
                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                    alpha_blend_op: vk::BlendOp::ADD,
                },
                // Index 1: Normals (no blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                // Index 2: Albedo (no blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                // Index 3: Motion Vectors (no blending)
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
            ];
            builder = builder.with_color_blend_attachments(blend_attachments);
        }

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;
        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

        let mut new_pipeline = builder.build()?;
        let pipeline_layout_id = self.pipeline_layout_id.ok_or_else(|| {
            AshError::VulkanError("Pipeline layout ID missing during recreation".into())
        })?;
        let render_pass_id = self.render_pass_id.ok_or_else(|| {
            AshError::VulkanError("Render pass ID missing during recreation".into())
        })?;

        let pipeline_id = self
            .resources
            .register_pipeline(new_pipeline.pipeline, &[pipeline_layout_id, render_pass_id])
            .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;

        new_pipeline.mark_managed_by_registry();
        self.pipeline = Some(new_pipeline);
        self.pipeline_id = Some(pipeline_id);

        log::info!("Pipeline recompiled successfully!");
        Ok(())
    }

    fn update_image_views(&mut self, image_views: &[vk::ImageView]) -> Result<()> {
        for id in self.swapchain_image_view_ids.drain(..) {
            if let Err(e) = self.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old swapchain image view {id}: {e}");
            }
        }

        self.swapchain_image_view_ids.clear();
        for &view in image_views {
            let id = self.resources.register_image_view(view).map_err(|e| {
                AshError::VulkanError(format!("Failed to register swapchain image view: {e}"))
            })?;
            self.swapchain_image_view_ids.push(id);
        }

        if let Some(ref mut sc) = self.swapchain {
            sc.mark_image_views_managed_by_registry();
        }

        Ok(())
    }

    fn recreate_depth_buffer(&mut self, extent: vk::Extent2D) -> Result<()> {
        if let Some(id) = self.depth_buffer_id.take() {
            if let Err(e) = self.resources.cleanup_resource(id) {
                log::warn!("Failed to cleanup old depth buffer: {e}");
            }
        }

        let mut depth_buffer = unsafe {
            DepthBuffer::new(
                Arc::clone(&self.device.device),
                Arc::clone(&self.alloc),
                extent.width,
                extent.height,
            )?
        };

        let depth_buffer_id = depth_buffer
            .register_with_registry(&self.resources)
            .map_err(|e| AshError::VulkanError(format!("Failed to register depth buffer: {e}")))?;

        self.depth_buffer = Some(depth_buffer);
        self.depth_buffer_id = Some(depth_buffer_id);

        Ok(())
    }

    fn recreate_gbuffer(&mut self, extent: vk::Extent2D) -> Result<()> {
        self.gbuffer = Some(unsafe {
            GBuffer::new(
                Arc::clone(&self.device.device),
                Arc::clone(&self.alloc),
                extent.width,
                extent.height,
            )?
        });
        Ok(())
    }

    fn recreate_ssgi_pass(&mut self, extent: vk::Extent2D) -> Result<()> {
        if let Some(ref mut ssgi) = self.ssgi_pass {
            unsafe {
                ssgi.destroy(&self.alloc.vma);
                ssgi.init(
                    &self.alloc.vma,
                    &self.device,
                    extent.width,
                    extent.height,
                    ssgi.quality(),
                );
            }
        }
        Ok(())
    }

    fn recreate_vsr_pass(&mut self, display_extent: vk::Extent2D) -> Result<()> {
        if let Some(ref mut vsr) = self.vsr_pass {
            unsafe {
                vsr.destroy(&self.alloc.vma);
                vsr.init(
                    &self.alloc.vma,
                    &self.device,
                    display_extent.width,
                    display_extent.height,
                    vsr.quality(),
                );
            }
        }
        Ok(())
    }

    fn create_render_pass_and_framebuffers(
        &mut self,
        extent: vk::Extent2D,
        color_format: vk::Format,
        image_views: &[vk::ImageView],
    ) -> Result<()> {
        // Cleanup already done by cleanup_framebuffers() and cleanup_render_pass()

        let depth_buffer = self.depth_buffer.as_ref().ok_or_else(|| {
            AshError::VulkanError("Depth buffer missing when rebuilding framebuffers".into())
        })?;

        let mut builder = vulkan::RenderPass::builder(Arc::clone(&self.device.device));

        let render_to_hdr = self.hdr_framebuffer.is_some();
        if render_to_hdr {
            let hdr = self.hdr_framebuffer.as_ref().unwrap();
            builder = builder
                .with_color_attachment(hdr.format(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        } else {
            builder = builder.with_swapchain_color(color_format);
        }

        if self.gbuffer.is_some() {
            // Index 1: Normals (RGBA16F)
            builder = builder.with_color_attachment(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            // Index 2: Albedo (RGBA8)
            builder = builder.with_color_attachment(
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            // Index 3: Motion Vectors (RG16F)
            builder = builder.with_color_attachment(
                vk::Format::R16G16_SFLOAT,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }

        let mut render_pass = builder
            .with_depth_attachment(
                depth_buffer.format(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
            .build()?;

        let render_pass_id = self
            .resources
            .register_render_pass(render_pass.handle())
            .map_err(|e| AshError::VulkanError(format!("Failed to register render pass: {e}")))?;
        render_pass.mark_managed_by_registry();
        self.render_pass = Some(render_pass);
        self.render_pass_id = Some(render_pass_id);

        let depth_buffer_id = self.depth_buffer_id.ok_or_else(|| {
            AshError::VulkanError("Depth buffer id missing while rebuilding framebuffers".into())
        })?;

        let mut framebuffers = Vec::with_capacity(image_views.len());
        let mut framebuffer_ids = Vec::with_capacity(image_views.len());

        let gbuffer = self.gbuffer.as_ref().ok_or_else(|| {
            AshError::VulkanError("G-Buffer missing when rebuilding framebuffers".into())
        })?;

        for (index, &view) in image_views.iter().enumerate() {
            // If rendering to HDR, we use a single HDR view for all framebuffers.
            // Otherwise we use the per-swapchain view.
            let color_view = if render_to_hdr {
                self.hdr_framebuffer.as_ref().unwrap().view()
            } else {
                view
            };

            // Order must match RenderPass: [Color, GBufferNormal, GBufferAlbedo, GBufferMotion, Depth]
            let attachments = [
                color_view,
                gbuffer.normal_view(),
                gbuffer.albedo_view(),
                gbuffer.motion_view(),
                depth_buffer.view(),
            ];
            let framebuffer = vulkan::Framebuffer::new(
                Arc::clone(&self.device.device),
                self.render_pass
                    .as_ref()
                    .expect("render pass just created")
                    .handle(),
                &attachments,
                extent,
            )?;

            let deps = if render_to_hdr {
                vec![render_pass_id, depth_buffer_id]
            } else {
                vec![
                    render_pass_id,
                    depth_buffer_id,
                    self.swapchain_image_view_ids[index],
                ]
            };

            let framebuffer_id = self
                .resources
                .register_framebuffer(framebuffer.handle(), &deps)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register framebuffer: {e}"))
                })?;

            let mut framebuffer = framebuffer;
            framebuffer.mark_managed_by_registry();
            framebuffers.push(framebuffer);
            framebuffer_ids.push(framebuffer_id);
        }

        self.framebuffers = framebuffers;
        self.framebuffer_ids = framebuffer_ids;

        // --- Post-Processing Framebuffers ---
        if let Some(ref pass) = self.fullscreen_pass {
            let mut post_framebuffers = Vec::with_capacity(image_views.len());
            for &view in image_views {
                let framebuffer = vulkan::Framebuffer::new(
                    Arc::clone(&self.device.device),
                    pass.render_pass(),
                    &[view],
                    extent,
                )?;
                post_framebuffers.push(framebuffer);
            }
            self.post_framebuffers = post_framebuffers;
        }

        Ok(())
    }

    fn recreate_frame_syncs(&mut self, count: usize) -> Result<()> {
        for (image_available_id, render_finished_id, fence_id) in self.frame_sync_ids.drain(..) {
            if let Err(e) = self.resources.cleanup_resource(image_available_id) {
                log::warn!("Failed to cleanup image-available semaphore: {e}");
            }
            if let Err(e) = self.resources.cleanup_resource(render_finished_id) {
                log::warn!("Failed to cleanup render-finished semaphore: {e}");
            }
            if let Err(e) = self.resources.cleanup_resource(fence_id) {
                log::warn!("Failed to cleanup in-flight fence: {e}");
            }
        }

        self.frame_syncs.clear();

        let mut frame_syncs = Vec::with_capacity(count);
        let mut frame_sync_ids = Vec::with_capacity(count);

        for _ in 0..count {
            let mut sync = vulkan::FrameSync::new(Arc::clone(&self.device.device))?;
            let image_available_id = self
                .resources
                .register_semaphore(sync.image_available)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to register image-available semaphore: {e}"
                    ))
                })?;
            let render_finished_id = self
                .resources
                .register_semaphore(sync.render_finished)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to register render-finished semaphore: {e}"
                    ))
                })?;
            let fence_id = self.resources.register_fence(sync.in_flight).map_err(|e| {
                AshError::VulkanError(format!("Failed to register in-flight fence: {e}"))
            })?;

            sync.mark_managed_by_registry();
            frame_syncs.push(sync);
            frame_sync_ids.push((image_available_id, render_finished_id, fence_id));
        }

        self.frame_syncs = frame_syncs;
        self.frame_sync_ids = frame_sync_ids;
        self.current_frame = 0;

        Ok(())
    }

    fn recreate_command_buffers(&mut self) -> Result<()> {
        self.cmds
            .reset_primary_pool(vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;

        self.command_buffers = self
            .cmds
            .allocate_primary_buffers(self.framebuffers.len() as u32)?;
        self.current_frame = 0;

        Ok(())
    }

    fn recreate_uniform_buffers(&mut self, count: usize) -> Result<()> {
        for ub in &mut self.uniform_buffers {
            let _ = ub.cleanup();
        }
        self.uniform_buffers.clear();

        unsafe {
            for _ in 0..count {
                let mut buffer =
                    UniformBuffer::new(Arc::clone(&self.alloc), Arc::clone(&self.device.device))?;

                {
                    // Initialize with identity matrices. Values are updated during render_frame.
                    let matrices = buffer.matrices_mut();
                    matrices.model = self.transform.model_matrix();
                    matrices.view = Mat4::IDENTITY;
                    matrices.projection = Mat4::IDENTITY;
                    matrices.view_proj = Mat4::IDENTITY;
                    matrices.camera_pos = glam::Vec4::W; // (0,0,0,1)
                }
                buffer.update()?;
                self.uniform_buffers.push(buffer);
            }
        }
        Ok(())
    }

    fn recreate_descriptor_sets(&mut self) -> Result<()> {
        if let Some(manager) = self.descriptors.as_mut() {
            manager.recreate_frame_sets(self.frame_syncs.len() as u32)?;

            let buffer_size =
                std::mem::size_of::<crate::renderer::resources::uniform::MvpMatrices>()
                    as vk::DeviceSize;
            for index in 0..manager.frame_set_count() {
                if let Some(ubo) = self.uniform_buffers.get(index) {
                    manager.bind_frame_uniform(index, ubo.buffer, buffer_size)?;
                }
            }
        }

        Ok(())
    }

    /// Render frame with the specified camera view.
    ///
    /// Arguments:
    /// - `view`: View matrix (camera look-at)
    /// - `projection`: Projection matrix (perspective/orthographic)
    /// - `camera_pos`: Camera world position (for lighting calculations)
    pub fn render_frame(
        &mut self,
        view: Mat4,
        projection: Mat4,
        camera_pos: glam::Vec3,
    ) -> Result<()> {
        self.flush_old_swapchains();

        // Recycle per-frame descriptor pools (static pools are unaffected)
        if let Some(dm) = self.descriptors.as_mut() {
            dm.next_frame();
        }

        // Hot-reload shaders if changed (throttled to every ~1 second)
        const SHADER_CHECK_INTERVAL: usize = 60;

        // Ensure mutable borrow of pipeline scope ends prior to recreation call.
        let shaders_changed = if self.current_frame % SHADER_CHECK_INTERVAL == 0 {
            if let Some(pipeline) = &mut self.pipeline {
                match pipeline.detect_shader_changes() {
                    Ok(changed) => changed,
                    Err(e) => {
                        log::warn!("Failed to check shader changes: {e}");
                        false
                    }
                }
            } else {
                false
            }
        } else {
            false
        };

        if shaders_changed {
            if let Err(e) = self.recreate_pipeline() {
                log::error!("Failed to recreate pipeline: {e}");
            }
        }

        self.resize_if_needed()?;
        if self.resize_pending {
            return Ok(());
        }

        unsafe {
            let swapchain_extent = self
                .swapchain
                .as_ref()
                .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
                .extent;
            let scene_pipeline = self
                .pipeline
                .as_ref()
                .map(|p| p.pipeline)
                .ok_or(AshError::VulkanError("Pipeline not available".to_string()))?;
            let main_render_pass =
                self.render_pass
                    .as_ref()
                    .map(|p| p.handle())
                    .ok_or(AshError::VulkanError(
                        "Render pass not available".to_string(),
                    ))?;

            // Fence synchronization prior to uniform buffer updates.
            // Ensure previous frame submission completes before writing to the uniform buffer.
            // Hot Path: Use unchecked access for frame-indexed resources.
            // SAFETY: frame_index is bounded by command_buffers.len() and frame_syncs.len()
            // which are established at initialization and swapchain recreation.
            let frame_index = self.current_frame;
            let command_buffer = *self.command_buffers.get_unchecked(frame_index);
            let frame_sync_ref = self.frame_syncs.get_unchecked(frame_index);

            let (image_available, render_finished, in_flight_fence) = (
                frame_sync_ref.image_available,
                frame_sync_ref.render_finished,
                frame_sync_ref.in_flight,
            );

            self.device
                .device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)?;
            self.device.device.reset_fences(&[in_flight_fence])?;

            // Prepare culling data for this frame
            self.occlusion_culling.begin_frame();
            for (i, item) in self.draw_items.iter().enumerate() {
                if let Some(uploaded) = self.model_renderer.get(&item.key) {
                    // Use mesh clusters for fine-grained culling
                    let bounds = CullBoundingBox::new(Vec3::ZERO, Vec3::ONE);
                    self.occlusion_culling.push_clusters(
                        bounds,
                        item.transform,
                        i as u32,
                        uploaded.clusters(),
                    );
                }
            }

            // Apply sub-pixel jitter for VSR/TSR if enabled
            let mut jittered_projection = projection;
            let mut jitter_uv = [0.0f32; 2];
            if let Some(ref mut vsr) = self.vsr_pass {
                let (jx, jy) = vsr.next_jitter();
                // Jitter is in pixels [-0.5, 0.5], convert to NDC/UV
                jitter_uv = [jx, jy];
                let extent = self.swapchain.as_ref().unwrap().extent;
                jittered_projection.col_mut(2).x += jx / extent.width as f32;
                jittered_projection.col_mut(2).y += jy / extent.height as f32;
            }

            // GPU synchronization confirmed; safe to update uniform buffer.
            {
                // SAFETY: frame_index validity verified by previous unchecked access logic.
                let uniform_buffer = self.uniform_buffers.get_unchecked_mut(frame_index);

                let elapsed = self.start_time.elapsed().as_secs_f32();
                let mut feature_ctx = FeatureFrameContext {
                    device: self.device.device.as_ref(),
                    descriptor_manager: self.descriptors.as_ref(),
                    transform: &mut self.transform,
                    auto_rotate: false, // Auto-rotate now handled by examples
                    elapsed_seconds: elapsed,
                };
                self.features.before_frame(&mut feature_ctx);

                // Matrices provided via function arguments.
                let matrices = uniform_buffer.matrices_mut();
                matrices.model = self.transform.model_matrix();
                matrices.view = view;
                matrices.projection = jittered_projection;
                matrices.view_proj = jittered_projection * view;
                matrices.camera_pos = camera_pos.extend(1.0);
                let light_dir = glam::Vec3::new(-0.35, -1.0, -0.25).normalize();

                matrices.set_lighting(light_dir, glam::Vec3::splat(1.5), glam::Vec3::splat(0.35));

                // Set light-space matrix for shadow mapping
                let light_space_matrix = self.shadow_feature.light_space_matrix();
                matrices.set_light_space_matrix(light_space_matrix);
                matrices.normal_matrix = matrices.model.inverse().transpose();

                uniform_buffer.update()?;
            }

            // Update post-processing descriptors once per frame to ensure they point to the correct VSR output.
            if self.tonemapping_enabled {
                self.update_post_descriptors()?;
            }

            let cmd_ctx = self.cmds.context(command_buffer);
            cmd_ctx.reset()?;

            let acquire_result = {
                let swapchain_ref = self
                    .swapchain
                    .as_ref()
                    .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?;
                swapchain_ref.acquire_next_image(image_available)
            };
            let image_index = match acquire_result {
                Ok(index) => index,
                Err(AshError::SwapchainOutOfDate(_)) => {
                    self.request_swapchain_resize(swapchain_extent);
                    return Ok(());
                }
                Err(err) => return Err(err),
            };

            let worker_index = self.worker_index_for_frame(frame_index);
            debug_assert!(
                worker_index < self.worker_count.max(1),
                "worker index {} out of bounds for {} workers",
                worker_index,
                self.worker_count
            );
            debug_assert_eq!(
                self.worker_count,
                self.material_buffers.len(),
                "material buffer pool must match worker count"
            );

            cmd_ctx.begin(vk::CommandBufferUsageFlags::empty())?;
            // Shadow Pass
            if let (Some(shadow_pipeline), Some(shadow_layout)) = (
                self.shadow_pipeline.as_ref(),
                self.shadow_pipeline_layout.as_ref(),
            ) {
                if let Some(shadow_map) = self.shadow_feature.shadow_map() {
                    let clear_values = [vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    }];

                    let render_pass_begin = vk::RenderPassBeginInfo::default()
                        .render_pass(shadow_map.render_pass)
                        .framebuffer(shadow_map.framebuffer)
                        .render_area(shadow_map.scissor())
                        .clear_values(&clear_values);

                    cmd_ctx.begin_render_pass(&render_pass_begin, vk::SubpassContents::INLINE);
                    cmd_ctx
                        .bind_pipeline(vk::PipelineBindPoint::GRAPHICS, shadow_pipeline.pipeline);

                    cmd_ctx.set_viewport(0, &[shadow_map.viewport()]);
                    cmd_ctx.set_scissor(0, &[shadow_map.scissor()]);

                    let light_space_matrix = self.shadow_feature.light_space_matrix();

                    // Draw all meshes
                    for item in &self.draw_items {
                        if let Some(uploaded) = self.model_renderer.get(&item.key) {
                            // Push constants for light space matrix and model transform.
                            let light_space_push =
                                crate::renderer::model_renderer::Mat4Push::from(light_space_matrix);
                            let model_push =
                                crate::renderer::model_renderer::Mat4Push::from(item.transform);

                            let mut push_data = Vec::with_capacity(128);
                            push_data.extend_from_slice(bytemuck::bytes_of(&light_space_push));
                            push_data.extend_from_slice(bytemuck::bytes_of(&model_push));

                            self.device.device.cmd_push_constants(
                                command_buffer,
                                shadow_layout.handle(),
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                &push_data,
                            );

                            // Bind vertex buffers
                            let offsets = [0];
                            self.device.device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[uploaded.vertex_buffer()],
                                &offsets,
                            );

                            // Bind Bindless Textures (Set 2)
                            if let Some(ref bindless) = self.bindless_manager {
                                self.device.device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    shadow_layout.handle(),
                                    2, // Set 2
                                    &[bindless.descriptor_set()],
                                    &[],
                                );
                            }

                            // Push texture index for alpha discard
                            let base_color_index = item.texture_indices[0];
                            self.device.device.cmd_push_constants(
                                command_buffer,
                                shadow_layout.handle(),
                                vk::ShaderStageFlags::FRAGMENT,
                                128,
                                bytemuck::bytes_of(&base_color_index),
                            );

                            if let Some(index_buffer) = uploaded.index_buffer() {
                                self.device.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    index_buffer,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                self.device.device.cmd_draw_indexed(
                                    command_buffer,
                                    uploaded.index_count(),
                                    1,
                                    0,
                                    0,
                                    0,
                                );
                            } else {
                                self.device.device.cmd_draw(
                                    command_buffer,
                                    uploaded.vertex_count(),
                                    1,
                                    0,
                                    0,
                                );
                            }
                        }
                    }

                    cmd_ctx.end_render_pass();
                }
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let framebuffer = self
                .framebuffers
                .get(image_index as usize)
                .ok_or_else(|| AshError::VulkanError("Framebuffer index out of range".into()))?;

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(main_render_pass)
                .framebuffer(framebuffer.handle())
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .clear_values(&clear_values);

            cmd_ctx.begin_render_pass(&render_pass_begin, vk::SubpassContents::INLINE);
            cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, scene_pipeline);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain_extent.width as f32,
                height: swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            };
            cmd_ctx.set_viewport(0, &[viewport]);
            cmd_ctx.set_scissor(0, &[scissor]);

            let render_ctx = FeatureRenderContext {
                device: self.device.device.as_ref(),
                descriptor_manager: self.descriptors.as_ref(),
                command_buffer,
                transform: &self.transform,
            };

            self.features.render(&render_ctx);

            let pipeline_layout = self.pipeline_layout.as_ref().ok_or_else(|| {
                AshError::VulkanError("Pipeline layout not available".to_string())
            })?;
            let pipeline_layout_handle = pipeline_layout.handle();

            let _ = (|| -> Result<vk::DescriptorSet> {
                if let Some(manager) = self.descriptors.as_ref() {
                    let frame_set = manager.frame_set(frame_index).ok_or_else(|| {
                        AshError::VulkanError("Frame descriptor set not available".to_string())
                    })?;
                    let material_set = manager.material_set(worker_index).ok_or_else(|| {
                        AshError::VulkanError("Material descriptor set not available".to_string())
                    })?;
                    cmd_ctx.bind_descriptor_sets(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout_handle,
                        0,
                        &[frame_set, material_set],
                        &[],
                    );

                    // Bind shadow map descriptor (set 3)
                    if let Some(shadow_map) = self.shadow_feature.shadow_map() {
                        if let Some(shadow_set) = manager.shadow_set(frame_index) {
                            // Bind shadow map texture to descriptor set
                            manager.bind_shadow_map(
                                frame_index,
                                shadow_map.depth_image_view,
                                shadow_map.sampler,
                            )?;
                            cmd_ctx.bind_descriptor_sets(
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline_layout_handle,
                                3, // Set 3: Shadow map
                                &[shadow_set],
                                &[],
                            );
                        }
                    }

                    // Bind bindless descriptor set (Set 2).
                    if let Some(ref bindless) = self.bindless_manager {
                        cmd_ctx.bind_descriptor_sets(
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout_handle,
                            2, // Set 2: Bindless textures
                            &[bindless.descriptor_set()],
                            &[],
                        );
                    }

                    // Bind Forward+ descriptor set (set 4)
                    if let Some(ref forward_plus) = self.forward_plus {
                        forward_plus.bind(
                            &self.device.device,
                            command_buffer,
                            pipeline_layout_handle,
                        );
                    }

                    Ok(vk::DescriptorSet::null())
                } else {
                    Ok(vk::DescriptorSet::null())
                }
            })()?;

            // Draw uploaded meshes in order
            for item in &self.draw_items {
                if let Some(uploaded) = self.model_renderer.get(&item.key) {
                    // Bindless architecture: indices passed via MaterialUniform.
                    // Explicit descriptor set binding for materials is bypassed.

                    if let Some(material_buffer) = self.material_buffers.get(worker_index) {
                        let mut material_buffer = material_buffer.lock();
                        log::debug!(
                            "Draw '{}' material: metallic {:.3}, roughness {:.3}, occlusion {:.3}, normal_scale {:.3}, flags {:?}",
                            item.key,
                            item.material.metallic,
                            item.material.roughness,
                            item.material.occlusion_strength,
                            item.material.normal_scale,
                            item.texture_flags
                        );
                        let uniform = material_buffer.uniform_mut();
                        uniform.set_base_color_factor(Vec4::from_array(item.material.color));
                        uniform.set_emissive_factor(Vec4::from_array(item.material.emissive));
                        uniform.set_metallic_roughness(
                            item.material.metallic,
                            item.material.roughness,
                        );
                        uniform.set_occlusion_strength(item.material.occlusion_strength);
                        uniform.set_normal_scale(item.material.normal_scale);

                        uniform.set_texture_indices(
                            item.texture_indices[0],
                            item.texture_indices[1],
                            item.texture_indices[2],
                            item.texture_indices[3],
                            item.emissive_index,
                        );
                        material_buffer.update()?;
                    }

                    let model_matrix = item.transform;
                    // Uniform buffer contains view and projection matrices from current frame synchronization.
                    let uniform_matrices = self.uniform_buffers[frame_index].matrices();
                    let view_matrix = uniform_matrices.view;
                    let projection_matrix = uniform_matrices.projection;
                    let base_color_binding = if item.texture_flags.base_color {
                        Some(0u32)
                    } else {
                        None
                    };
                    let mut material_push =
                        MaterialPushConstants::from_material(&item.material, base_color_binding);
                    material_push.normal_texture_set =
                        if item.texture_flags.normal { 1 } else { -1 };
                    material_push.metallic_roughness_texture_set =
                        if item.texture_flags.metallic_roughness {
                            2
                        } else {
                            -1
                        };
                    material_push.occlusion_texture_set =
                        if item.texture_flags.occlusion { 3 } else { -1 };
                    material_push.emissive_texture_set =
                        if item.texture_flags.emissive { 4 } else { -1 };

                    self.model_renderer.draw_mesh(
                        command_buffer,
                        pipeline_layout_handle,
                        uploaded,
                        model_matrix,
                        view_matrix,
                        projection_matrix,
                        &material_push,
                    );
                } else {
                    log::warn!("Uploaded data for mesh key '{}' missing", item.key);
                }
            }

            cmd_ctx.end_render_pass();

            // --- SSGI Pass ---
            if let (Some(ref mut gbuffer), Some(ref mut ssgi)) =
                (&mut self.gbuffer, &mut self.ssgi_pass)
            {
                let depth_buffer = self.depth_buffer.as_ref().unwrap();
                let inv_view_proj = (jittered_projection * view).inverse();

                ssgi.compute_gi(
                    command_buffer,
                    depth_buffer.view(),
                    gbuffer.normal_view(),
                    gbuffer.albedo_view(),
                    inv_view_proj,
                )?;
                ssgi.next_frame();
            }

            // --- VSR (Upscaling) Pass ---
            if let (Some(ref mut vsr), Some(ref mut gbuffer)) =
                (&mut self.vsr_pass, &mut self.gbuffer)
            {
                let depth_buffer = self.depth_buffer.as_ref().unwrap();
                // Pass current color result (which is now correctly the HDR buffer if initialized)
                // and upsample to VSR history.
                vsr.upscale(
                    command_buffer,
                    framebuffer.attachments()[0], // Index 0 is now HDR if active
                    depth_buffer.view(),
                    gbuffer.motion_view(),
                    jitter_uv,
                )?;
                vsr.next_frame();
            }

            // --- Post-Processing (Tonemapping & Resolve) ---
            if self.tonemapping_enabled {
                self.render_post_processing(command_buffer, image_index as usize)?;
            }

            cmd_ctx.end()?;

            let wait_semaphores = [image_available];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [render_finished];
            let command_buffers_submit = [command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers_submit)
                .signal_semaphores(&signal_semaphores);

            self.cmds
                .submit(self.device.graphics_queue, &[submit_info], in_flight_fence)?;

            let present_result = {
                let swapchain_ref = self
                    .swapchain
                    .as_ref()
                    .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?;
                swapchain_ref.present(self.device.present_queue, image_index, render_finished)
            };

            match present_result {
                Ok(()) => {
                    if self.swapchain_cleanup_pending {
                        self.flush_old_swapchains();
                    }
                }
                Err(AshError::SwapchainOutOfDate(_)) => {
                    self.request_swapchain_resize(swapchain_extent);
                    return Ok(());
                }
                Err(err) => return Err(err),
            }

            self.current_frame = (frame_index + 1) % self.command_buffers.len();

            Ok(())
        }
    }

    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    pub fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transform
    }

    pub fn buffer_pool(&self) -> Arc<BufferPool> {
        Arc::clone(&self.buffer_pool)
    }

    pub fn mesh_mut(&mut self) -> Option<&mut Mesh> {
        self.mesh.as_mut()
    }

    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    // ──────────────────────────────────────────────────────────
    // Post-Processing API
    // ──────────────────────────────────────────────────────────

    /// Sets the MSAA preset.
    pub fn set_msaa_preset(&mut self, preset: MsaaPreset) {
        self.msaa_preset = preset;
        log::info!("MSAA preset set to {preset:?}");
        // MSAA targets require recreation upon preset modification.
    }

    /// Returns the current MSAA preset
    pub fn msaa_preset(&self) -> MsaaPreset {
        self.msaa_preset
    }

    /// Enables or disables tonemapping
    pub fn set_tonemapping_enabled(&mut self, enabled: bool) {
        self.tonemapping_enabled = enabled;
    }

    /// Returns whether tonemapping is enabled
    pub fn tonemapping_enabled(&self) -> bool {
        self.tonemapping_enabled
    }

    /// Sets the tonemapping exposure value
    pub fn set_tonemapping_exposure(&mut self, exposure: f32) {
        self.tonemapping_exposure = exposure.max(0.0);
    }

    /// Returns the tonemapping exposure value
    pub fn tonemapping_exposure(&self) -> f32 {
        self.tonemapping_exposure
    }

    /// Sets the tonemapping gamma value
    pub fn set_tonemapping_gamma(&mut self, gamma: f32) {
        self.tonemapping_gamma = gamma.max(0.1);
    }

    /// Returns the tonemapping gamma value
    pub fn tonemapping_gamma(&self) -> f32 {
        self.tonemapping_gamma
    }

    /// Enables or disables bloom
    pub fn set_bloom_enabled(&mut self, enabled: bool) {
        self.bloom_enabled = enabled;
    }

    /// Returns whether bloom is enabled
    pub fn bloom_enabled(&self) -> bool {
        self.bloom_enabled
    }

    /// Sets the bloom intensity
    pub fn set_bloom_intensity(&mut self, intensity: f32) {
        self.bloom_intensity = intensity.clamp(0.0, 2.0);
    }

    /// Returns the bloom intensity
    pub fn bloom_intensity(&self) -> f32 {
        self.bloom_intensity
    }

    // ──────────────────────────────────────────────────────────
    // Forward+ Lighting API
    // ──────────────────────────────────────────────────────────

    /// Update point lights for Forward+ rendering
    ///
    /// Call this each frame to update light positions and properties.
    pub fn update_point_lights(&mut self, lights: &[crate::renderer::features::PointLight]) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(lights, &[]);
            // Upload to GPU
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma);
            }
        }
    }

    /// Update directional lights for Forward+ rendering
    pub fn update_directional_lights(
        &mut self,
        lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(&[], lights);
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma);
            }
        }
    }

    /// Update all lights (point and directional) for Forward+ rendering
    pub fn update_lights(
        &mut self,
        point_lights: &[crate::renderer::features::PointLight],
        directional_lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(point_lights, directional_lights);
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma);
            }
        }
    }

    /// Returns whether Forward+ lighting is enabled
    pub fn forward_plus_enabled(&self) -> bool {
        self.forward_plus
            .as_ref()
            .map(|fp| fp.is_enabled())
            .unwrap_or(false)
    }

    /// Returns the number of active lights
    pub fn forward_plus_light_count(&self) -> usize {
        self.forward_plus
            .as_ref()
            .map(|fp| fp.light_count())
            .unwrap_or(0)
    }

    // ──────────────────────────────────────────────────────────
    // GPU-Driven Occlusion Culling API
    // ──────────────────────────────────────────────────────────

    /// Enables GPU-driven occlusion culling using Hi-Z pyramid.
    ///
    /// This initializes the Hi-Z pass and indirect draw pass for GPU-based
    /// visibility culling. Objects are tested against a hierarchical depth
    /// buffer before rendering, reducing draw calls significantly.
    ///
    /// # Safety
    /// Should be called after the renderer is fully initialized.
    pub fn enable_occlusion_culling(&mut self) -> Result<()> {
        if self.hiz_pass.is_some() {
            return Ok(()); // Already enabled
        }

        let extent = self
            .swapchain
            .as_ref()
            .map(|s| s.extent)
            .unwrap_or(vk::Extent2D {
                width: 1920,
                height: 1080,
            });

        // Create Hi-Z pass
        let mut hiz = HiZPass::new(Arc::clone(&self.device.device));
        unsafe {
            hiz.init(&self.alloc.vma, &self.device, extent.width, extent.height);
        }

        // Create Indirect Draw pass
        let mut indirect = IndirectDrawPass::new(Arc::clone(&self.device.device));
        unsafe {
            indirect.init(
                &self.alloc.vma,
                &self.device,
                crate::renderer::indirect_draw::MAX_INDIRECT_OBJECTS,
            );
            indirect.update_hiz_descriptor(&hiz);
        }

        self.hiz_pass = Some(hiz);
        self.indirect_draw_pass = Some(indirect);

        log::info!("Occlusion culling enabled (Hi-Z + Indirect Draw)");
        Ok(())
    }

    /// Returns whether GPU-driven occlusion culling is enabled
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.hiz_pass.is_some() && self.indirect_draw_pass.is_some()
    }

    // ──────────────────────────────────────────────────────────
    // Temporal Super-Resolution API
    // ──────────────────────────────────────────────────────────

    /// Enables Temporal Super-Resolution (TSR)
    ///
    /// TSR renders at a lower internal resolution and uses temporal
    /// accumulation to reconstruct higher quality output. This improves
    /// performance while maintaining near-native image quality.
    ///
    /// # Arguments
    /// * `quality` - The TSR quality preset (affects internal render resolution)
    pub fn enable_vsr(&mut self, quality: VsrQuality) -> Result<()> {
        if self.vsr_pass.is_some() {
            return Ok(()); // Already enabled
        }

        let extent = self
            .swapchain
            .as_ref()
            .map(|s| s.extent)
            .unwrap_or(vk::Extent2D {
                width: 1920,
                height: 1080,
            });

        let mut vsr = VsrPass::new(Arc::clone(&self.device.device));
        unsafe {
            vsr.init(
                &self.alloc.vma,
                &self.device,
                extent.width,
                extent.height,
                quality,
            );
        }

        self.vsr_pass = Some(vsr);
        log::info!(
            "TSR enabled with {:?} quality ({}x upscale)",
            quality,
            quality.factor()
        );
        Ok(())
    }

    /// Returns whether TSR is enabled
    pub fn tsr_enabled(&self) -> bool {
        self.vsr_pass.is_some()
    }

    /// Returns the current TSR quality preset
    pub fn vsr_quality(&self) -> Option<VsrQuality> {
        self.vsr_pass.as_ref().map(|t| t.quality())
    }

    /// Get jittered projection matrix for TAA/TSR
    ///
    /// Call this each frame to get a projection matrix with sub-pixel jitter
    /// applied. This is essential for temporal accumulation quality.
    pub fn jitter_projection(&mut self, projection: glam::Mat4) -> glam::Mat4 {
        if let Some(ref mut vsr) = self.vsr_pass {
            vsr.jitter_projection(projection)
        } else {
            projection
        }
    }

    // ──────────────────────────────────────────────────────────
    // Screen-Space Global Illumination API
    // ──────────────────────────────────────────────────────────

    /// Enables Screen-Space Global Illumination (SSGI)
    ///
    /// SSGI provides real-time indirect lighting by tracing rays
    /// against the depth buffer. Runs at half resolution with
    /// temporal accumulation for improved quality.
    ///
    /// # Arguments
    /// * `quality` - The SSGI quality preset (affects ray/step counts)
    pub fn enable_ssgi(&mut self, quality: SsgiQuality) -> Result<()> {
        if self.ssgi_pass.is_some() {
            return Ok(()); // Already enabled
        }

        let extent = self
            .swapchain
            .as_ref()
            .map(|s| s.extent)
            .unwrap_or(vk::Extent2D {
                width: 1920,
                height: 1080,
            });

        let mut ssgi = SsgiPass::new(Arc::clone(&self.device.device));
        unsafe {
            ssgi.init(
                &self.alloc.vma,
                &self.device,
                extent.width,
                extent.height,
                quality,
            );
        }

        self.ssgi_pass = Some(ssgi);
        log::info!(
            "SSGI enabled with {:?} quality ({} rays, {} steps)",
            quality,
            quality.ray_count(),
            quality.step_count()
        );
        Ok(())
    }

    /// Returns whether SSGI is enabled
    pub fn ssgi_enabled(&self) -> bool {
        self.ssgi_pass.is_some()
    }

    /// Returns the current SSGI quality preset
    pub fn ssgi_quality(&self) -> Option<SsgiQuality> {
        self.ssgi_pass.as_ref().map(|s| s.quality())
    }

    /// Set SSGI intensity
    pub fn set_ssgi_intensity(&mut self, intensity: f32) {
        if let Some(ref mut ssgi) = self.ssgi_pass {
            ssgi.set_intensity(intensity);
        }
    }

    // ──────────────────────────────────────────────────────────
    // Post-Processing Initialization & Application
    // ──────────────────────────────────────────────────────────

    /// Enables HDR rendering. Should be called after initialization.
    /// Allocates GPU memory for the HDR buffer.
    pub fn initialize_hdr(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            let hdr = hdr_framebuffer::HdrFramebuffer::new(
                Arc::clone(&self.device.device),
                Arc::clone(&self.alloc),
                width,
                height,
            )?;
            self.hdr_framebuffer = Some(hdr);
            log::info!("HDR framebuffer initialized ({width}x{height})");
        }

        Ok(())
    }

    /// Enables fullscreen effects. Should be called after initialization.
    pub fn initialize_fullscreen_pass(&mut self) -> Result<()> {
        let format = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
            .format;

        unsafe {
            let pass =
                fullscreen_pass::FullscreenPass::new(Arc::clone(&self.device.device), format)?;
            self.fullscreen_pass = Some(pass);
            log::info!("Fullscreen pass initialized");
        }

        Ok(())
    }

    /// Enables post-processing with default settings
    ///
    /// Initializes HDR, fullscreen pass, and enables tonemapping.
    pub fn enable_post_processing(&mut self) -> Result<()> {
        let extent = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain not available".into()))?
            .extent;

        self.initialize_hdr(extent.width, extent.height)?;
        self.initialize_fullscreen_pass()?;
        self.create_post_descriptors()?;
        self.recreate_post_pipeline()?;

        self.tonemapping_enabled = true;
        log::info!("Post-processing pipeline enabled (HDR + Tonemapping)");
        Ok(())
    }

    fn create_post_descriptors(&mut self) -> Result<()> {
        if self.fullscreen_pass.is_none() {
            return Ok(());
        }

        let device = &self.device.device;
        let count = self.framebuffers.len() as u32;

        // Cleanup old pool if exists
        if self.post_descriptor_pool != vk::DescriptorPool::null() {
            unsafe {
                device.destroy_descriptor_pool(self.post_descriptor_pool, None);
            }
        }

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: count * 3, // HDR, Bloom, SSGI
        }];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(&pool_sizes);

        self.post_descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create post descriptor pool: {e}"))
                })?
        };

        let layouts = vec![
            self.fullscreen_pass
                .as_ref()
                .unwrap()
                .descriptor_set_layout();
            count as usize
        ];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.post_descriptor_pool)
            .set_layouts(&layouts);

        self.post_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
                AshError::VulkanError(format!("Failed to allocate post descriptor sets: {e}"))
            })?
        };

        self.update_post_descriptors()?;

        Ok(())
    }

    fn update_post_descriptors(&mut self) -> Result<()> {
        if self.post_descriptor_sets.is_empty() {
            return Ok(());
        }

        let hdr = self.hdr_framebuffer.as_ref();
        let vsr = self.vsr_pass.as_ref();

        let color_view = if let Some(vsr) = vsr {
            vsr.output_view()
        } else if let Some(hdr) = hdr {
            hdr.view()
        } else {
            return Ok(());
        };

        let sampler = if let Some(hdr) = hdr {
            hdr.sampler()
        } else {
            // Default sampler if HDR not available (though it should be)
            unsafe {
                self.device
                    .device
                    .create_sampler(&vk::SamplerCreateInfo::default(), None)
                    .unwrap()
            }
        };

        let layout = if vsr.is_some() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };

        let ssgi_view = self
            .ssgi_pass
            .as_ref()
            .map(|s| s.gi_view())
            .unwrap_or(color_view);
        let bloom_view = color_view; // Placeholder until bloom is fully implemented

        for descriptor_set in &self.post_descriptor_sets {
            let color_info = vk::DescriptorImageInfo {
                sampler,
                image_view: color_view,
                image_layout: layout,
            };

            let bloom_info = vk::DescriptorImageInfo {
                sampler,
                image_view: bloom_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let ssgi_info = vk::DescriptorImageInfo {
                sampler,
                image_view: ssgi_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let color_infos = [color_info];
            let bloom_infos = [bloom_info];
            let ssgi_infos = [ssgi_info];

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&color_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&bloom_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&ssgi_infos),
            ];

            unsafe {
                self.device
                    .device
                    .update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        Ok(())
    }

    fn recreate_post_pipeline(&mut self) -> Result<()> {
        if let Some(ref pass) = self.fullscreen_pass {
            let mut builder = vulkan::Pipeline::builder(Arc::clone(&self.device.device))
                .with_layout(pass.pipeline_layout())
                .with_render_pass(pass.render_pass())
                .with_extent(self.swapchain.as_ref().unwrap().extent)
                .with_cull_mode(vk::CullModeFlags::NONE);

            builder = builder.add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/postprocess.vert.spv")),
                vk::ShaderStageFlags::VERTEX,
                "main",
            )?;

            builder = builder.add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/tonemapping.frag.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                "main",
            )?;

            let pipeline = builder.build()?;
            self.post_pipeline = Some(pipeline.pipeline);
            // Pipeline cleanup is handled by resource registry if we register it,
            // but for simplicity we'll just manage it manually for now.
        }
        Ok(())
    }

    /// Checks if HDR and fullscreen pass are initialized.
    pub fn post_processing_ready(&self) -> bool {
        self.hdr_framebuffer.is_some() && self.fullscreen_pass.is_some()
    }

    /// Returns post-processing settings as a tuple (exposure, gamma, bloom_intensity)
    pub fn post_processing_settings(&self) -> (f32, f32, f32) {
        (
            self.tonemapping_exposure,
            self.tonemapping_gamma,
            self.bloom_intensity,
        )
    }

    // ========== Diagnostics API ==========

    /// Get current diagnostics state
    pub fn diagnostics(&self) -> &DiagnosticsState {
        &self.diagnostics
    }

    /// Get mutable diagnostics state
    pub fn diagnostics_mut(&mut self) -> &mut DiagnosticsState {
        &mut self.diagnostics
    }

    /// Set diagnostics display mode
    pub fn set_diagnostics_mode(&mut self, mode: DiagnosticsMode) {
        self.diagnostics.mode = mode;
        log::info!("Diagnostics mode set to {mode:?}");
    }

    /// Toggle diagnostics mode (F6 behavior)
    pub fn toggle_diagnostics(&mut self) {
        self.diagnostics.toggle_mode();
    }

    /// Collects frame diagnostics.
    /// Call this after render_frame() to collect stats
    pub fn update_diagnostics(&mut self) {
        // Begin frame profiling
        self.frame_profiler.begin_frame();

        // Collect frame stats
        self.diagnostics.frame_stats = self.frame_profiler.stats(
            self.diagnostics.frame_stats.draw_calls,
            self.diagnostics.frame_stats.triangles,
        );

        // Collect memory stats from buffer pool
        let (available, in_use, total_allocated) = self.buffer_pool.stats();
        self.diagnostics.memory_stats.buffer_pool = (available, in_use, total_allocated);

        // Collect GPU timings (if profiler initialized)
        if let Some(ref mut profiler) = self.gpu_profiler {
            self.diagnostics.gpu_timings = profiler.end_frame();
        }

        // Print to console if enabled
        if self.diagnostics.should_print_console() {
            self.diagnostics.print_console();
        }
    }

    /// Initialize GPU profiler for timing queries
    ///
    /// Automatically initialized when diagnostics are active.
    pub fn initialize_gpu_profiler(&mut self) -> Result<()> {
        if self.gpu_profiler.is_some() {
            return Ok(());
        }

        let timestamp_period = self.device.timestamp_period_ns;
        let timestamps_supported = timestamp_period > 0.0;

        unsafe {
            let profiler = GpuProfiler::new(
                Arc::clone(&self.device.device),
                timestamp_period,
                timestamps_supported,
            )?;
            self.gpu_profiler = Some(profiler);
        }

        Ok(())
    }

    /// Get overlay vertices for current frame
    ///
    /// Returns (text_vertices, background_vertices) for rendering.
    /// Call this after update_diagnostics() to get fresh data.
    pub fn overlay_vertices(
        &mut self,
    ) -> (
        &[crate::renderer::diagnostics::TextVertex],
        &[crate::renderer::diagnostics::TextVertex],
    ) {
        let extent = self
            .swapchain
            .as_ref()
            .map(|s| (s.extent.width as f32, s.extent.height as f32))
            .unwrap_or((1920.0, 1080.0));

        self.diagnostics_overlay
            .generate_vertices(&self.diagnostics, extent.0, extent.1)
    }

    /// Check if overlay should be rendered this frame
    pub fn should_render_overlay(&self) -> bool {
        self.diagnostics.mode.overlay_enabled()
    }

    /// Get mutable reference to diagnostics overlay for configuration
    pub fn diagnostics_overlay_mut(&mut self) -> &mut DiagnosticsOverlay {
        &mut self.diagnostics_overlay
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            log::info!("Shutting down Ash Renderer...");

            let _ = self.device.device.device_wait_idle();

            self.flush_old_swapchains();

            if let Err(e) = self.resources.cleanup() {
                log::error!("Resource registry cleanup failed: {e}");
            }

            if let Some(manager) = self.descriptors.take() {
                drop(manager);
            }

            self.features.cleanup();

            // Cleanup Forward+ integration (Phase 5)
            if let Some(mut fp) = self.forward_plus.take() {
                fp.destroy(&self.alloc.vma);
            }

            // Cleanup UE5 feature modules (Phase 5)
            if let Some(mut hiz) = self.hiz_pass.take() {
                hiz.destroy(&self.alloc.vma);
            }
            if let Some(mut indirect) = self.indirect_draw_pass.take() {
                indirect.destroy(&self.alloc.vma);
            }
            if let Some(mut vsr) = self.vsr_pass.take() {
                vsr.destroy(&self.alloc.vma);
            }
            if let Some(mut ssgi) = self.ssgi_pass.take() {
                ssgi.destroy(&self.alloc.vma);
            }

            for ub in &mut self.uniform_buffers {
                let _ = ub.cleanup();
            }
            self.uniform_buffers.clear();

            for buffer in &self.material_buffers {
                let mut buffer = buffer.lock();
                let _ = buffer.cleanup();
            }
            self.material_buffers.clear();

            self.model_renderer.clear();
            self.draw_items.clear();

            self.mesh = None;

            self.depth_buffer = None;
            self.pipeline = None;
            self.render_pass = None;
            self.swapchain = None;

            log::info!("Ash Renderer shut down successfully");
        }
    }
}
