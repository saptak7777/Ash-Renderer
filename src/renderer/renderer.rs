use crate::{
    renderer::{
        diagnostics::{
            DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler,
        },
        features::{
            AutoRotateFeature, FeatureFrameContext, FeatureManager, FeatureRenderContext,
            ShadowFeature,
        },
        fullscreen_pass, hdr_framebuffer,
        model_renderer::{MaterialPushConstants, MeshPushConstants, ModelRenderer},
        resource_registry::{ResourceId, ResourceRegistry},
        resources,
        resources::uniform::{MaterialBuffer, UniformBuffer},
        DepthBuffer, Material, Mesh, PipelineCache, Texture, TextureData, Transform,
    },
    vulkan, AshError, Result,
};

use ash::vk;
use bytemuck::Pod;
use glam::{Mat4, Vec4};
use parking_lot::Mutex;
use resources::BufferPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use vk_mem;

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

/// Main renderer - Phase 5 (Stable)
pub struct Renderer {
    // Resources that depend on allocator/device - dropped first
    buffer_pool: Arc<BufferPool>,
    resource_registry: Arc<ResourceRegistry>,
    feature_manager: FeatureManager,
    _pipeline_cache: PipelineCache,
    command_manager: vulkan::CommandBufferManager,
    worker_count: usize,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_syncs: Vec<vulkan::FrameSync>,
    current_frame: usize,
    default_texture: Texture,
    model_renderer: ModelRenderer,
    draw_items: Vec<DrawItem>,
    swapchain: Option<vulkan::SwapchainWrapper>,
    render_pass: Option<vulkan::RenderPass>,
    render_pass_id: Option<ResourceId>,
    pipeline: Option<vulkan::Pipeline>,
    depth_buffer: Option<DepthBuffer>,
    uniform_buffers: Vec<UniformBuffer>,
    material_buffers: Vec<Mutex<MaterialBuffer>>,
    pipeline_layout: Option<vulkan::PipelineLayout>,
    descriptor_manager: Option<vulkan::DescriptorManager>,
    framebuffers: Vec<vulkan::Framebuffer>,
    framebuffer_ids: Vec<ResourceId>,
    start_time: Instant,
    pub mesh: Option<Mesh>,
    material: Material,
    transform: Transform,
    mesh_registry: HashMap<u32, String>,
    mesh_texture_sets: HashMap<String, vk::DescriptorSet>,
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
    // IMPORTANT: These must be at the end so they drop LAST
    // All resources above depend on allocator, which depends on device
    allocator: Arc<vulkan::Allocator>,
    vulkan_device: vulkan::VulkanDevice,
}

#[derive(Clone)]
struct DrawItem {
    key: String,
    transform: Mat4,
    material: Material,
    texture_flags: TexturePresenceFlags,
    texture_set: vk::DescriptorSet,
    /// Bindless texture indices (Phase 6)
    bindless_indices: [i32; 5],
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
    fn any(&self) -> bool {
        self.base_color || self.normal || self.metallic_roughness || self.occlusion || self.emissive
    }
}

#[derive(Copy, Clone)]
struct TextureSlot {
    view: vk::ImageView,
    sampler: vk::Sampler,
    has_texture: bool,
}

impl TextureSlot {
    fn from_texture(texture: Option<&Texture>, fallback: &Texture) -> Self {
        match texture {
            Some(tex) => Self {
                view: tex.view(),
                sampler: tex.sampler(),
                has_texture: true,
            },
            None => Self {
                view: fallback.view(),
                sampler: fallback.sampler(),
                has_texture: false,
            },
        }
    }
}

#[derive(Copy, Clone)]
struct TextureBindingInfo {
    base_color: TextureSlot,
    normal: TextureSlot,
    metallic_roughness: TextureSlot,
    occlusion: TextureSlot,
    emissive: TextureSlot,
    flags: TexturePresenceFlags,
    /// Bindless texture indices (Phase 6): -1 means no texture
    bindless_indices: [i32; 5],
}

impl TextureBindingInfo {
    fn from_mesh(mesh: &Mesh, default_texture: &Texture) -> Self {
        let base_color = TextureSlot::from_texture(mesh.texture(), default_texture);
        let normal = TextureSlot::from_texture(mesh.normal_texture(), default_texture);
        let metallic_roughness =
            TextureSlot::from_texture(mesh.metallic_roughness_texture(), default_texture);
        let occlusion = TextureSlot::from_texture(mesh.occlusion_texture(), default_texture);
        let emissive = TextureSlot::from_texture(mesh.emissive_texture(), default_texture);

        let flags = TexturePresenceFlags {
            base_color: base_color.has_texture,
            normal: normal.has_texture,
            metallic_roughness: metallic_roughness.has_texture,
            occlusion: occlusion.has_texture,
            emissive: emissive.has_texture,
        };

        Self {
            base_color,
            normal,
            metallic_roughness,
            occlusion,
            emissive,
            flags,
            bindless_indices: [-1; 5], // Not registered yet
        }
    }

    /// Register textures with BindlessManager and store indices (Phase 6)
    fn register_with_bindless(&mut self, bindless: &mut vulkan::BindlessManager) -> Result<()> {
        // Register each texture slot if it has a texture
        let slots = [
            (&self.base_color, self.flags.base_color),
            (&self.normal, self.flags.normal),
            (&self.metallic_roughness, self.flags.metallic_roughness),
            (&self.occlusion, self.flags.occlusion),
            (&self.emissive, self.flags.emissive),
        ];

        for (i, (slot, has_texture)) in slots.iter().enumerate() {
            if *has_texture {
                let index = bindless.add_sampled_image(slot.view, slot.sampler)?;
                self.bindless_indices[i] = index as i32;
            } else {
                self.bindless_indices[i] = -1;
            }
        }

        Ok(())
    }

    fn flags(&self) -> TexturePresenceFlags {
        self.flags
    }

    fn descriptor_bindings(&self) -> [(u32, vk::ImageView, vk::Sampler); 5] {
        [
            (0, self.base_color.view, self.base_color.sampler),
            (1, self.normal.view, self.normal.sampler),
            (
                2,
                self.metallic_roughness.view,
                self.metallic_roughness.sampler,
            ),
            (3, self.occlusion.view, self.occlusion.sampler),
            (4, self.emissive.view, self.emissive.sampler),
        ]
    }

    fn has_any_texture(&self) -> bool {
        self.flags.any()
    }
}

impl Renderer {
    /// Create renderer - Phase 5 (Stable, no Phase 6 descriptors)
    pub fn new<S: crate::vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        unsafe {
            log::info!("Initializing Ash renderer (Phase 5 - Stable)...");

            let vulkan_instance = Arc::new(vulkan::VulkanInstance::from_surface(
                surface_provider,
                cfg!(debug_assertions),
            )?);
            let vulkan_device = vulkan::VulkanDevice::new(Arc::clone(&vulkan_instance))?;
            let allocator = Arc::new(vulkan::Allocator::new(&vulkan_device)?);
            let resource_registry =
                Arc::new(ResourceRegistry::new(Arc::clone(&vulkan_device.device)));
            let mut feature_manager = FeatureManager::new();
            feature_manager.set_device(Arc::clone(&vulkan_device.device));
            feature_manager.add_feature(AutoRotateFeature::new());

            // Initialize Shadow Feature
            let mut shadow_feature = ShadowFeature::new();
            if shadow_feature.is_active() || shadow_feature.config.enabled {
                let shadow_map = crate::renderer::shadow_map::ShadowMap::new(
                    Arc::clone(&vulkan_device.device),
                    vulkan_device.memory_properties,
                    shadow_feature.config.clone(),
                )?;
                shadow_feature.set_shadow_map(shadow_map);
            }
            // Create pipeline cache with disk persistence for faster startup
            let cache_file = std::env::temp_dir()
                .join("ash_renderer")
                .join("pipeline.cache");
            let pipeline_cache = PipelineCache::with_persistence(
                Arc::clone(&vulkan_device.device),
                Some(cache_file),
            )?;
            let renderer_config = RendererConfig::default();
            let pipeline_cfg = &renderer_config.pipeline;
            let buffer_pool = Arc::new(BufferPool::new(Arc::clone(&allocator)));

            // Phase 5.5: Handle headless vs windowed
            let is_headless = surface_provider.required_extensions().is_empty();
            let (swapchain, swapchain_image_views, swapchain_format, swapchain_extent) =
                if is_headless {
                    log::info!("Initializing in HEADLESS mode (no swapchain)");
                    let extent = surface_provider.extent();
                    let format = vk::Format::R8G8B8A8_UNORM; // Fallback format
                    let frame_count = 2; // Simulate double buffering

                    let mut views = Vec::new();
                    // Create offscreen images for headless
                    for _ in 0..frame_count {
                        let create_info = vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(format)
                            .extent(vk::Extent3D {
                                width: extent.width,
                                height: extent.height,
                                depth: 1,
                            })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(
                                vk::ImageUsageFlags::COLOR_ATTACHMENT
                                    | vk::ImageUsageFlags::TRANSFER_SRC,
                            )
                            .sharing_mode(vk::SharingMode::EXCLUSIVE);

                        let image = allocator
                            .create_image(&create_info, vk_mem::MemoryUsage::AutoPreferDevice)?;
                        // Note: We leak the image here as we don't store it properly for cleanup in this simplified headless implementation
                        // Ideally we'd store these in a dedicated vector in Renderer

                        let view_info = vk::ImageViewCreateInfo::default()
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format)
                            .components(vk::ComponentMapping::default())
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image(image.0);

                        let view = vulkan_device
                            .device
                            .create_image_view(&view_info, None)
                            .map_err(|e| {
                                AshError::VulkanError(format!(
                                    "Failed to create headless view: {e}"
                                ))
                            })?;

                        // In real code we'd register these for cleanup, but for now just push
                        views.push(view);
                    }

                    (None, views, format, extent)
                } else {
                    let mut sw = vulkan::SwapchainWrapper::new(&vulkan_device)?;
                    let views = sw.image_views.clone();
                    sw.mark_image_views_managed_by_registry();
                    let format = sw.format;
                    let extent = sw.extent;
                    (Some(sw), views, format, extent)
                };

            let mut swapchain_image_view_ids = Vec::with_capacity(swapchain_image_views.len());
            for &image_view in &swapchain_image_views {
                let image_view_id =
                    resource_registry
                        .register_image_view(image_view)
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to register swapchain image view: {e}"
                            ))
                        })?;
                swapchain_image_view_ids.push(image_view_id);
            }

            let mut depth_buffer = DepthBuffer::new(
                Arc::clone(&vulkan_device.device),
                Arc::clone(&allocator),
                swapchain_extent.width,
                swapchain_extent.height,
            )?;
            let depth_buffer_id = depth_buffer
                .register_with_registry(&resource_registry)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register depth buffer: {e}"))
                })?;

            let mut render_pass = vulkan::RenderPass::builder(Arc::clone(&vulkan_device.device))
                .with_swapchain_color(swapchain_format)
                .with_depth_attachment(depth_buffer.format())
                .build()?;
            let render_pass_id = resource_registry
                .register_render_pass(render_pass.handle())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register render pass: {e}"))
                })?;
            render_pass.mark_managed_by_registry();

            let mut framebuffers = Vec::new();
            let mut framebuffer_ids = Vec::new();
            for (index, &image_view) in swapchain_image_views.iter().enumerate() {
                let attachments = [image_view, depth_buffer.view()];
                let framebuffer = vulkan::Framebuffer::new(
                    Arc::clone(&vulkan_device.device),
                    render_pass.handle(),
                    &attachments,
                    swapchain_extent,
                )?;
                let framebuffer_id = resource_registry
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

            log::info!(
                "Created {} framebuffers with depth attachment",
                framebuffers.len()
            );

            let worker_count = thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);

            let command_manager = vulkan::CommandBufferManager::new(
                Arc::clone(&vulkan_device.device),
                vulkan_device.graphics_queue_family,
                worker_count,
            )?;
            log::info!(
                "Command manager initialized for {} frames",
                framebuffers.len()
            );

            let command_buffers =
                command_manager.allocate_primary_buffers(framebuffers.len() as u32)?;

            let mut frame_syncs = Vec::with_capacity(framebuffers.len());
            let mut frame_sync_ids = Vec::with_capacity(framebuffers.len());
            for _ in 0..framebuffers.len() {
                let mut sync = vulkan::FrameSync::new(Arc::clone(&vulkan_device.device))?;
                let image_available_id = resource_registry
                    .register_semaphore(sync.image_available)
                    .map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to register image-available semaphore: {e}"
                        ))
                    })?;
                let render_finished_id = resource_registry
                    .register_semaphore(sync.render_finished)
                    .map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to register render-finished semaphore: {e}"
                        ))
                    })?;
                let fence_id = resource_registry
                    .register_fence(sync.in_flight)
                    .map_err(|e| {
                        AshError::VulkanError(format!("Failed to register in-flight fence: {e}"))
                    })?;
                sync.mark_managed_by_registry();
                frame_syncs.push(sync);
                frame_sync_ids.push((image_available_id, render_finished_id, fence_id));
            }

            resource_registry
                .register_command_pool(command_manager.upload_command_pool_handle())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register command pool: {e}"))
                })?;
            command_manager.mark_pool_managed_by_registry();

            let mut model_renderer =
                ModelRenderer::new(Arc::clone(&allocator), Arc::clone(&vulkan_device.device));

            // Phase 5: Create uniform buffers (Double Buffering)
            let mut uniform_buffers = Vec::with_capacity(framebuffers.len());
            let aspect = swapchain_extent.width as f32 / swapchain_extent.height as f32;

            for _ in 0..framebuffers.len() {
                let mut buffer =
                    UniformBuffer::new(Arc::clone(&allocator), Arc::clone(&vulkan_device.device))?;

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
            log::info!(
                "Phase 5: Uniform buffers (count: {}) initialized",
                uniform_buffers.len()
            );

            // Create descriptor manager and pipeline layout
            let default_texture_data = TextureData::solid_color([255, 255, 255, 255]);
            let default_texture = Texture::from_data(
                Arc::clone(&allocator),
                Arc::clone(&vulkan_device.device),
                command_manager.upload_command_pool_handle(),
                vulkan_device.graphics_queue,
                &default_texture_data,
                vk::Format::R8G8B8A8_SRGB,
                Some("default_texture"),
            )?;

            let material = Material::default();

            let mut material_buffers = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                let mut material_buffer =
                    MaterialBuffer::new(Arc::clone(&allocator), Arc::clone(&vulkan_device.device))?;
                {
                    let uniform = material_buffer.uniform_mut();
                    uniform.set_base_color_factor(Vec4::from_array(material.color));
                    uniform.set_emissive_factor(Vec4::from_array(material.emissive));
                    uniform.set_metallic_roughness(material.metallic, material.roughness);
                    uniform.set_occlusion_strength(material.occlusion_strength);
                    uniform.set_normal_scale(material.normal_scale);
                    // Phase 6: Use -1 for no texture (bindless indices set during texture registration)
                    uniform.set_texture_indices(-1, -1, -1, -1, -1);
                    uniform.set_alpha_cutoff(0.1);
                }
                material_buffer.update()?;
                material_buffers.push(Mutex::new(material_buffer));
            }

            let mut descriptor_manager = vulkan::DescriptorManager::new(
                Arc::clone(&vulkan_device.device),
                framebuffers.len() as u32,
                worker_count as u32,
                Some(Arc::clone(&resource_registry)),
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
                    worker_index,
                    buffer.buffer,
                    material_size,
                )?;
            }

            let default_bindings = [
                (0, default_texture.view(), default_texture.sampler()),
                (1, default_texture.view(), default_texture.sampler()),
                (2, default_texture.view(), default_texture.sampler()),
                (3, default_texture.view(), default_texture.sampler()),
                (4, default_texture.view(), default_texture.sampler()),
            ];

            descriptor_manager.bind_material_textures(
                descriptor_manager.default_texture_array_set(),
                &default_bindings,
            )?;

            validate_worker_resources(
                worker_count,
                descriptor_manager.material_set_count(),
                material_buffers.len(),
            )?;

            // Create bindless manager for texture array access
            let mut bindless_manager = vulkan::BindlessManager::new(
                Arc::clone(&vulkan_device.device),
                descriptor_manager.allocator_mut(),
                1024, // Max bindless resources
            )?;
            log::info!("BindlessManager created with 1024 max resources");

            let set_layouts = [
                descriptor_manager.frame_layout(),
                descriptor_manager.material_layout(),
                descriptor_manager.material_texture_layout(),
                descriptor_manager.shadow_layout(), // Set 3: Shadow map sampler
                bindless_manager.layout(),          // Set 4: Bindless textures
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
                vulkan::PipelineLayout::builder(Arc::clone(&vulkan_device.device));
            for layout in &set_layouts {
                pipeline_layout_builder = pipeline_layout_builder.add_set_layout(*layout);
            }
            for range in &push_constant_ranges {
                pipeline_layout_builder = pipeline_layout_builder.add_push_constant(*range);
            }
            let mut pipeline_layout = pipeline_layout_builder.build()?;
            let pipeline_layout_id = resource_registry
                .register_pipeline_layout(pipeline_layout.handle())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to register pipeline layout: {e}"))
                })?;
            pipeline_layout.mark_managed_by_registry();

            log::info!("Pipeline layout created with descriptor set layout");

            // NOW create pipeline
            let mut pipeline_builder = vulkan::Pipeline::builder(Arc::clone(&vulkan_device.device))
                .with_layout(pipeline_layout.handle())
                .with_render_pass(render_pass.handle())
                .with_extent(swapchain_extent)
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
                .add_shader_with_options(
                    "shaders/vert.spv",
                    vk::ShaderStageFlags::VERTEX,
                    pipeline_cfg.watch_shaders,
                )?
                .add_shader_with_options(
                    "shaders/frag.spv",
                    vk::ShaderStageFlags::FRAGMENT,
                    pipeline_cfg.watch_shaders,
                )?;

            let mut pipeline = pipeline_builder.build()?;
            resource_registry
                .register_pipeline(pipeline.pipeline, &[pipeline_layout_id, render_pass_id])
                .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;
            pipeline.mark_managed_by_registry();

            // Create Shadow Pipeline
            let (shadow_pipeline, shadow_pipeline_layout) = if let Some(shadow_map) =
                shadow_feature.shadow_map()
            {
                let shadow_push_range = vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: 128, // mat4 lightSpace + mat4 model
                };

                let shadow_pipeline_layout =
                    vulkan::PipelineLayout::builder(Arc::clone(&vulkan_device.device))
                        .add_push_constant(shadow_push_range)
                        .build()?;

                let shadow_builder = vulkan::Pipeline::builder(Arc::clone(&vulkan_device.device))
                    .with_layout(shadow_pipeline_layout.handle())
                    .with_render_pass(shadow_map.render_pass)
                    .with_extent(vk::Extent2D {
                        width: shadow_map.resolution,
                        height: shadow_map.resolution,
                    })
                    .with_pipeline_cache(pipeline_cache.handle())
                    .with_depth_format(vk::Format::D32_SFLOAT)
                    .with_cull_mode(vk::CullModeFlags::FRONT)
                    .add_shader_with_options(
                        "shaders/shadow.vert.spv",
                        vk::ShaderStageFlags::VERTEX,
                        false,
                    )?
                    .add_shader_with_options(
                        "shaders/shadow.frag.spv",
                        vk::ShaderStageFlags::FRAGMENT,
                        false,
                    )?;

                let shadow_pipeline = shadow_builder.build()?;
                (Some(shadow_pipeline), Some(shadow_pipeline_layout))
            } else {
                (None, None)
            };

            let mut mesh = Mesh::create_cube();
            log::trace!("Ensuring cube mesh textures...");
            mesh.ensure_texture(
                Arc::clone(&allocator),
                Arc::clone(&vulkan_device.device),
                command_manager.upload_command_pool_handle(),
                vulkan_device.graphics_queue,
            )?;
            log::trace!("Cube mesh textures ready, registering with model renderer...");
            model_renderer.ensure_mesh(
                &mesh.name,
                &mesh,
                command_manager.upload_command_pool_handle(),
                vulkan_device.graphics_queue,
            )?;
            log::trace!("Cube mesh registered successfully");

            let material = Material::default();
            let transform = Transform::identity();
            let transform_matrix = transform.model_matrix();
            let mut mesh_registry = HashMap::new();
            mesh_registry.insert(0, mesh.name.clone());
            let mut material_registry = HashMap::new();
            material_registry.insert(0, material.clone());
            let mut mesh_texture_sets = HashMap::new();
            let mut mesh_texture_flags = HashMap::new();

            let mut initial_binding = TextureBindingInfo::from_mesh(&mesh, &default_texture);
            // Phase 6: Register textures with BindlessManager
            initial_binding.register_with_bindless(&mut bindless_manager)?;
            log::info!(
                "Registered mesh textures with BindlessManager: {:?}",
                initial_binding.bindless_indices
            );

            let initial_texture_set =
                Self::prepare_texture_set(&mut descriptor_manager, None, initial_binding)?;
            let initial_flags = initial_binding.flags();
            mesh_texture_sets.insert(mesh.name.clone(), initial_texture_set);
            mesh_texture_flags.insert(mesh.name.clone(), initial_flags);
            let start_time = Instant::now();

            log::info!("Ash renderer (Phase 5 - Stable) initialized successfully!");

            Ok(Self {
                buffer_pool,
                resource_registry,
                feature_manager,
                _pipeline_cache: pipeline_cache,
                command_manager,
                worker_count,
                command_buffers,
                frame_syncs,
                current_frame: 0,
                default_texture,
                model_renderer,
                draw_items: vec![DrawItem {
                    key: mesh.name.clone(),
                    transform: transform_matrix,
                    material: material.clone(),
                    texture_flags: initial_flags,
                    texture_set: initial_texture_set,
                    bindless_indices: initial_binding.bindless_indices,
                }],
                swapchain,
                render_pass: Some(render_pass),
                render_pass_id: Some(render_pass_id),
                pipeline: Some(pipeline),
                depth_buffer: Some(depth_buffer),
                mesh: Some(mesh),
                material,
                transform,
                uniform_buffers,
                material_buffers,
                pipeline_layout: Some(pipeline_layout),
                descriptor_manager: Some(descriptor_manager),
                framebuffers,
                framebuffer_ids,
                start_time,
                allocator,
                vulkan_device,
                mesh_registry,
                mesh_texture_sets,
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
                gpu_profiler: None, // Initialized lazily when diagnostics enabled
                diagnostics_overlay: DiagnosticsOverlay::new(),
                shadow_feature,
                shadow_pipeline,
                shadow_pipeline_layout,
                bindless_manager: Some(bindless_manager),
            })
        }
    }

    fn prepare_texture_set(
        descriptor_manager: &mut vulkan::DescriptorManager,
        existing: Option<vk::DescriptorSet>,
        binding: TextureBindingInfo,
    ) -> Result<vk::DescriptorSet> {
        let default_set = descriptor_manager.default_texture_array_set();
        if binding.has_any_texture() {
            let set = match existing {
                Some(set) if set != default_set => set,
                _ => descriptor_manager.allocate_material_texture_set()?,
            };
            let bindings = binding.descriptor_bindings();
            descriptor_manager.bind_material_textures(set, &bindings)?;
            Ok(set)
        } else {
            Ok(default_set)
        }
    }

    fn worker_index_for_frame(&self, frame_index: usize) -> usize {
        compute_worker_index(self.worker_count, frame_index)
    }

    fn update_mesh_texture_set(
        &mut self,
        key: &str,
        binding: TextureBindingInfo,
    ) -> Result<vk::DescriptorSet> {
        let descriptor_manager = self
            .descriptor_manager
            .as_mut()
            .ok_or_else(|| AshError::VulkanError("Descriptor manager not available".to_string()))?;
        let existing = self.mesh_texture_sets.get(key).copied();
        Self::prepare_texture_set(descriptor_manager, existing, binding)
    }

    /// Set mesh to render
    pub fn set_mesh(&mut self, mut mesh: Mesh) {
        unsafe {
            let upload_pool = self.command_manager.upload_command_pool_handle();
            if let Err(e) = mesh.ensure_texture(
                Arc::clone(&self.allocator),
                Arc::clone(&self.vulkan_device.device),
                upload_pool,
                self.vulkan_device.graphics_queue,
            ) {
                log::error!("Failed to ensure mesh texture: {e}");
            }

            let key = mesh.name.clone();
            match self.model_renderer.ensure_mesh(
                &key,
                &mesh,
                upload_pool,
                self.vulkan_device.graphics_queue,
            ) {
                Ok(_) => {
                    self.mesh_texture_sets.clear();
                    self.mesh_texture_flags.clear();
                    let binding = TextureBindingInfo::from_mesh(&mesh, &self.default_texture);
                    match self.update_mesh_texture_set(&key, binding) {
                        Ok(texture_set) => {
                            self.mesh_texture_sets.insert(key.clone(), texture_set);
                            self.mesh_texture_flags.insert(key.clone(), binding.flags());
                            self.draw_items.clear();
                            self.draw_items.push(DrawItem {
                                key: key.clone(),
                                transform: self.transform.model_matrix(),
                                material: self.material.clone(),
                                texture_flags: binding.flags(),
                                texture_set,
                                bindless_indices: binding.bindless_indices,
                            });
                            self.mesh_registry.clear();
                            self.mesh_registry.insert(0, key.clone());
                            self.material_registry.insert(0, self.material.clone());
                            self.mesh = Some(mesh);
                        }
                        Err(err) => {
                            log::error!(
                                "Failed to update texture descriptor set for mesh '{key}': {err}"
                            );
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to upload mesh via ModelRenderer: {e}");
                }
            }
        }
    }

    pub fn register_mesh_handle(&mut self, handle: u32, mesh: &mut Mesh) -> Result<()> {
        unsafe {
            let key = mesh.name.clone();
            let upload_pool = self.command_manager.upload_command_pool_handle();
            mesh.ensure_texture(
                Arc::clone(&self.allocator),
                Arc::clone(&self.vulkan_device.device),
                upload_pool,
                self.vulkan_device.graphics_queue,
            )?;

            self.model_renderer.ensure_mesh(
                &key,
                mesh,
                upload_pool,
                self.vulkan_device.graphics_queue,
            )?;

            let binding = TextureBindingInfo::from_mesh(mesh, &self.default_texture);
            match self.update_mesh_texture_set(&key, binding) {
                Ok(texture_set) => {
                    self.mesh_texture_sets.insert(key.clone(), texture_set);
                    self.mesh_texture_flags.insert(key.clone(), binding.flags());
                }
                Err(err) => {
                    log::error!(
                        "Failed to update texture descriptor set for registered mesh '{key}': {err}"
                    );
                }
            }
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
                    let texture_set = self
                        .mesh_texture_sets
                        .get(mesh_key)
                        .copied()
                        .or_else(|| {
                            self.descriptor_manager
                                .as_ref()
                                .map(|manager| manager.default_texture_array_set())
                        })
                        .unwrap_or(vk::DescriptorSet::null());
                    self.draw_items.push(DrawItem {
                        key: mesh_key.clone(),
                        transform: command.transform,
                        material: material.clone(),
                        texture_flags,
                        texture_set,
                        bindless_indices: [-1; 5], // TODO: Get from mesh registry
                    });
                }
            }
        }

        if self.draw_items.is_empty() {
            if let Some(mesh) = self.mesh.as_ref() {
                let texture_flags = self
                    .mesh_texture_flags
                    .get(&mesh.name)
                    .copied()
                    .unwrap_or_default();
                let texture_set = self
                    .mesh_texture_sets
                    .get(&mesh.name)
                    .copied()
                    .or_else(|| {
                        self.descriptor_manager
                            .as_ref()
                            .map(|manager| manager.default_texture_array_set())
                    })
                    .unwrap_or(vk::DescriptorSet::null());
                self.draw_items.push(DrawItem {
                    key: mesh.name.clone(),
                    transform: self.transform.model_matrix(),
                    material: self.material.clone(),
                    texture_flags,
                    texture_set,
                    bindless_indices: [-1; 5], // TODO: Get from mesh registry
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
            // Wait for idle to prevent using old resources while resizing
            unsafe {
                let _ = self.vulkan_device.device.device_wait_idle();
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
                // Window minimized; skip until we get a valid extent.
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
        if self.swapchain.is_none() {
            log::warn!("Skipping swapchain recreation in headless mode (not implemented)");
            return Ok(());
        }

        let old_swapchain = unsafe {
            if let Some(ref mut swapchain) = self.swapchain {
                Some(swapchain.recreate(&self.vulkan_device)?)
            } else {
                self.swapchain = Some(vulkan::SwapchainWrapper::new(&self.vulkan_device)?);
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

        // CRITICAL: Cleanup in correct order (dependents first)
        // 1. First destroy framebuffers (they reference render pass + depth buffer + image views)
        self.cleanup_framebuffers();
        // 2. Then destroy render pass
        self.cleanup_render_pass();
        // 3. Then update image views (destroys old ones)
        self.update_image_views(&image_views)?;
        // 4. Then recreate depth buffer (can now safely destroy old one)
        self.recreate_depth_buffer(swapchain_extent)?;
        // 5. Finally create new render pass and framebuffers
        self.create_render_pass_and_framebuffers(swapchain_extent, swapchain_format, &image_views)?;

        self.recreate_frame_syncs(self.framebuffers.len())?;
        self.recreate_command_buffers()?;
        self.recreate_uniform_buffers(self.framebuffers.len())?;
        self.recreate_descriptor_sets()?;

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
            if let Err(e) = self.resource_registry.cleanup_resource(id) {
                log::warn!("Failed to cleanup framebuffer {id}: {e}");
            }
        }
    }

    fn cleanup_render_pass(&mut self) {
        if let Some(render_pass_id) = self.render_pass_id.take() {
            if let Err(e) = self.resource_registry.cleanup_resource(render_pass_id) {
                log::warn!("Failed to cleanup render pass: {e}");
            }
        }
    }

    fn update_image_views(&mut self, image_views: &[vk::ImageView]) -> Result<()> {
        for id in self.swapchain_image_view_ids.drain(..) {
            if let Err(e) = self.resource_registry.cleanup_resource(id) {
                log::warn!("Failed to cleanup old swapchain image view {id}: {e}");
            }
        }

        self.swapchain_image_view_ids.clear();
        for &view in image_views {
            let id = self
                .resource_registry
                .register_image_view(view)
                .map_err(|e| {
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
            if let Err(e) = self.resource_registry.cleanup_resource(id) {
                log::warn!("Failed to cleanup old depth buffer: {e}");
            }
        }

        let mut depth_buffer = unsafe {
            DepthBuffer::new(
                Arc::clone(&self.vulkan_device.device),
                Arc::clone(&self.allocator),
                extent.width,
                extent.height,
            )?
        };

        let depth_buffer_id = depth_buffer
            .register_with_registry(&self.resource_registry)
            .map_err(|e| AshError::VulkanError(format!("Failed to register depth buffer: {e}")))?;

        self.depth_buffer = Some(depth_buffer);
        self.depth_buffer_id = Some(depth_buffer_id);

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

        let mut render_pass = vulkan::RenderPass::builder(Arc::clone(&self.vulkan_device.device))
            .with_swapchain_color(color_format)
            .with_depth_attachment(depth_buffer.format())
            .build()?;

        let render_pass_id = self
            .resource_registry
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

        for (index, &view) in image_views.iter().enumerate() {
            let attachments = [view, depth_buffer.view()];
            let framebuffer = vulkan::Framebuffer::new(
                Arc::clone(&self.vulkan_device.device),
                self.render_pass
                    .as_ref()
                    .expect("render pass just created")
                    .handle(),
                &attachments,
                extent,
            )?;

            let framebuffer_id = self
                .resource_registry
                .register_framebuffer(
                    framebuffer.handle(),
                    &[
                        render_pass_id,
                        depth_buffer_id,
                        self.swapchain_image_view_ids[index],
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

        self.framebuffers = framebuffers;
        self.framebuffer_ids = framebuffer_ids;

        Ok(())
    }

    fn recreate_frame_syncs(&mut self, count: usize) -> Result<()> {
        for (image_available_id, render_finished_id, fence_id) in self.frame_sync_ids.drain(..) {
            if let Err(e) = self.resource_registry.cleanup_resource(image_available_id) {
                log::warn!("Failed to cleanup image-available semaphore: {e}");
            }
            if let Err(e) = self.resource_registry.cleanup_resource(render_finished_id) {
                log::warn!("Failed to cleanup render-finished semaphore: {e}");
            }
            if let Err(e) = self.resource_registry.cleanup_resource(fence_id) {
                log::warn!("Failed to cleanup in-flight fence: {e}");
            }
        }

        self.frame_syncs.clear();

        let mut frame_syncs = Vec::with_capacity(count);
        let mut frame_sync_ids = Vec::with_capacity(count);

        for _ in 0..count {
            let mut sync = vulkan::FrameSync::new(Arc::clone(&self.vulkan_device.device))?;
            let image_available_id = self
                .resource_registry
                .register_semaphore(sync.image_available)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to register image-available semaphore: {e}"
                    ))
                })?;
            let render_finished_id = self
                .resource_registry
                .register_semaphore(sync.render_finished)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to register render-finished semaphore: {e}"
                    ))
                })?;
            let fence_id = self
                .resource_registry
                .register_fence(sync.in_flight)
                .map_err(|e| {
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
        self.command_manager
            .reset_primary_pool(vk::CommandPoolResetFlags::RELEASE_RESOURCES)?;

        self.command_buffers = self
            .command_manager
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
                let mut buffer = UniformBuffer::new(
                    Arc::clone(&self.allocator),
                    Arc::clone(&self.vulkan_device.device),
                )?;

                {
                    // Initialize with identity matrices - caller will provide real values via render_frame
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
        if let Some(manager) = self.descriptor_manager.as_mut() {
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
        self.resize_if_needed()?;
        if self.resize_pending {
            return Ok(());
        }

        // CRITICAL FIX: Flush old swapchain handles to prevent memory leak
        if self.swapchain_cleanup_pending {
            self.flush_old_swapchains();
        }

        // HOT-RELOAD: Check for shader changes (logs changes, full recompilation is TODO)
        if let Some(ref mut pipeline) = self.pipeline {
            if let Ok(changed) = pipeline.detect_shader_changes() {
                if changed {
                    log::warn!("Shader files changed - restart required for hot-reload (pipeline recompilation not yet implemented)");
                }
            }
        }

        unsafe {
            let swapchain_extent = if let Some(swapchain) = self.swapchain.as_ref() {
                swapchain.extent
            } else {
                self.pending_extent.unwrap_or(vk::Extent2D {
                    width: 800,
                    height: 600,
                })
            };

            let pipeline = self
                .pipeline
                .as_ref()
                .ok_or(AshError::VulkanError("Pipeline not available".to_string()))?;
            let render_pass = self.render_pass.as_ref().ok_or(AshError::VulkanError(
                "Render pass not available".to_string(),
            ))?;

            // ===== FENCE WAIT MUST HAPPEN BEFORE UNIFORM BUFFER UPDATE =====
            // Wait for the current frame's previous submission to complete
            // before we write new data to the uniform buffer.
            let frame_index = self.current_frame;
            let command_buffer = *self
                .command_buffers
                .get(frame_index)
                .ok_or_else(|| AshError::VulkanError("Command buffer index out of range".into()))?;
            let frame_sync = self
                .frame_syncs
                .get(frame_index)
                .ok_or_else(|| AshError::VulkanError("Frame sync index out of range".into()))?;

            self.vulkan_device
                .device
                .wait_for_fences(&[frame_sync.in_flight], true, u64::MAX)?;
            self.vulkan_device
                .device
                .reset_fences(&[frame_sync.in_flight])?;

            // CRITICAL FIX: Advance descriptor manager frame to clear stale dynamic sets
            if let Some(ref mut dm) = self.descriptor_manager {
                dm.next_frame();
            }

            // NOW it's safe to update the uniform buffer since the GPU is done reading it
            {
                let uniform_buffer = &mut self.uniform_buffers[frame_index];

                let elapsed = self.start_time.elapsed().as_secs_f32();
                let mut feature_ctx = FeatureFrameContext {
                    device: self.vulkan_device.device.as_ref(),
                    descriptor_manager: self.descriptor_manager.as_ref(),
                    transform: &mut self.transform,
                    auto_rotate: false, // Auto-rotate now handled by examples
                    elapsed_seconds: elapsed,
                };
                self.feature_manager.before_frame(&mut feature_ctx);

                // Use matrices provided by caller (stateless rendering)
                let matrices = uniform_buffer.matrices_mut();
                matrices.model = self.transform.model_matrix();
                matrices.view = view;
                matrices.projection = projection;
                matrices.view_proj = projection * view;
                matrices.camera_pos = camera_pos.extend(1.0);
                let light_dir = glam::Vec3::new(-0.35, -1.0, -0.25).normalize();

                matrices.set_lighting(light_dir, glam::Vec3::splat(1.5), glam::Vec3::splat(0.35));

                // Set light-space matrix for shadow mapping
                let light_space_matrix = self.shadow_feature.light_space_matrix();
                matrices.set_light_space_matrix(light_space_matrix);
                matrices.normal_matrix = matrices.model.inverse().transpose();

                uniform_buffer.update()?;
            }

            let cmd_ctx = self.command_manager.context(command_buffer);
            cmd_ctx.reset()?;

            // Phase 5.5: Headless support - Skip acquire if no swapchain
            let (image_index, image_acquired) = if let Some(swapchain) = self.swapchain.as_ref() {
                match swapchain.acquire_next_image(frame_sync.image_available) {
                    Ok(index) => (index, true),
                    Err(AshError::SwapchainOutOfDate(_)) => {
                        self.request_swapchain_resize(swapchain_extent);
                        return Ok(());
                    }
                    Err(err) => return Err(err),
                }
            } else {
                (
                    self.current_frame as u32 % self.framebuffers.len() as u32,
                    false,
                )
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

                    // CRITICAL FIX: Bind frame descriptor set (set 0) for uniform buffer access
                    if let Some(ref dm) = self.descriptor_manager {
                        if let Some(frame_set) = dm.frame_set(frame_index) {
                            cmd_ctx.bind_descriptor_sets(
                                vk::PipelineBindPoint::GRAPHICS,
                                shadow_layout.handle(),
                                0, // Set 0: Frame uniforms
                                &[frame_set],
                                &[],
                            );
                        }
                    }

                    let light_space_matrix = self.shadow_feature.light_space_matrix();

                    // Draw all meshes
                    for item in &self.draw_items {
                        if let Some(uploaded) = self.model_renderer.get(&item.key) {
                            // Push constants: lightSpaceMatrix (64) + model (64)
                            let light_space_push =
                                crate::renderer::model_renderer::Mat4Push::from(light_space_matrix);
                            let model_push =
                                crate::renderer::model_renderer::Mat4Push::from(item.transform);

                            let mut push_data = Vec::with_capacity(128);
                            push_data.extend_from_slice(bytemuck::bytes_of(&light_space_push));
                            push_data.extend_from_slice(bytemuck::bytes_of(&model_push));

                            self.vulkan_device.device.cmd_push_constants(
                                command_buffer,
                                shadow_layout.handle(),
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                &push_data,
                            );

                            // Bind vertex buffers
                            let offsets = [0];
                            self.vulkan_device.device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[uploaded.vertex_buffer()],
                                &offsets,
                            );

                            if let Some(index_buffer) = uploaded.index_buffer() {
                                self.vulkan_device.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    index_buffer,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                self.vulkan_device.device.cmd_draw_indexed(
                                    command_buffer,
                                    uploaded.index_count(),
                                    1,
                                    0,
                                    0,
                                    0,
                                );
                            } else {
                                self.vulkan_device.device.cmd_draw(
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
                .render_pass(render_pass.handle())
                .framebuffer(framebuffer.handle())
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_extent,
                })
                .clear_values(&clear_values);

            cmd_ctx.begin_render_pass(&render_pass_begin, vk::SubpassContents::INLINE);
            cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);

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
                device: self.vulkan_device.device.as_ref(),
                descriptor_manager: self.descriptor_manager.as_ref(),
                command_buffer,
                transform: &self.transform,
            };

            self.feature_manager.render(&render_ctx);

            let pipeline_layout = self.pipeline_layout.as_ref().ok_or_else(|| {
                AshError::VulkanError("Pipeline layout not available".to_string())
            })?;
            let pipeline_layout_handle = pipeline_layout.handle();

            let default_texture_set = (|| -> Result<vk::DescriptorSet> {
                if let Some(manager) = self.descriptor_manager.as_ref() {
                    let frame_set = manager.frame_set(frame_index).ok_or_else(|| {
                        AshError::VulkanError("Frame descriptor set not available".to_string())
                    })?;
                    let material_set = manager.material_set(worker_index).ok_or_else(|| {
                        AshError::VulkanError("Material descriptor set not available".to_string())
                    })?;
                    let default_texture_set = manager.default_texture_array_set();
                    cmd_ctx.bind_descriptor_sets(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout_handle,
                        0,
                        &[frame_set, material_set, default_texture_set],
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

                    // Bind bindless descriptor set (set 4)
                    if let Some(ref bindless) = self.bindless_manager {
                        cmd_ctx.bind_descriptor_sets(
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout_handle,
                            4, // Set 4: Bindless textures
                            &[bindless.descriptor_set()],
                            &[],
                        );
                    }

                    Ok(default_texture_set)
                } else {
                    Ok(vk::DescriptorSet::null())
                }
            })()?;

            // Draw uploaded meshes in order
            for item in &self.draw_items {
                if let Some(uploaded) = self.model_renderer.get(&item.key) {
                    let texture_set = if item.texture_set != vk::DescriptorSet::null() {
                        item.texture_set
                    } else {
                        default_texture_set
                    };

                    if texture_set != vk::DescriptorSet::null() {
                        cmd_ctx.bind_descriptor_sets(
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout_handle,
                            2,
                            &[texture_set],
                            &[],
                        );
                    }

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
                        // Phase 6: Use bindless texture indices from DrawItem
                        uniform.set_texture_indices(
                            item.bindless_indices[0],
                            item.bindless_indices[1],
                            item.bindless_indices[2],
                            item.bindless_indices[3],
                            item.bindless_indices[4],
                        );
                        material_buffer.update()?;
                    }

                    let model_matrix = item.transform;
                    // Note: In draw_items path, uniform buffer already contains view/proj from render_frame call
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
            cmd_ctx.end()?;

            let wait_semaphores = if image_acquired {
                vec![frame_sync.image_available]
            } else {
                vec![]
            };
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [frame_sync.render_finished];
            let command_buffers_submit = [command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers_submit)
                .signal_semaphores(&signal_semaphores);

            self.command_manager.submit(
                self.vulkan_device.graphics_queue,
                &[submit_info],
                frame_sync.in_flight,
            )?;

            // Conditionally present if swapchain exists
            if let Some(swapchain_ref) = self.swapchain.as_ref() {
                let present_result = swapchain_ref.present(
                    self.vulkan_device.present_queue,
                    image_index,
                    frame_sync.render_finished,
                );

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
            } else {
                // Headless: Just wait for GPU or proceed
                // We don't present, so no need to check result
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

    /// Sets the MSAA preset (Off, X2, X4, X8)
    pub fn set_msaa_preset(&mut self, preset: MsaaPreset) {
        self.msaa_preset = preset;
        log::info!("MSAA preset set to {preset:?}");
        // Note: MSAA targets need to be recreated when preset changes
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
    // Post-Processing Initialization & Application
    // ──────────────────────────────────────────────────────────

    /// Initializes HDR framebuffer for post-processing
    ///
    /// Call this after renderer creation to enable HDR rendering.
    /// Note: This allocates GPU memory for the HDR buffer.
    pub fn initialize_hdr(&mut self) -> Result<()> {
        let extent = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
            .extent;

        unsafe {
            let hdr = hdr_framebuffer::HdrFramebuffer::new(
                Arc::clone(&self.vulkan_device.device),
                Arc::clone(&self.allocator),
                extent.width,
                extent.height,
            )?;
            self.hdr_framebuffer = Some(hdr);
            log::info!(
                "HDR framebuffer initialized ({}x{})",
                extent.width,
                extent.height
            );
        }

        Ok(())
    }

    /// Initializes the fullscreen pass for post-processing
    ///
    /// Call this after renderer creation to enable fullscreen effects.
    pub fn initialize_fullscreen_pass(&mut self) -> Result<()> {
        let format = self
            .swapchain
            .as_ref()
            .ok_or(AshError::VulkanError("Swapchain not available".to_string()))?
            .format;

        unsafe {
            let pass = fullscreen_pass::FullscreenPass::new(
                Arc::clone(&self.vulkan_device.device),
                format,
            )?;
            self.fullscreen_pass = Some(pass);
            log::info!("Fullscreen pass initialized");
        }

        Ok(())
    }

    /// Enables post-processing with default settings
    ///
    /// Convenience method that initializes HDR, fullscreen pass, and enables tonemapping.
    pub fn enable_post_processing(&mut self) -> Result<()> {
        self.initialize_hdr()?;
        self.initialize_fullscreen_pass()?;
        self.tonemapping_enabled = true;
        log::info!(
            "Post-processing enabled (tonemapping: exposure={}, gamma={})",
            self.tonemapping_exposure,
            self.tonemapping_gamma
        );
        Ok(())
    }

    /// Returns whether post-processing is ready (HDR and fullscreen pass initialized)
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

    /// Update diagnostics at end of frame
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
    /// This is called automatically when diagnostics mode is set to anything other than Off.
    pub fn initialize_gpu_profiler(&mut self) -> Result<()> {
        if self.gpu_profiler.is_some() {
            return Ok(());
        }

        let timestamp_period = self.vulkan_device.timestamp_period_ns;
        let timestamps_supported = timestamp_period > 0.0;

        unsafe {
            let profiler = GpuProfiler::new(
                Arc::clone(&self.vulkan_device.device),
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
            log::info!("Shutting down Ash renderer (Phase 5)...");

            let _ = self.vulkan_device.device.device_wait_idle();

            self.flush_old_swapchains();

            if let Err(e) = self.resource_registry.cleanup() {
                log::error!("Resource registry cleanup failed: {e}");
            }

            if let Some(manager) = self.descriptor_manager.take() {
                drop(manager);
            }

            self.feature_manager.cleanup();

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

            log::info!("Ash renderer shut down successfully (Phase 5)");
        }
    }
}
