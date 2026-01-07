use crate::{
    renderer::{
        diagnostics::{
            DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler,
        },
        features::{
            AutoRotateFeature, FeatureFrameContext, FeatureManager, FeatureRenderContext,
            RenderFeature, ShadowFeature,
        },
        forward_plus_integration::ForwardPlusIntegration,
        fullscreen_pass, hdr_framebuffer,
        hiz_pass::HiZPass,
        indirect_draw::IndirectDrawPass,
        instancing::{BatchKey, InstanceData, InstancingManager},
        model_renderer::{DrawContext, MaterialPushConstants, MeshPushConstants, ModelRenderer},
        occlusion_culling::{CullBoundingBox, OcclusionCulling},
        pass_manager::{RenderPassManager, RenderingMode},
        resource_registry::{ResourceId, ResourceRegistry},
        resources,
        resources::uniform::{StorageBuffer, UniformBuffer},
        ssgi_pass::{SsgiPass, SsgiQuality},
        temporal_upscaling::{VsrPass, VsrQuality},
        vram_budget, DepthBuffer, GBuffer, Material, MaterialHandle, MaterialManager, Mesh,
        PipelineCache, SkinnedVertex, Texture, TextureData, Transform,
    },
    vulkan::{self, Allocator, BindlessManager},
    AshError, Result,
};

use ash::vk;
use bytemuck::Pod;
use glam::{Mat4, Vec3, Vec4};
use parking_lot::Mutex;
use rayon::prelude::*;
use resources::BufferPool;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::renderer::resources::buffer::BufferHandle;
use crate::renderer::resources::mesh::{MaterialDescriptor, MeshDescriptor};

#[derive(Clone, Copy, Debug, Default)]
pub enum MsaaPreset {
    #[default]
    Off,
    X2,
    X4,
    X8,
}

#[derive(Clone, Debug)]
pub struct RenderCommand {
    /// Handle identifying the mesh to render
    pub mesh_handle: u32,
    /// Handle identifying the material to use
    pub material_handle: MaterialHandle,
    /// Transform matrix for positioning the mesh in world space
    pub transform: Mat4,
    /// Whether this is a skinned mesh
    pub is_skinned: bool,
    /// Offset into the joint matrices SSBO (for skeletal animation)
    pub joint_offset: u32,
    /// Whether this object should cast shadows
    pub cast_shadows: bool,
}

impl Default for RenderCommand {
    fn default() -> Self {
        Self {
            mesh_handle: 0,
            material_handle: MaterialHandle::null(),
            transform: Mat4::IDENTITY,
            is_skinned: false,
            joint_offset: 0,
            cast_shadows: true,
        }
    }
}

struct RendererResources {
    uniform_buffers: Vec<UniformBuffer>,
    joint_matrices_buffer: Vec<resources::JointMatricesBuffer>,
    default_texture: Texture,
    material_storage_buffer: StorageBuffer<resources::uniform::MaterialUniform>,
    instance_buffers: Vec<resources::InstanceBuffer>,
}

fn compute_worker_index(worker_count: usize, frame_index: usize) -> usize {
    if worker_count == 0 {
        0
    } else {
        frame_index % worker_count
    }
}

#[allow(dead_code)]
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

#[derive(Clone, Debug)]
pub struct RendererConfig {
    pub pipeline: PipelineConfig,
    pub texture_compression: bool,
    pub allow_auto_material: bool,
    pub strict_mode: bool,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            texture_compression: true,
            allow_auto_material: true,
            strict_mode: false,
        }
    }
}

/// Main rendering system.
///
/// # Safety Contract
///
/// The `Renderer` manages complex GPU resource lifecycles. To ensure memory safety:
///
/// 1. **Drop Order**: Rust's drop order (top-to-bottom for fields) is critical. Buffers,
///    textures, and pipelines must be destroyed before the `Allocator` and `Device`.
/// 2. **Map Safety**: All `MapGuard` instances must be dropped before the underlying
///    buffer is destroyed or the allocator is dropped.
/// 3. **Validation**: Use Vulkan validation layers (`VK_LAYER_KHRONOS_validation`) in
///    development to verify that no resources leak or are accessed after destruction.
pub struct Renderer {
    // Resources dependent on allocator/device - dropped in reverse order.
    buffer_pool: Arc<BufferPool>,
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
    material_storage_buffer: Option<StorageBuffer<resources::uniform::MaterialUniform>>,
    #[allow(dead_code)]
    material_buffer_index: u32,
    pipeline_layout: Option<vulkan::PipelineLayout>,
    pipeline_layout_id: Option<ResourceId>,
    descriptors: Option<vulkan::DescriptorManager>,
    framebuffers: Vec<vulkan::Framebuffer>,
    framebuffer_ids: Vec<ResourceId>,
    start_time: Instant,
    pub mesh: Option<Mesh>,
    material: Material,
    transform: Transform,
    mesh_data: Vec<MeshData>, // Indexed by mesh handle for O(1) access
    material_manager: MaterialManager,
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
    pub tonemapping_enabled: bool,
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
    // Lighting
    light_direction: Vec3,
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    // Post-processing descriptors
    post_descriptor_pool: vk::DescriptorPool,
    post_descriptor_sets: Vec<vk::DescriptorSet>,
    post_pipeline: Option<vk::Pipeline>,
    post_framebuffers: Vec<vulkan::Framebuffer>,
    // GPU skinning
    joint_matrices_buffer: Vec<resources::JointMatricesBuffer>,
    max_bones: usize,
    skinned_pipeline: Option<vulkan::Pipeline>,
    skinned_pipeline_id: Option<ResourceId>,
    vram_budget: vram_budget::VramBudget,
    texture_compression: bool,
    instancing_manager: InstancingManager,
    instance_buffer: Vec<resources::InstanceBuffer>, // One per frame
    // Pass management
    pass_manager: RenderPassManager,
    // Image-Based Lighting
    brdf_lut_pass: Option<crate::renderer::features::brdf_lut::BrdfLutPass>,
    irradiance_map: Option<resources::ImageHandle>,
    prefiltered_map: Option<resources::ImageHandle>,
    ibl_sampler: vk::Sampler,
    ibl_manager: crate::renderer::features::ibl_manager::IblManager,
    allow_auto_material: bool,
    strict_mode: bool,
    // Headless support
    readback_buffer: Option<BufferHandle>,
    last_image_index: u32,

    // Bindless Buffer Indices
    pub instance_buffer_indices: Vec<u32>,
    pub joint_buffer_indices: Vec<u32>,

    // Core foundation - dropped in declaration order (Top to Bottom)
    // So these should be at the VERY END to be dropped LAST.
    // However, they are dropped Top to Bottom?
    // WAIT! In Rust, fields are dropped in the order they are DECLARED.
    // So the FIRST field is dropped FIRST.
    // This means foundation should be at the BOTTOM so they are dropped LAST.
    texture_streamer: Mutex<Option<resources::TextureStreamer>>,
    resources: Arc<ResourceRegistry>,
    alloc: Arc<vulkan::Allocator>,
    device: vulkan::VulkanDevice,
}

#[derive(Clone)]
#[allow(dead_code)]
struct DrawItem {
    key: Arc<str>,
    transform: Mat4,
    material: Material,
    material_handle: MaterialHandle,
    texture_flags: TexturePresenceFlags,
    texture_indices: [i32; 4], // base, normal, mr, occ
    emissive_index: i32,
    is_skinned: bool,
    joint_offset: u32,
    alpha_cutoff: f32,
    cast_shadows: bool,
}

#[derive(Copy, Clone, Default, Debug)]
pub struct TexturePresenceFlags {
    pub base_color: bool,
    pub normal: bool,
    pub metallic_roughness: bool,
    pub occlusion: bool,
    pub emissive: bool,
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

/// Consolidated mesh data for efficient lookup.
/// Replaces multiple HashMap lookups with a single Vec access.
#[derive(Clone, Debug)]
pub struct MeshData {
    pub name: Arc<str>,
    pub texture_indices: [i32; 4], // base, normal, mr, occlusion
    pub emissive_index: i32,
    pub texture_flags: TexturePresenceFlags,
    pub material_handle: MaterialHandle,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            name: Arc::from(""),
            texture_indices: [-1, -1, -1, -1],
            emissive_index: -1,
            texture_flags: TexturePresenceFlags::default(),
            material_handle: MaterialHandle { index: 0, version: 0 },
        }
    }
}

/// Internal struct for swapchain-related resources used during initialization.
struct SwapchainData {
    swapchain: vulkan::SwapchainWrapper,
    swapchain_image_view_ids: Vec<ResourceId>,
    depth_buffer: DepthBuffer,
    depth_buffer_id: ResourceId,
    render_pass: vulkan::RenderPass,
    render_pass_id: ResourceId,
}

/// Internal struct for frame-related resources used during initialization.
struct FrameData {
    framebuffers: Vec<vulkan::Framebuffer>,
    framebuffer_ids: Vec<ResourceId>,
    command_manager: vulkan::CommandBufferManager,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_syncs: Vec<vulkan::FrameSync>,
    frame_sync_ids: Vec<(ResourceId, ResourceId, ResourceId)>,
    worker_count: usize,
}

impl Renderer {
    /// Initializes the renderer.
    pub fn new<S: vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        unsafe {
            let instance = Arc::new(vulkan::VulkanInstance::new(
                surface_provider,
                cfg!(debug_assertions),
            )?);
            let device =
                vulkan::VulkanDevice::new(Arc::clone(&instance), surface_provider.is_headless())?;
            let alloc = Arc::new(vulkan::Allocator::new(&device)?);
            let resources = Arc::new(ResourceRegistry::new(Arc::clone(&device.device)));
            let dev_mem_props = device.memory_properties;
            let mut vram_budget = vram_budget::VramBudget::new(&dev_mem_props);

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
            let texture_compression = renderer_config.texture_compression;
            let pipeline_cfg = &renderer_config.pipeline;
            let buffer_pool = Arc::new(BufferPool::new(Arc::clone(&alloc)));
            let (width, height) = surface_provider.physical_size();
            let extent = vk::Extent2D { width, height };
            let swapchain_data = Self::create_swapchain_data(&device, &alloc, &resources, extent)?;
            let frame_data = Self::create_frame_resources(&device, &resources, &swapchain_data)?;

            // Unpack for use in the rest of initialization
            let SwapchainData {
                swapchain,
                swapchain_image_view_ids,
                depth_buffer,
                depth_buffer_id,
                render_pass,
                render_pass_id,
            } = swapchain_data;

            let FrameData {
                framebuffers,
                framebuffer_ids,
                command_manager,
                command_buffers,
                frame_syncs,
                frame_sync_ids,
                worker_count,
            } = frame_data;

            let mut model_renderer =
                ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device.device));

            let aspect = swapchain.extent.width as f32 / swapchain.extent.height as f32;
            let max_bones = 1024;
            let renderer_resources = Self::init_resources(
                &alloc,
                &device,
                command_manager.upload_command_pool_handle(),
                worker_count,
                framebuffers.len(),
                aspect,
                max_bones,
            )?;
            let RendererResources {
                uniform_buffers,
                joint_matrices_buffer,
                default_texture,
                material_storage_buffer,
                instance_buffers,
            } = renderer_resources;

            let material = Material::default();

            let mut descriptor_manager = vulkan::DescriptorManager::new(
                Arc::clone(&device.device),
                framebuffers.len() as u32,
                Some(Arc::clone(&resources)),
            )?;

            let mut bindless_manager = crate::vulkan::BindlessManager::new(
                Arc::clone(&device.device),
                descriptor_manager.allocator_mut(),
                crate::vulkan::BindlessManager::DEFAULT_MAX_TEXTURES,
            )?;

            let buffer_size =
                std::mem::size_of::<crate::renderer::resources::uniform::MvpMatrices>()
                    as vk::DeviceSize;
            for set_index in 0..descriptor_manager.frame_set_count() {
                if let Some(ubo) = uniform_buffers.get(set_index) {
                    descriptor_manager.bind_frame_uniform(set_index, ubo.buffer, buffer_size)?;
                }
            }

            // Register global material storage buffer (Set 1)
            let max_materials = material_storage_buffer.capacity();
            let material_size = (max_materials
                * std::mem::size_of::<crate::renderer::resources::uniform::MaterialUniform>())
                as vk::DeviceSize;
            let material_buffer_index = bindless_manager.add_material_buffer(
                material_storage_buffer.buffer,
                0,
                material_size,
            )?;
            log::info!(
                "Registered global material buffer at bindless index {material_buffer_index}"
            );

            // Register instance buffers (Bindless Storage Buffers - Set 1, Binding 2)
            let mut instance_buffer_indices = Vec::with_capacity(instance_buffers.len());
            let instance_buffer_size = (crate::renderer::occlusion_culling::MAX_CULLABLE_OBJECTS
                * std::mem::size_of::<crate::renderer::occlusion_culling::CullObjectData>())
                as vk::DeviceSize;
            for buffer in &instance_buffers {
                let index = bindless_manager
                    .add_instance_buffer(buffer.buffer, 0, instance_buffer_size)
                    .unwrap_or(0);
                instance_buffer_indices.push(index);
                log::info!("Registered instance buffer at bindless index {index}");
            }

            // Register joint matrices buffers (Bindless Storage Buffers - Set 1, Binding 3)
            let mut joint_buffer_indices = Vec::with_capacity(joint_matrices_buffer.len());
            let joint_size = (max_bones * std::mem::size_of::<Mat4>()) as vk::DeviceSize;
            for buffer in &joint_matrices_buffer {
                let index = bindless_manager
                    .add_indirect_buffer(buffer.buffer(), 0, joint_size)
                    .unwrap_or(0);
                joint_buffer_indices.push(index);
                log::info!("Registered joint buffer at bindless index {index}");
            }

            // Register default texture as a fallback for all slots.
            let default_tex_index = bindless_manager
                .add_sampled_image(default_texture.view(), default_texture.sampler())
                .unwrap_or(0); // Fallback to index 0 if registration fails.
            log::info!("Registered default texture at bindless index {default_tex_index}");

            // Forward+ lighting integration
            let mut forward_plus =
                ForwardPlusIntegration::new(Arc::clone(&device.device), &alloc.vma)?;
            forward_plus.init(&alloc.vma);
            forward_plus.on_resize(swapchain.extent.width, swapchain.extent.height);

            let set_layouts = [
                descriptor_manager.frame_layout(),
                bindless_manager.layout(), // Set 1: Bindless (Textures, Materials, Instances, Joints)
                descriptor_manager.environment_layout(), // Set 2: Global Environment (Shadow + IBL)
                forward_plus.layout(),     // Set 3: Forward+ lights
            ];

            let (pipeline_layout, pipeline_layout_id, pipeline, pipeline_id) =
                Self::create_main_pipeline(
                    &device,
                    &resources,
                    render_pass.handle(),
                    swapchain.extent,
                    pipeline_cache.handle(),
                    depth_buffer.format(),
                    pipeline_cfg,
                    &set_layouts,
                )?;

            // Create Shadow Pipeline
            let (shadow_pipeline, shadow_pipeline_layout) =
                if let Some(shadow_map) = shadow_feature.shadow_map() {
                    let (p, l) = Self::create_shadow_pipeline(
                        &device,
                        pipeline_cache.handle(),
                        shadow_map,
                        descriptor_manager.frame_layout(),
                        bindless_manager.layout(),
                    )?;
                    (Some(p), Some(l))
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
                &mut vram_budget,
                texture_compression,
            )?;

            // Initialize Texture Streamer
            let transfer_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(device.graphics_queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let transfer_command_pool = device
                .device
                .create_command_pool(&transfer_pool_info, None)?;

            let texture_streamer = resources::TextureStreamer::new(
                Arc::clone(&alloc),
                Arc::clone(&device.device),
                transfer_command_pool,
                device.present_queue,
            );
            log::trace!("Cube mesh textures ready, registering with model renderer...");
            // Register mesh textures with bindless manager FIRST
            if let Some(tex) = mesh.texture.as_deref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.texture_index = Some(idx);
            }
            if let Some(tex) = mesh.normal_texture.as_deref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.normal_texture_index = Some(idx);
            }
            if let Some(tex) = mesh.metallic_roughness_texture.as_deref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.metallic_roughness_texture_index = Some(idx);
            }
            if let Some(tex) = mesh.occlusion_texture.as_deref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.occlusion_texture_index = Some(idx);
            }
            if let Some(tex) = mesh.emissive_texture.as_deref() {
                let idx = bindless_manager.add_sampled_image(tex.view(), tex.sampler())?;
                mesh.emissive_texture_index = Some(idx);
            }

            model_renderer.ensure_mesh(
                &mesh.name,
                &mesh,
                command_manager.upload_command_pool_handle(),
                device.graphics_queue,
            )?;
            log::trace!("Cube mesh registered successfully");

            let transform = Transform::identity();
            let transform_matrix = transform.model_matrix();

            let initial_flags = TexturePresenceFlags::from_mesh(&mesh);

            let mut material_manager = MaterialManager::new();
            let initial_material_handle = material_manager.register_material(material.clone());

            // Initialize mesh_data with the cube mesh and CORRECT indices
            let mesh_data = vec![MeshData {
                name: Arc::clone(&mesh.name),
                texture_indices: [
                    mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.metallic_roughness_texture_index
                        .map(|i| i as i32)
                        .unwrap_or(-1),
                    mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
                ],
                emissive_index: mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1),
                texture_flags: initial_flags,
                material_handle: initial_material_handle,
            }];

            // Mesh data already added to mesh_data Vec above
            let start_time = Instant::now();

            let swapchain_extent = swapchain.extent;

            let gbuffer = GBuffer::new(
                Arc::clone(&device.device),
                Arc::clone(&alloc),
                swapchain_extent.width,
                swapchain_extent.height,
            )?;

            // Image-Based Lighting Initialization
            let (brdf_lut_pass, ibl_sampler, ibl_manager) = Self::init_ibl(
                &device,
                &alloc,
                command_manager.upload_command_pool_handle(),
            )?;

            log::info!("Renderer initialization complete. Constructing struct.");

            // Create readback buffer if headless
            let readback_buffer = if device.headless {
                let buffer_size = (swapchain.extent.width * swapchain.extent.height * 4) as u64; // R8G8B8A8
                Some(BufferHandle::new_with_flags(
                    Arc::clone(&alloc),
                    buffer_size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    vk_mem::MemoryUsage::Auto,
                    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    Some("Headless Readback Buffer".to_string()),
                )?)
            } else {
                None
            };

            let renderer = Self {
                texture_streamer: Mutex::new(Some(texture_streamer)),
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
                    key: Arc::clone(&mesh.name),
                    transform: transform_matrix,
                    material: material.clone(),
                    material_handle: initial_material_handle,
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
                    is_skinned: false,
                    joint_offset: 0,
                    alpha_cutoff: material.alpha_cutoff,
                    cast_shadows: true,
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
                material_storage_buffer: Some(material_storage_buffer),
                material_buffer_index,
                pipeline_layout: Some(pipeline_layout),
                pipeline_layout_id: Some(pipeline_layout_id),
                descriptors: Some(descriptor_manager),
                framebuffers,
                framebuffer_ids,
                start_time,
                alloc,
                device,
                mesh_data,
                material_manager,
                swapchain_image_view_ids,
                depth_buffer_id: Some(depth_buffer_id),
                frame_sync_ids,
                old_swapchain_handles: Vec::new(),
                swapchain_cleanup_pending: false,
                resize_pending: false,
                pending_extent: Some(swapchain_extent),
                msaa_preset: MsaaPreset::default(),
                hdr_framebuffer: None,
                fullscreen_pass: None,
                tonemapping_enabled: true,
                tonemapping_exposure: 1.0,
                tonemapping_gamma: 2.2,
                bloom_enabled: true,
                bloom_intensity: 0.1,
                diagnostics: DiagnosticsState::default(),
                frame_profiler: FrameProfiler::new(),
                gpu_profiler: None,
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
                light_direction: Vec3::new(-0.35, -1.0, -0.25).normalize(),
                light_color: [1.5, 1.5, 1.5, 1.0],
                ambient_color: [0.0, 0.2, 0.8, 1.0], // Default to vibrant blue
                post_descriptor_pool: vk::DescriptorPool::null(),
                post_descriptor_sets: Vec::new(),
                post_pipeline: None,
                post_framebuffers: Vec::new(),
                joint_matrices_buffer,
                max_bones,
                skinned_pipeline: None,
                skinned_pipeline_id: None,
                vram_budget,
                texture_compression,
                instancing_manager: InstancingManager::new(),
                instance_buffer: instance_buffers,
                instance_buffer_indices,
                joint_buffer_indices,
                pass_manager: RenderPassManager::new(RenderingMode::GPUDriven),
                brdf_lut_pass: Some(brdf_lut_pass),
                irradiance_map: None,
                prefiltered_map: None,
                ibl_sampler,
                ibl_manager,
                allow_auto_material: renderer_config.allow_auto_material,
                strict_mode: renderer_config.strict_mode,
                readback_buffer,
                last_image_index: 0,
            };
            Ok(renderer)
        }
    }

    pub fn read_headless_image(&mut self) -> Result<Vec<u8>> {
        if !self.device.headless {
            return Err(AshError::VulkanError("Not in headless mode".to_string()));
        }

        let swapchain = self.swapchain.as_ref().ok_or(AshError::VulkanError(
            "Swapchain not initialized".to_string(),
        ))?;

        let readback_buffer = self.readback_buffer.as_mut().ok_or(AshError::VulkanError(
            "Readback buffer not initialized".to_string(),
        ))?;

        // Wait for the device to be idle to ensure rendering is complete
        unsafe {
            self.device.device.device_wait_idle().map_err(|e| {
                AshError::VulkanError(format!("Failed to wait for device idle: {e}"))
            })?;
        }

        let image_index = self.last_image_index;
        if image_index >= swapchain.images.len() as u32 {
            return Err(AshError::VulkanError("Invalid image index".to_string()));
        }
        let src_image = swapchain.images[image_index as usize];

        // Create a command buffer for the copy
        let cmd = self.cmds.get_transfer_command_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
                })?;

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    depth: 1,
                });

            self.device.device.cmd_copy_image_to_buffer(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buffer.handle(),
                &[copy_region],
            );

            // Add barrier to ensure write is visible to host
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(readback_buffer.handle())
                .offset(0)
                .size(vk::WHOLE_SIZE);

            self.device.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            self.device
                .device
                .end_command_buffer(cmd)
                .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

            let command_buffers = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            self.device
                .device
                .queue_submit(
                    self.device.graphics_queue,
                    &[submit_info],
                    vk::Fence::null(),
                )
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to submit copy command: {e}"))
                })?;

            self.device
                .device
                .queue_wait_idle(self.device.graphics_queue)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to wait for queue idle: {e}"))
                })?;

            self.device
                .device
                .free_command_buffers(self.cmds.upload_command_pool_handle(), &[cmd]);
        }

        // Map memory and read
        let size = (swapchain.extent.width * swapchain.extent.height * 4) as usize;
        let mut data = vec![0u8; size];

        unsafe {
            let ptr = self
                .alloc
                .vma
                .map_memory(readback_buffer.allocation_mut())
                .map_err(|e| AshError::VulkanError(format!("Failed to map memory: {e}")))?;

            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size);

            self.alloc
                .vma
                .unmap_memory(readback_buffer.allocation_mut());
        }

        Ok(data)
    }

    fn init_resources(
        alloc: &Arc<vulkan::Allocator>,
        device: &vulkan::VulkanDevice,
        command_pool: vk::CommandPool,
        _worker_count: usize,
        frame_count: usize,
        aspect: f32,
        max_bones: usize,
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
            unsafe {
                buffer.update()?;
            }
            uniform_buffers.push(buffer);
        }

        // Initialize joint matrices buffers
        let mut joint_matrices_buffer = Vec::with_capacity(frame_count);
        for _ in 0..frame_count {
            let buffer =
                unsafe { resources::JointMatricesBuffer::new(Arc::clone(alloc), max_bones)? };
            joint_matrices_buffer.push(buffer);
        }

        // Create default texture
        let default_texture_data = TextureData::solid_color([255, 255, 255, 255]);
        let default_texture = unsafe {
            Texture::from_data(
                Arc::clone(alloc),
                Arc::clone(&device.device),
                command_pool,
                device.graphics_queue,
                &default_texture_data,
                vk::Format::R8G8B8A8_SRGB,
                Some("default_texture"),
            )?
        };

        // Initialize material storage buffer (Bindless-ready)
        let max_materials = 1024;
        let mut material_storage_buffer = unsafe {
            StorageBuffer::<resources::uniform::MaterialUniform>::new(
                Arc::clone(alloc),
                Arc::clone(&device.device),
                max_materials,
                "material_storage_buffer",
            )?
        };

        // Populate with default material at index 0
        let default_mat = Material::default();
        let mut initial_materials =
            vec![resources::uniform::MaterialUniform::default(); max_materials];

        let mut first_mat = resources::uniform::MaterialUniform::default();
        first_mat.set_base_color_factor(glam::Vec4::from_array(default_mat.color));
        first_mat.set_emissive_factor(glam::Vec4::from_array(default_mat.emissive));
        first_mat.set_metallic_roughness(default_mat.metallic, default_mat.roughness);
        first_mat.set_occlusion_strength(default_mat.occlusion_strength);
        first_mat.set_normal_scale(default_mat.normal_scale);
        first_mat.set_alpha_cutoff(default_mat.alpha_cutoff);
        initial_materials[0] = first_mat;

        unsafe {
            material_storage_buffer.update(&initial_materials)?;
        }

        // Initialize instance buffers for GPU culling/instancing
        let mut instance_buffers = Vec::with_capacity(frame_count);
        for _ in 0..frame_count {
            let buffer = unsafe {
                resources::InstanceBuffer::new(
                    Arc::clone(alloc),
                    Arc::clone(&device.device),
                    crate::renderer::occlusion_culling::MAX_CULLABLE_OBJECTS,
                )?
            };
            instance_buffers.push(buffer);
        }

        Ok(RendererResources {
            uniform_buffers,
            joint_matrices_buffer,
            default_texture,
            material_storage_buffer,
            instance_buffers,
        })
    }

    fn init_ibl(
        device: &vulkan::VulkanDevice,
        alloc: &Arc<vulkan::Allocator>,
        command_pool: vk::CommandPool,
    ) -> Result<(
        crate::renderer::features::brdf_lut::BrdfLutPass,
        vk::Sampler,
        crate::renderer::features::ibl_manager::IblManager,
    )> {
        log::info!("Initializing BrdfLutPass...");
        let mut brdf_lut_pass = crate::renderer::features::brdf_lut::BrdfLutPass::new();

        log::info!("Baking BRDF LUT...");
        unsafe {
            brdf_lut_pass.bake(device, command_pool, Arc::clone(alloc))?;
        }
        log::info!("BRDF LUT Baked.");

        let ibl_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .compare_enable(false)
            .anisotropy_enable(false)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);

        let ibl_sampler = unsafe { device.device.create_sampler(&ibl_sampler_info, None)? };

        log::info!("Initializing IblManager...");
        let ibl_manager = crate::renderer::features::ibl_manager::IblManager::new(
            Arc::clone(&device.device),
            Arc::clone(alloc),
        );

        Ok((brdf_lut_pass, ibl_sampler, ibl_manager))
    }

    unsafe fn create_swapchain_data(
        device: &vulkan::VulkanDevice,
        alloc: &Arc<vulkan::Allocator>,
        resources: &Arc<ResourceRegistry>,
        extent: vk::Extent2D,
    ) -> Result<SwapchainData> {
        let mut swapchain = vulkan::SwapchainWrapper::new(device, device.headless, extent)?;
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
            Arc::clone(alloc),
            swapchain.extent.width,
            swapchain.extent.height,
        )?;
        let depth_buffer_id = depth_buffer
            .register_with_registry(resources)
            .map_err(|e| AshError::VulkanError(format!("Failed to register depth buffer: {e}")))?;

        let mut render_pass_builder = vulkan::RenderPass::builder(Arc::clone(&device.device));

        if device.headless {
            render_pass_builder = render_pass_builder
                .with_color_attachment(swapchain.format, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        } else {
            render_pass_builder = render_pass_builder.with_swapchain_color(swapchain.format);
        }

        let mut render_pass = render_pass_builder
            .with_depth_attachment(
                depth_buffer.format(),
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )
            .build()?;
        let render_pass_id = resources
            .register_render_pass(render_pass.handle())
            .map_err(|e| AshError::VulkanError(format!("Failed to register render pass: {e}")))?;
        render_pass.mark_managed_by_registry();

        Ok(SwapchainData {
            swapchain,
            swapchain_image_view_ids,
            depth_buffer,
            depth_buffer_id,
            render_pass,
            render_pass_id,
        })
    }

    unsafe fn create_frame_resources(
        device: &vulkan::VulkanDevice,
        resources: &Arc<ResourceRegistry>,
        swapchain_data: &SwapchainData,
    ) -> Result<FrameData> {
        let mut framebuffers = Vec::new();
        let mut framebuffer_ids = Vec::new();
        for (index, &image_view) in swapchain_data.swapchain.image_views.iter().enumerate() {
            let attachments = [image_view, swapchain_data.depth_buffer.view()];
            let framebuffer = vulkan::Framebuffer::new(
                Arc::clone(&device.device),
                swapchain_data.render_pass.handle(),
                &attachments,
                swapchain_data.swapchain.extent,
            )?;
            let framebuffer_id = resources
                .register_framebuffer(
                    framebuffer.handle(),
                    &[
                        swapchain_data.render_pass_id,
                        swapchain_data.depth_buffer_id,
                        swapchain_data.swapchain_image_view_ids[index],
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

        let command_buffers = command_manager.allocate_primary_buffers(framebuffers.len() as u32)?;

        let mut frame_syncs = Vec::with_capacity(framebuffers.len());
        let mut frame_sync_ids = Vec::with_capacity(framebuffers.len());
        for _ in 0..framebuffers.len() {
            let mut sync = vulkan::FrameSync::new(Arc::clone(&device.device))?;
            let image_available_id =
                resources
                    .register_semaphore(sync.image_available)
                    .map_err(|e| {
                        AshError::VulkanError(format!(
                            "Failed to register image-available semaphore: {e}"
                        ))
                    })?;
            let render_finished_id =
                resources
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
            .map_err(|e| AshError::VulkanError(format!("Failed to register command pool: {e}")))?;
        command_manager.mark_pool_managed_by_registry();

        Ok(FrameData {
            framebuffers,
            framebuffer_ids,
            command_manager,
            command_buffers,
            frame_syncs,
            frame_sync_ids,
            worker_count,
        })
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn create_main_pipeline(
        device: &vulkan::VulkanDevice,
        resources: &Arc<ResourceRegistry>,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        pipeline_cache: vk::PipelineCache,
        depth_format: vk::Format,
        pipeline_cfg: &PipelineConfig,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<(
        vulkan::PipelineLayout,
        ResourceId,
        vulkan::Pipeline,
        ResourceId,
    )> {
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
                offset: 128,
                size: material_push_size,
            },
        ];

        let mut pipeline_layout_builder =
            vulkan::PipelineLayout::builder(Arc::clone(&device.device));
        for layout in set_layouts {
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
            .with_render_pass(render_pass)
            .with_extent(extent)
            .with_pipeline_cache(pipeline_cache)
            .with_depth_format(depth_format)
            .with_cull_mode(vk::CullModeFlags::BACK)
            .with_multisampling(pipeline_cfg.multisample_config())
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

        for specialization in &pipeline_cfg.specialization_constants {
            pipeline_builder = pipeline_builder.with_specialization_bytes(
                specialization.stage,
                specialization.constant_id,
                specialization.bytes(),
            );
        }

        let mut pipeline = pipeline_builder.build()?;
        let pipeline_id = resources
            .register_pipeline(pipeline.pipeline, &[pipeline_layout_id])
            .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;
        pipeline.mark_managed_by_registry();

        Ok((pipeline_layout, pipeline_layout_id, pipeline, pipeline_id))
    }

    unsafe fn create_shadow_pipeline(
        device: &vulkan::VulkanDevice,
        pipeline_cache: vk::PipelineCache,
        shadow_map: &crate::renderer::shadow_map::ShadowMap,
        frame_layout: vk::DescriptorSetLayout,
        bindless_layout: vk::DescriptorSetLayout,
    ) -> Result<(vulkan::Pipeline, vulkan::PipelineLayout)> {
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

        let shadow_pipeline_layout = vulkan::PipelineLayout::builder(Arc::clone(&device.device))
            .add_push_constant(shadow_push_range)
            .add_push_constant(shadow_push_range_frag)
            .add_set_layout(frame_layout) // Set 0: Frame
            .add_set_layout(bindless_layout) // Set 1: Bindless (was 2)
            .build()?;

        let shadow_builder = vulkan::Pipeline::builder(Arc::clone(&device.device))
            .with_layout(shadow_pipeline_layout.handle())
            .with_render_pass(shadow_map.render_pass)
            .with_extent(vk::Extent2D {
                width: shadow_map.res,
                height: shadow_map.res,
            })
            .with_pipeline_cache(pipeline_cache)
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
        Ok((shadow_pipeline, shadow_pipeline_layout))
    }

    fn worker_index_for_frame(&self, frame_index: usize) -> usize {
        compute_worker_index(self.worker_count, frame_index)
    }

    fn render_post_processing(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<()> {
        if self.fullscreen_pass.is_none()
            || self.post_pipeline.is_none()
            || self.post_framebuffers.is_empty()
            || self.post_descriptor_sets.is_empty()
        {
            return Ok(());
        }

        let pass = self.fullscreen_pass.as_ref().ok_or_else(|| {
            AshError::RenderPassMissing("Fullscreen post-process pass".to_string())
        })?;
        let pipeline = self
            .post_pipeline
            .ok_or_else(|| AshError::PipelineMissing("Post-process pipeline".to_string()))?;
        let framebuffer = &self.post_framebuffers[image_index];
        let descriptor_set = self.post_descriptor_sets[image_index];
        let extent = self
            .swapchain
            .as_ref()
            .ok_or_else(|| AshError::SwapchainMissing("Required for post-processing".to_string()))?
            .extent;

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0], // Use black for post-processing clear to avoid flashes
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
                tonemapping_enabled: if self.tonemapping_enabled { 1.0 } else { 0.0 },
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
    pub fn set_mesh(&mut self, mut mesh: Mesh) -> Result<()> {
        unsafe {
            let upload_pool = self.cmds.upload_command_pool_handle();
            let key = mesh.name.clone();
            self.model_renderer
                .ensure_mesh(&key, &mesh, upload_pool, self.device.graphics_queue)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to upload mesh via ModelRenderer: {e}"))
                })?;

            mesh.ensure_texture(
                Arc::clone(&self.alloc),
                Arc::clone(&self.device.device),
                upload_pool,
                self.device.graphics_queue,
                &mut self.vram_budget,
                self.texture_compression,
            )
            .map_err(|e| AshError::VulkanError(format!("Failed to ensure mesh texture: {e}")))?;

            // Register textures with the bindless manager.
            if let Some(bindless_manager) = self.bindless_manager.as_mut() {
                if let Some(tex) = mesh.texture.as_ref() {
                    let idx = bindless_manager
                        .add_sampled_image(tex.view(), tex.sampler())
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to register base_color texture: {e}"
                            ))
                        })?;
                    mesh.texture_index = Some(idx);
                }
                if let Some(tex) = mesh.normal_texture.as_ref() {
                    let idx = bindless_manager
                        .add_sampled_image(tex.view(), tex.sampler())
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to register normal texture: {e}"))
                        })?;
                    mesh.normal_texture_index = Some(idx);
                }
                if let Some(tex) = mesh.metallic_roughness_texture.as_ref() {
                    let idx = bindless_manager
                        .add_sampled_image(tex.view(), tex.sampler())
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to register metallic_roughness texture: {e}"
                            ))
                        })?;
                    mesh.metallic_roughness_texture_index = Some(idx);
                }
                if let Some(tex) = mesh.occlusion_texture.as_ref() {
                    let idx = bindless_manager
                        .add_sampled_image(tex.view(), tex.sampler())
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to register occlusion texture: {e}"
                            ))
                        })?;
                    mesh.occlusion_texture_index = Some(idx);
                }
                if let Some(tex) = mesh.emissive_texture.as_ref() {
                    let idx = bindless_manager
                        .add_sampled_image(tex.view(), tex.sampler())
                        .map_err(|e| {
                            AshError::VulkanError(format!(
                                "Failed to register emissive texture: {e}"
                            ))
                        })?;
                    mesh.emissive_texture_index = Some(idx);
                }
            }

            let indices = [
                mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.metallic_roughness_texture_index
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
            ];
            let emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

            let flags = TexturePresenceFlags::from_mesh(&mesh);

            let material_handle = self.material_manager.register_material(self.material.clone());

            let mesh_data = MeshData {
                name: Arc::clone(&key),
                texture_indices: indices,
                emissive_index,
                texture_flags: flags,
                material_handle,
            };

            self.draw_items.clear();
            self.draw_items.push(DrawItem {
                key: Arc::clone(&key),
                transform: self.transform.model_matrix(),
                material: self.material.clone(),
                material_handle,
                texture_flags: flags,
                texture_indices: indices,
                emissive_index,
                is_skinned: false,
                joint_offset: 0,
                alpha_cutoff: self.material.alpha_cutoff,
                cast_shadows: true,
            });

            if self.mesh_data.is_empty() {
                self.mesh_data.push(mesh_data);
            } else {
                self.mesh_data[0] = mesh_data;
            }

            self.mesh = Some(mesh);
        }

        Ok(())
    }

    /// Set the rendering mode (GPU-driven, Legacy, or Hybrid).
    pub fn set_rendering_mode(&mut self, mode: RenderingMode) {
        self.pass_manager.set_mode(mode);
    }

    /// Returns the current rendering mode.
    pub fn rendering_mode(&self) -> RenderingMode {
        self.pass_manager.mode()
    }

    /// Access the underlying memory allocator.
    pub fn allocator(&self) -> &Allocator {
        &self.alloc
    }

    /// Access the bindless manager (mutable) if enabled.
    pub fn bindless_manager_mut(&mut self) -> &mut Option<BindlessManager> {
        &mut self.bindless_manager
    }

    /// Set global lighting parameters.
    pub fn set_lighting(&mut self, direction: Vec3, color: [f32; 4], ambient_strength: f32) {
        self.light_direction = direction;
        self.light_color = color;
        self.ambient_color = [ambient_strength, ambient_strength, ambient_strength, 1.0];

        // Synchronize with shadow feature
        self.shadow_feature.set_light_direction(direction);
    }

    pub fn ambient_color_mut(&mut self) -> &mut [f32; 4] {
        &mut self.ambient_color
    }

    /// Get a mesh handle by name.
    pub fn get_mesh_handle(&self, name: &str) -> Option<u32> {
        self.mesh_data
            .iter()
            .enumerate()
            .find(|(_, data)| &*data.name == name)
            .map(|(i, _)| i as u32)
    }

    /// Get immutable access to consolidated mesh data.
    pub fn mesh_data(&self) -> &[MeshData] {
        &self.mesh_data
    }

    /// Get mutable access to mesh data by handle.
    pub fn get_mesh_data_mut(&mut self, handle: u32) -> Option<&mut MeshData> {
        self.mesh_data.get_mut(handle as usize)
    }

    /// Get mutable access to the material manager.
    pub fn material_manager_mut(&mut self) -> &mut MaterialManager {
        &mut self.material_manager
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
                &mut self.vram_budget,
                self.texture_compression,
            )?;

            self.model_renderer
                .ensure_mesh(&key, mesh, upload_pool, self.device.graphics_queue)?;

            // Register textures with bindless manager
            if let Some(bindless_manager) = self.bindless_manager.as_mut() {
                register_mesh_textures(mesh, bindless_manager)?;
            }

            // Register material from mesh properties
            let mut material_handle = self.material_manager.default_material();
            if let Some(props) = &mesh.material_properties {
                if !self.allow_auto_material {
                    log::warn!(
                        "Mesh '{}' has material properties but auto-material creation is disabled.",
                        &*mesh.name
                    );
                } else {
                    let material = Material {
                        name: format!("{}_material_{}", &*mesh.name, handle),
                        color: props.base_color_factor,
                        metallic: props.metallic_factor,
                        roughness: props.roughness_factor,
                        emissive: props.emissive_factor,
                        occlusion_strength: props.occlusion_strength,
                        normal_scale: props.normal_scale,
                        alpha_cutoff: props.alpha_cutoff,
                        tint_index: -1,
                    };
                    
                    material_handle = self.material_manager.register_material(material);
                    
                    log::debug!(
                        "Registered auto-material for mesh '{}': handle={:?}, metallic={:.2}, roughness={:.2}",
                        &*mesh.name, material_handle, props.metallic_factor, props.roughness_factor
                    );
                }
            }

            let flags = TexturePresenceFlags::from_mesh(mesh);

            let indices = [
                mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                mesh.metallic_roughness_texture_index
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
            ];
            let emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

            let mesh_data = MeshData {
                name: Arc::clone(&key),
                texture_indices: indices,
                emissive_index,
                texture_flags: flags,
                material_handle,
            };

            if handle as usize >= self.mesh_data.len() {
                self.mesh_data
                    .resize(handle as usize + 1, MeshData::default());
            }
            self.mesh_data[handle as usize] = mesh_data;
        }

        Ok(())
    }

    pub fn get_stats(&self) -> crate::renderer::diagnostics::RendererStats {
        crate::renderer::diagnostics::RendererStats {
            vram_usage: self.vram_budget.get_stats(),
            draw_calls_per_frame: self.diagnostics.frame_stats.draw_calls,
            triangles_rendered: self.diagnostics.frame_stats.triangles,
        }
    }

    /// Logs the current frame statistics to the debug log.
    pub fn log_frame_stats(&self) {
        self.get_stats().log_frame_stats();
    }

    /// Updates the GPU material buffer with a material at the specified index
    /// This must be called after registering the material to ensure the GPU sees the correct material
    pub fn upload_material_to_gpu(&mut self, handle: u32, material: &Material) -> Result<()> {
        if let Some(buffer) = self.material_storage_buffer.as_mut() {
            let capacity = buffer.capacity();
            
            // Validate handle within capacity
            if (handle as usize) >= capacity {
                return Err(AshError::VulkanError(format!(
                    "Material handle {handle} exceeds buffer capacity {capacity}"
                )));
            }
            
            // Create MaterialUniform from the material
            let mut mat_uniform = resources::uniform::MaterialUniform::default();
            mat_uniform.set_base_color_factor(glam::Vec4::from_array(material.color));
            mat_uniform.set_emissive_factor(glam::Vec4::from_array(material.emissive));
            mat_uniform.set_metallic_roughness(material.metallic, material.roughness);
            mat_uniform.set_occlusion_strength(material.occlusion_strength);
            mat_uniform.set_normal_scale(material.normal_scale);
            mat_uniform.set_alpha_cutoff(material.alpha_cutoff);

            // Read current buffer contents, update one element, and reupload
            // This preserves all other materials in the buffer
            let mut all_materials = vec![resources::uniform::MaterialUniform::default(); capacity];
            
            // TEMP: Map and read existing data (if needed for proper preservation)
            // For now, we assume buffer is initialized with defaults at index 0
            // and we're updating subsequent indices
            
            // Set the default at index 0 (ensure it's preserved)
            let default_mat = Material::default();
            let mut default_uniform = resources::uniform::MaterialUniform::default();
            default_uniform.set_base_color_factor(glam::Vec4::from_array(default_mat.color));
            default_uniform.set_emissive_factor(glam::Vec4::from_array(default_mat.emissive));
            default_uniform.set_metallic_roughness(default_mat.metallic, default_mat.roughness);
            default_uniform.set_occlusion_strength(default_mat.occlusion_strength);
            default_uniform.set_normal_scale(default_mat.normal_scale);
            default_uniform.set_alpha_cutoff(default_mat.alpha_cutoff);
            all_materials[0] = default_uniform;
            
            // Copy the new material at the specified index
            all_materials[handle as usize] = mat_uniform;

            // Update the buffer on GPU
            unsafe {
                buffer.update(&all_materials)?;
            }
            
            log::info!("Uploaded material to GPU at index {handle}");
            Ok(())
        } else {
            Err(AshError::VulkanError("Material storage buffer not initialized".to_string()))
        }
    }

    /// Get access to the material manager (for testing)
    pub fn material_manager(&self) -> &MaterialManager {
        &self.material_manager
    }

    /// Registers mesh data described by a [`MeshDescriptor`] with the renderer and returns the
    /// internal key used for lookup.
    pub fn register_mesh_descriptor(
        &mut self,
        handle: u32,
        descriptor: &MeshDescriptor,
    ) -> Result<String> {
        let mut mesh = Mesh::from_descriptor(descriptor);
        let key = Arc::clone(&mesh.name);

        self.register_mesh_handle(handle, &mut mesh)?;

        Ok(key.to_string())
    }

    /// Converts a material descriptor into a renderer material and registers it.
    pub fn register_material_descriptor(
        &mut self,
        _handle: u32,
        descriptor: &MaterialDescriptor,
    ) -> MaterialHandle {
        let material = descriptor.material.clone();
        self.material_manager.register_material(material)
    }

    /// Registers a generic storage buffer with the bindless manager.
    ///
    /// This is the "pro" way to handle bindless storage buffers, providing type safety
    /// and automatic memory management.
    pub fn register_bindless_storage_buffer<T: Copy>(
        &mut self,
        data: &[T],
        name: &str,
    ) -> Result<(
        Arc<parking_lot::Mutex<resources::uniform::StorageBuffer<T>>>,
        u32,
    )> {
        let bindless_manager = self
            .bindless_manager
            .as_mut()
            .ok_or_else(|| AshError::VulkanError("Bindless manager not enabled".to_string()))?;

        unsafe {
            let mut buffer = resources::uniform::StorageBuffer::new(
                Arc::clone(&self.alloc),
                Arc::clone(&self.device.device),
                data.len(),
                name,
            )?;

            // Initial upload
            buffer.update(data)?;

            // Register with bindless manager (Binding 2)
            let index = bindless_manager.add_storage_buffer(
                buffer.buffer,
                0,
                std::mem::size_of_val(data) as vk::DeviceSize,
            )?;

            Ok((Arc::new(parking_lot::Mutex::new(buffer)), index))
        }
    }

    /// Update the joint matrices buffer for GPU skinning
    ///
    /// # Safety
    /// Caller must ensure matrices slice does not exceed max_bones capacity
    pub unsafe fn update_joint_ssbo(&mut self, matrices: &[Mat4]) -> Result<()> {
        self.update_joint_ssbo_offset(matrices, 0)
    }

    /// Update the joint matrices buffer for GPU skinning at a specific offset
    ///
    /// # Safety
    /// Caller must ensure matrices slice and offset do not exceed max_bones capacity
    pub unsafe fn update_joint_ssbo_offset(
        &mut self,
        matrices: &[Mat4],
        offset: usize,
    ) -> Result<()> {
        if offset + matrices.len() > self.max_bones {
            return Err(AshError::VulkanError(format!(
                "Joint matrices update (offset: {}, count: {}) exceeds max_bones ({})",
                offset,
                matrices.len(),
                self.max_bones
            )));
        }

        if let Some(buffer) = self.joint_matrices_buffer.get_mut(self.current_frame) {
            buffer.update_offset(matrices, offset)?;
        }

        Ok(())
    }

    pub fn max_bones(&self) -> usize {
        self.max_bones
    }

    /// Queues a skinned mesh for rendering.
    pub fn draw_skinned_mesh(
        &mut self,
        mesh_handle: u32,
        material_handle: MaterialHandle,
        transform: Mat4,
        joint_offset: u32,
    ) {
        let _ = self.submit_render_commands(&[RenderCommand {
            mesh_handle,
            material_handle,
            transform,
            is_skinned: true,
            joint_offset,
            cast_shadows: true,
        }]);
    }

    /// Submit render commands for the current frame.
    ///
    /// Each `RenderCommand` specifies a mesh handle, material handle, and transform.
    /// For large command counts (>1000), uses parallel processing across all CPU cores.
    pub fn submit_render_commands(&mut self, commands: &[RenderCommand]) -> Result<()> {
        self.draw_items.clear();
        self.instancing_manager.begin_frame();

        const PARALLEL_THRESHOLD: usize = 1000;

        if commands.len() > PARALLEL_THRESHOLD {
            // Parallel extraction for large command counts
            use std::collections::HashMap;

            // Capture only thread-safe fields
            let mesh_data = &self.mesh_data;
            let material_manager = &self.material_manager;
            let strict_mode = self.strict_mode;

            let (draw_items, instance_batches) = commands
                .par_iter()
                .fold(
                    || (Vec::new(), HashMap::<BatchKey, Vec<InstanceData>>::new()),
                    |(mut items, mut batches), command| {
                        if let Some(mesh_data_entry) = mesh_data.get(command.mesh_handle as usize) {
                            let mesh_key = &mesh_data_entry.name;

                            let material_handle = if command.material_handle.is_null() {
                                mesh_data_entry.material_handle
                            } else {
                                command.material_handle
                            };
                            
                            // Unreal-Style validation: get material or fallback to default
                            let material = material_manager.get_material(material_handle);
                            
                            // Safety check: log if version mismatch (rare but possible)
                            if !material_manager.is_handle_valid(material_handle) {
                                let msg = format!("Invalid material handle {material_handle:?} detected for mesh handle {}, using default", command.mesh_handle);
                                if strict_mode {
                                    log::error!("{msg}");
                                } else {
                                    log::warn!("{msg}");
                                }
                            }

                            let texture_flags = mesh_data_entry.texture_flags;
                            let (indices, emissive_index) = (mesh_data_entry.texture_indices, mesh_data_entry.emissive_index);

                            if command.is_skinned {
                                items.push(DrawItem {
                                    key: mesh_key.clone(),
                                    transform: command.transform,
                                    material: material.clone(),
                                    material_handle,
                                    texture_flags,
                                    texture_indices: indices,
                                    emissive_index,
                                    is_skinned: true,
                                    joint_offset: command.joint_offset,
                                    alpha_cutoff: material.alpha_cutoff,
                                    cast_shadows: command.cast_shadows,
                                });
                            } else {
                                let key = BatchKey::new(command.mesh_handle, material_handle);
                                let mut instance = InstanceData::from_matrix(command.transform);
                                if command.cast_shadows {
                                    instance.set_flag(crate::renderer::occlusion_culling::CULL_FLAG_CAST_SHADOWS, true);
                                }
                                batches
                                    .entry(key)
                                    .or_default()
                                    .push(instance);
                            }
                        } else if strict_mode {
                            log::error!("Mesh handle {} not found in registry", command.mesh_handle);
                        }
                        (items, batches)
                    },
                )
                .reduce(
                    || (Vec::new(), HashMap::<BatchKey, Vec<InstanceData>>::new()),
                    |(mut a_items, mut a_batches), (b_items, b_batches)| {
                        a_items.extend(b_items);
                        for (key, instances) in b_batches {
                            a_batches
                                .entry(key)
                                .or_default()
                                .extend(instances);
                        }
                        (a_items, a_batches)
                    },
                );

            // Merge results
            self.draw_items = draw_items;
            for (key, instances) in instance_batches {
                self.instancing_manager.add_instances(key, instances);
            }
        } else {
            // Sequential processing for small command counts (avoids rayon overhead)
            for command in commands {
                if let Some(mesh_data) = self.mesh_data.get(command.mesh_handle as usize) {
                    let mesh_key = &mesh_data.name;

                    let material_handle = if command.material_handle.is_null() {
                        mesh_data.material_handle
                    } else {
                        command.material_handle
                    };

                    // Unreal-Style validation: get material or fallback to default
                    let material = self.material_manager.get_material(material_handle);
                    
                    // Safety check: log if version mismatch (rare but possible)
                    if !self.material_manager.is_handle_valid(material_handle) {
                        let msg = format!("Invalid material handle {material_handle:?} detected for mesh handle {}, using default", command.mesh_handle);
                        if self.strict_mode {
                            log::error!("{msg}");
                        } else {
                            log::warn!("{msg}");
                        }
                    }

                    let texture_flags = mesh_data.texture_flags;
                    let (indices, emissive_index) =
                        (mesh_data.texture_indices, mesh_data.emissive_index);

                    if command.is_skinned {
                        self.draw_items.push(DrawItem {
                            key: mesh_key.clone(),
                            transform: command.transform,
                            material: material.clone(),
                            material_handle,
                            texture_flags,
                            texture_indices: indices,
                            emissive_index,
                            is_skinned: true,
                            joint_offset: command.joint_offset,
                            alpha_cutoff: material.alpha_cutoff,
                            cast_shadows: command.cast_shadows,
                        });
                    } else {
                        let key = BatchKey::new(command.mesh_handle, material_handle);
                        let instance = InstanceData::from_matrix(command.transform)
                            .with_cast_shadows(command.cast_shadows);
                        self.instancing_manager.add_instance(key, instance);
                    }
                } else {
                    let msg = format!("Mesh handle {} not found in registry", command.mesh_handle);
                    if self.strict_mode {
                        log::error!("{msg}");
                        return Err(AshError::MeshNotFound(command.mesh_handle));
                    } else {
                        log::warn!("{msg}");
                    }
                }
            }
        }

        // Fallback: if no commands were submitted, use default cube
        if self.draw_items.is_empty() && self.instancing_manager.stats().total_instances == 0 {
            if let Some(_mesh) = self.mesh.as_ref() {
                let key = BatchKey::new(0, self.material_manager.default_material());
                let instance = InstanceData::from_matrix(self.transform.model_matrix())
                    .with_cast_shadows(true);
                self.instancing_manager.add_instance(key, instance);
            }
        }

        self.instancing_manager.finalize();

        // Sort draw items to minimize pipeline and material changes
        self.draw_items.sort_by(|a, b| {
            a.is_skinned
                .cmp(&b.is_skinned)
                .then_with(|| a.material.name.cmp(&b.material.name))
                .then_with(|| a.key.cmp(&b.key))
        });

        Ok(())
    }

    /// Bake IBL maps from an equirectangular texture.
    ///
    /// This function converts an equirectangular environment map to cubemap format
    /// and generates irradiance and prefiltered maps for image-based lighting.
    /// Currently unused but preserved for runtime environment map loading features.
    pub fn bake_ibl_from_equirect(&mut self, equirect: &resources::Texture) -> Result<()> {
        log::info!("Baking IBL maps from equirectangular texture...");
        let command_pool = self.cmds.upload_command_pool_handle();

        // 1. Convert Equirect to Cubemap
        let env_cubemap = self.ibl_manager.create_cubemap_from_equirect(
            &self.device,
            command_pool,
            equirect.view(),
            equirect.sampler(),
            512, // Environment resolution
        )?;

        // 2. Generate Irradiance Map
        let irradiance_map = self.ibl_manager.generate_irradiance(
            &self.device,
            command_pool,
            env_cubemap.view(),
            self.ibl_sampler,
        )?;

        // 3. Generate Prefiltered Map
        let prefiltered_map = self.ibl_manager.generate_prefiltered(
            &self.device,
            command_pool,
            env_cubemap.view(),
            self.ibl_sampler,
        )?;

        self.irradiance_map = Some(irradiance_map);
        self.prefiltered_map = Some(prefiltered_map);

        log::info!("IBL maps baked successfully.");
        Ok(())
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
                let extent = self.pending_extent.unwrap_or(vk::Extent2D {
                    width: 1280,
                    height: 720,
                });
                self.swapchain = Some(vulkan::SwapchainWrapper::new(
                    &self.device,
                    self.device.headless,
                    extent,
                )?);
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

        self.recreate_joint_buffers(image_count)?;
        self.recreate_frame_syncs(image_count)?;
        self.recreate_command_buffers()?;
        self.recreate_uniform_buffers(image_count)?;
        self.recreate_vsr_pass(
            self.swapchain
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("Swapchain missing".to_string()))?
                .extent,
        )?;
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
        let layout = self
            .pipeline_layout
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Pipeline layout missing".to_string()))?
            .handle();
        let render_pass = self
            .render_pass
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Render pass missing".to_string()))?
            .handle();
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

        // Build Skinned Pipeline
        log::info!("Compiling skinned pipeline...");
        let mut skinned_builder = vulkan::Pipeline::builder(Arc::clone(&self.device.device))
            .with_layout(layout)
            .with_render_pass(render_pass)
            .with_extent(extent)
            .with_pipeline_cache(cache)
            .with_depth_format(depth_format)
            .with_cull_mode(vk::CullModeFlags::BACK)
            .with_multisampling(multisample_config)
            .with_vertex_input(
                vec![SkinnedVertex::binding_description()],
                SkinnedVertex::attribute_descriptions().to_vec(),
            );

        if self.gbuffer.is_some() {
            let blend_attachments = vec![
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
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
            ];
            skinned_builder = skinned_builder.with_color_blend_attachments(blend_attachments);
        }

        skinned_builder = skinned_builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/skinning.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;
        skinned_builder = skinned_builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

        let mut new_skinned_pipeline = skinned_builder.build()?;
        let skinned_pipeline_id = self
            .resources
            .register_pipeline(
                new_skinned_pipeline.pipeline,
                &[pipeline_layout_id, render_pass_id],
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to register skinned pipeline: {e}"))
            })?;

        new_skinned_pipeline.mark_managed_by_registry();
        self.skinned_pipeline = Some(new_skinned_pipeline);
        self.skinned_pipeline_id = Some(skinned_pipeline_id);

        log::info!("Pipelines recompiled successfully!");
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
            let hdr = self
                .hdr_framebuffer
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("HDR framebuffer missing".to_string()))?;
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
                self.hdr_framebuffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("HDR framebuffer missing".to_string()))?
                    .view()
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
            let count = self.frame_syncs.len() as u32;
            manager.recreate_frame_sets(count)?;
            manager.recreate_environment_sets(count)?;
            // Joint matrices are now in bindless Set 1 Binding 3, no need to recreate separate sets

            let buffer_size =
                std::mem::size_of::<crate::renderer::resources::uniform::MvpMatrices>()
                    as vk::DeviceSize;
            for index in 0..manager.frame_set_count() {
                if let Some(ubo) = self.uniform_buffers.get(index) {
                    manager.bind_frame_uniform(index, ubo.buffer, buffer_size)?;
                }
            }

            // Joint matrices buffers are managed via bindless manager, not descriptor manager
        }

        Ok(())
    }

    fn recreate_joint_buffers(&mut self, count: usize) -> Result<()> {
        self.joint_matrices_buffer.clear();
        for _ in 0..count {
            let buffer = unsafe {
                resources::JointMatricesBuffer::new(Arc::clone(&self.alloc), self.max_bones)?
            };
            self.joint_matrices_buffer.push(buffer);
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

        // Automatic material synchronization: Ensure DrawItems reflect current material state
        // This prevents stale tint_index and other material properties from causing rendering issues
        self.refresh_draw_items();

        log::debug!(
            "Frame {}: Material synchronization complete",
            self.current_frame
        );

        self.resize_if_needed()?;
        if self.resize_pending {
            log::debug!(
                "Frame {}: Resize pending, skipping render",
                self.current_frame
            );
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
            assert!(
                frame_index < self.joint_matrices_buffer.len(),
                "Frame index {} out of bounds for joint buffers ({})",
                frame_index,
                self.joint_matrices_buffer.len()
            );
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
                    // We assume it's a sphere for now until we have better bounds
                    let bounds = CullBoundingBox::new(Vec3::ZERO, Vec3::ONE * 100.0);
                    self.occlusion_culling.push_clusters(
                        bounds,
                        item.transform,
                        i as u32,
                        uploaded.clusters(),
                    );
                }
            }

            // Phase 10: Execute Hi-Z construction
            if let (Some(ref mut hiz), Some(depth_buffer)) =
                (&mut self.hiz_pass, &self.depth_buffer)
            {
                // Hiz pyramid is built from previous frame's depth
                hiz.build_pyramid(command_buffer, depth_buffer.image())?;
            }

            // Apply sub-pixel jitter for VSR/TSR if enabled
            let mut jittered_projection = projection;
            let mut jitter_uv = [0.0f32; 2];
            if let Some(ref mut vsr) = self.vsr_pass {
                let (jx, jy) = vsr.next_jitter();
                // Jitter is in pixels [-0.5, 0.5], convert to NDC/UV
                jitter_uv = [jx, jy];
                let extent = self
                    .swapchain
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Swapchain missing".to_string()))?
                    .extent;
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
                self.shadow_feature.before_frame(&mut feature_ctx);

                // Matrices provided via function arguments.
                let matrices = uniform_buffer.matrices_mut();
                matrices.model = self.transform.model_matrix();
                matrices.view = view;
                matrices.projection = jittered_projection;
                matrices.view_proj = jittered_projection * view;
                matrices.camera_pos = camera_pos.extend(1.0);
                matrices.set_lighting(
                    self.light_direction,
                    Vec4::from_array(self.light_color).truncate(),
                    Vec4::from_array(self.ambient_color).truncate(),
                );

                // Set light-space matrix for shadow mapping
                let light_space_matrix = self.shadow_feature.light_space_matrix();
                matrices.set_light_space_matrix(light_space_matrix);
                matrices.normal_matrix = matrices.model.inverse().transpose();

                uniform_buffer.update()?;
            }

            // CRITICAL: Ensure all host-written buffers (including bindless storage buffers) are visible to GPU
            // This is required because examples might update buffers directly on the host.
            let global_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::UNIFORM_READ);

            self.device.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[global_barrier],
                &[],
                &[],
            );

            // Update post-processing descriptors once per frame to ensure they point to the correct VSR output.
            self.update_post_descriptors()?;

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
                Ok(index) => {
                    log::debug!(
                        "Frame {}: Successfully acquired image index {}",
                        self.current_frame,
                        index
                    );
                    index
                }
                Err(AshError::SwapchainOutOfDate(_)) => {
                    log::warn!(
                        "Frame {}: Swapchain out of date, requesting resize",
                        self.current_frame
                    );
                    self.request_swapchain_resize(swapchain_extent);
                    return Ok(());
                }
                Err(err) => {
                    log::error!(
                        "Frame {}: Failed to acquire next image: {}",
                        self.current_frame,
                        err
                    );
                    return Err(err);
                }
            };

            let worker_index = self.worker_index_for_frame(frame_index);
            debug_assert!(
                worker_index < self.worker_count.max(1),
                "worker index {} out of bounds for {} workers",
                worker_index,
                self.worker_count
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

                    // Draw all shadow casters
                    // 1. Draw skinned meshes (draw_items)
                    for item in &self.draw_items {
                        if !item.cast_shadows {
                            continue;
                        }

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

                    // 2. Draw instanced meshes (batches) that cast shadows
                    for batch in self.instancing_manager.batches() {
                        let mesh_data = if let Some(m) = self.mesh_data.get(batch.key.mesh_id as usize) {
                            m
                        } else {
                            continue;
                        };
                        
                        if let Some(uploaded) = self.model_renderer.get(&mesh_data.name) {
                            // Bind vertex buffers once per batch
                            let offsets = [0];
                            self.device.device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[uploaded.vertex_buffer()],
                                &offsets,
                            );

                            // Bind index buffer once per batch if available
                            if let Some(index_buffer) = uploaded.index_buffer() {
                                self.device.device.cmd_bind_index_buffer(
                                    command_buffer,
                                    index_buffer,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                            }

                            // Bind Bindless Textures (Set 2) once per batch
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

                            for instance in &batch.instances {
                                // Check if this instance casts shadows using the flag we added
                                if !instance.has_flag(crate::renderer::occlusion_culling::CULL_FLAG_CAST_SHADOWS) {
                                    continue;
                                }

                                // Construct model matrix from instance data
                                let model_matrix = instance.model_matrix();
                                
                                let light_space_push =
                                    crate::renderer::model_renderer::Mat4Push::from(light_space_matrix);
                                let model_push =
                                    crate::renderer::model_renderer::Mat4Push::from(model_matrix);

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

                                // Push texture index for alpha discard (fetch from mesh_data)
                                let base_color_index = mesh_data.texture_indices[0] as u32;
                                
                                self.device.device.cmd_push_constants(
                                    command_buffer,
                                    shadow_layout.handle(),
                                    vk::ShaderStageFlags::FRAGMENT,
                                    128,
                                    bytemuck::bytes_of(&base_color_index),
                                );

                                if uploaded.index_buffer().is_some() {
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
                    }

                    cmd_ctx.end_render_pass();
                }
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0], // Background is always black for better contrast
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let framebuffer = self.framebuffers.get(image_index as usize).ok_or_else(|| {
                log::error!(
                    "Frame {}: Framebuffer index {} out of range (max: {})",
                    self.current_frame,
                    image_index,
                    self.framebuffers.len()
                );
                AshError::VulkanError("Framebuffer index out of range".into())
            })?;

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
                    cmd_ctx.bind_descriptor_sets(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout_handle,
                        0, // Set 0: Frame Data
                        &[frame_set],
                        &[],
                    );

                    // Bind global bindless descriptor set (Set 1)
                    if let Some(ref bindless) = self.bindless_manager {
                        cmd_ctx.bind_descriptor_sets(
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout_handle,
                            1, // Set 1: Bindless (Textures, Materials, Instances, Joints)
                            &[bindless.descriptor_set()],
                            &[],
                        );
                    }

                    // Bind Global Environment descriptor (Set 2: ShadowMap + IBL)
                    if let Some(env_set) = manager.environment_set(frame_index) {
                        if let Some(shadow_map) = self.shadow_feature.shadow_map() {
                            manager.bind_shadow_map(
                                frame_index,
                                shadow_map.depth_image_view,
                                shadow_map.sampler,
                            )?;
                        }

                        if let (Some(irradiance), Some(prefilter), Some(brdf_view)) = (
                            &self.irradiance_map,
                            &self.prefiltered_map,
                            self.brdf_lut_pass.as_ref().and_then(|p| p.get_lut_view()),
                        ) {
                            let ibl_resources = vulkan::IBLResources {
                                irradiance_view: irradiance.view(),
                                prefiltered_view: prefilter.view(),
                                brdf_lut_view: brdf_view,
                                sampler: self.ibl_sampler,
                            };
                            manager.bind_ibl_resources(frame_index, &ibl_resources)?;
                        }

                        cmd_ctx.bind_descriptor_sets(
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout_handle,
                            2, // Set 2: Environment
                            &[env_set],
                            &[],
                        );
                    }

                    // Bind Forward+ descriptor set (Set 3)
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

            // --- Phase 10: GPU Instancing & Culling Integration ---

            // 1. Prepare and upload all instances to the InstanceBuffer
            let mut all_instances = Vec::new();
            let mut batch_offsets = Vec::new();
            {
                for batch in self.instancing_manager.batches() {
                    batch_offsets.push(all_instances.len() as u32);
                    all_instances.extend_from_slice(&batch.instances);
                }
            }

            if !all_instances.is_empty() {
                self.instance_buffer[frame_index].update(&all_instances)?;
            }

            // 3. Render all instanced batches (static meshes)
            let mut current_object_offset = 0;
            for (i, batch) in self.instancing_manager.batches().enumerate() {
                if let Some(mesh_data) = self.mesh_data.get(batch.key.mesh_id as usize) {
                    let mesh_key = &mesh_data.name;

                    let _material = self.material_manager.get_material(batch.key.material_id);
                    
                    if !self.material_manager.is_handle_valid(batch.key.material_id) {
                        let msg = format!(
                            "Invalid material handle {:?} detected for mesh '{}', using default",
                            batch.key.material_id, mesh_key
                        );
                        if self.strict_mode {
                            log::error!("{msg}");
                        } else {
                            log::warn!("{msg}");
                        }
                    }

                    if let Some(uploaded) = self.model_renderer.get(mesh_key) {
                        // Push constants - use material handle from batch
                        let material_push = MaterialPushConstants::new(batch.key.material_id);

                        // --- GPU-Driven Path ---
                        if self.pass_manager.use_gpu_driven() && self.indirect_draw_pass.is_some() {
                            let indirect = self.indirect_draw_pass.as_mut().unwrap();
                            // 1. Upload instances for this batch to the object buffer
                            indirect.upload_objects(
                                &self.alloc.vma,
                                &batch.instances,
                                current_object_offset,
                            )?;

                            // 2. Upload draw template for this mesh
                            let template = vk::DrawIndexedIndirectCommand {
                                index_count: uploaded.index_count(),
                                instance_count: 1,
                                first_index: 0,
                                vertex_offset: 0,
                                first_instance: 0, // Set by compute shader
                            };
                            indirect.upload_templates(&self.alloc.vma, &[template], 0)?;

                            // 4. Run Culling Compute Shader
                            let bindless_manager =
                                self.bindless_manager.as_ref().expect("Bindless manager");
                            indirect.execute_culling(
                                command_buffer,
                                &self.occlusion_culling,
                                bindless_manager,
                                projection * view,
                                swapchain_extent.width,
                                swapchain_extent.height,
                                current_object_offset as u32,
                                batch.count() as u32,
                                0, // indirect_offset
                            )?;

                            // 5. Draw Indirect
                            let material_push = material_push.with_debug_path(1); // 1: GPU-Driven
                            let ctx = DrawContext {
                                command_buffer,
                                pipeline_layout: pipeline_layout_handle,
                                uploaded,
                                material: &material_push,
                                instance_buffer_index: indirect.object_buffer_index().unwrap_or(0),
                                joint_buffer_index: self.joint_buffer_indices[frame_index],
                            };
                            self.model_renderer.draw_mesh_indirect_count(
                                &ctx,
                                &crate::renderer::model_renderer::IndirectDrawCountParams {
                                    indirect_buffer: indirect.indirect_buffer(),
                                    indirect_offset: 0,
                                    count_buffer: indirect.count_buffer(),
                                    count_offset: 0,
                                    max_draw_count: batch.count() as u32,
                                    stride: std::mem::size_of::<vk::DrawIndexedIndirectCommand>()
                                        as u32,
                                },
                            );

                            current_object_offset += batch.count();
                        } else if self.pass_manager.use_legacy() || (self.pass_manager.use_gpu_driven() && self.indirect_draw_pass.is_none()) {
                            // --- Direct Path Fallback ---
                            let material_push = material_push.with_debug_path(2); // 2: Legacy
                            let ctx = DrawContext {
                                command_buffer,
                                pipeline_layout: pipeline_layout_handle,
                                uploaded,
                                material: &material_push,
                                instance_buffer_index: self.instance_buffer_indices[frame_index],
                                joint_buffer_index: self.joint_buffer_indices[frame_index],
                            };
                            self.model_renderer.draw_mesh_instanced(
                                &ctx,
                                batch.count() as u32,
                                batch_offsets[i],
                            );
                        }
                    }
                }
            }

            // 4. Draw remaining meshes (draw_items path) - FIXED to use correct material index
            // This path is used for simple meshes that don't use the GPU-driven batching system
            // Skinned meshes always use this path for now as they aren't handled by GPU-driven instancing.
            {
                let mut current_pipeline = scene_pipeline;
                for item in &self.draw_items {
                    // TODO: Implement frustum culling for draw_items here
                    
                    if let Some(uploaded) = self.model_renderer.get(&item.key) {
                        log::info!(
                            "Rendering mesh: key='{}', vertices={}, tint_index={}",
                            item.key,
                            uploaded.vertex_count(),
                            item.material.tint_index
                        );
                        // Switch pipeline if needed
                        let target_pipeline = if item.is_skinned {
                            self.skinned_pipeline
                                .as_ref()
                                .map(|p| p.pipeline)
                                .unwrap_or(scene_pipeline)
                        } else {
                            scene_pipeline
                        };

                        if target_pipeline != current_pipeline {
                            cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, target_pipeline);
                            current_pipeline = target_pipeline;
                        }
                        let model_matrix = item.transform;
                        
                        // FIXED: Use material_handle from the item itself
                        let material_handle = item.material_handle;
                        let material_push = MaterialPushConstants::new(material_handle)
                            .with_debug_path(2); // 2: Legacy (Direct path)

                        let ctx = DrawContext {
                            command_buffer,
                            pipeline_layout: pipeline_layout_handle,
                            uploaded,
                            material: &material_push,
                            instance_buffer_index: self.instance_buffer_indices[frame_index],
                            joint_buffer_index: self.joint_buffer_indices[frame_index],
                        };

                        self.model_renderer
                            .draw_mesh(&ctx, model_matrix, item.joint_offset);
                    } else {
                        log::error!("CRITICAL: Mesh not found in ModelRenderer cache! key='{}'. Mesh will not render.", item.key);
                    }
                }
            }

            cmd_ctx.end_render_pass();

            // --- SSGI Pass ---
            if let (Some(ref mut gbuffer), Some(ref mut ssgi)) =
                (&mut self.gbuffer, &mut self.ssgi_pass)
            {
                let depth_buffer = self
                    .depth_buffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Depth buffer missing".to_string()))?;
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
                let depth_buffer = self
                    .depth_buffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Depth buffer missing".to_string()))?;
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
            // Resolve HDR target to swapchain (always needed even if tonemapping is disabled)
            self.render_post_processing(command_buffer, image_index as usize)?;

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
            self.last_image_index = image_index;

            match present_result {
                Ok(()) => {
                    log::debug!(
                        "Frame {}: Successfully presented image {}",
                        self.current_frame,
                        image_index
                    );
                    if self.swapchain_cleanup_pending {
                        self.flush_old_swapchains();
                    }
                }
                Err(AshError::SwapchainOutOfDate(_)) => {
                    log::warn!(
                        "Frame {}: Swapchain out of date during presentation, requesting resize",
                        self.current_frame
                    );
                    self.request_swapchain_resize(swapchain_extent);
                    return Ok(());
                }
                Err(err) => {
                    log::error!(
                        "Frame {}: Failed to present image: {}",
                        self.current_frame,
                        err
                    );
                    return Err(err);
                }
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

    /// Updates the draw items to reflect the current material state
    pub fn refresh_draw_items(&mut self) {
        if self.mesh.is_some() && !self.draw_items.is_empty() {
            // Update the material in the first draw item (since we only support one mesh for now)
            if let Some(draw_item) = self.draw_items.get_mut(0) {
                draw_item.material = self.material.clone();
            }
        }
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
            // SAFETY: `forward_plus.upload_to_gpu` manages its own internal buffers. We provide the VMA allocator which is valid.
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
        let manager = self.descriptors.as_ref().ok_or(AshError::VulkanError(
            "DescriptorManager not initialized".to_string(),
        ))?;
        let bindless_manager = self.bindless_manager.as_mut().ok_or(AshError::VulkanError(
            "BindlessManager not initialized".to_string(),
        ))?;

        unsafe {
            indirect.init(
                &self.alloc.vma,
                &self.device,
                manager.frame_layout(),
                bindless_manager,
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
                .ok_or_else(|| AshError::RenderPassMissing("Fullscreen pass missing".to_string()))?
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
            // SAFETY: `create_sampler` is called with valid default parameters.
            unsafe {
                self.device
                    .device
                    .create_sampler(&vk::SamplerCreateInfo::default(), None)?
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
                .with_extent(
                    self.swapchain
                        .as_ref()
                        .ok_or_else(|| {
                            AshError::SwapchainMissing("Required for post-pipeline".to_string())
                        })?
                        .extent,
                )
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

        // SAFETY: `GpuProfiler::new` checks device limits internally.
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
        // Shutdown streamer FIRST to prevent background thread accessing resources while we destroy them
        // Shutdown streamer FIRST to prevent background thread accessing resources while we destroy them
        {
            let mut lock = self.texture_streamer.lock();
            if let Some(streamer) = lock.take() {
                drop(streamer); // Joins thread and destroys command pool
            }
        }

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

            if let Some(mut buffer) = self.material_storage_buffer.take() {
                let _ = buffer.cleanup();
            }

            self.model_renderer.clear();
            self.draw_items.clear();

            self.mesh = None;

            self.depth_buffer = None;
            self.pipeline = None;
            self.render_pass = None;
            self.swapchain = None;

            // IBL Cleanup
            if self.ibl_sampler != vk::Sampler::null() {
                self.device.device.destroy_sampler(self.ibl_sampler, None);
            }
            if let Some(mut pass) = self.brdf_lut_pass.take() {
                pass.destroy(&self.alloc.vma, &self.device.device);
            }
            self.ibl_manager.destroy();

            log::info!("Ash Renderer shut down successfully");
        }
    }
}

/// Helper to register a single texture with the bindless manager.
fn register_single_texture(
    bindless_manager: &mut vulkan::BindlessManager,
    texture_name: &str,
    texture: Option<&resources::texture::Texture>,
) -> Result<Option<u32>> {
    match texture {
        Some(tex) => match bindless_manager.add_sampled_image(tex.view(), tex.sampler()) {
            Ok(idx) => {
                log::debug!("Registered {texture_name} texture at bindless index {idx}");
                Ok(Some(idx))
            }
            Err(e) => {
                log::error!("Failed to register {texture_name} texture: {e}");
                Err(AshError::TextureBindingFailed(format!(
                    "{texture_name}: {e}"
                )))
            }
        },
        None => {
            log::debug!("{texture_name} texture not provided");
            Ok(None)
        }
    }
}

/// DRY helper to register all standard PBR textures for a mesh.
fn register_mesh_textures(
    mesh: &mut Mesh,
    bindless_manager: &mut vulkan::BindlessManager,
) -> Result<()> {
    mesh.texture_index =
        register_single_texture(bindless_manager, "base_color", mesh.texture.as_deref())?;
    mesh.normal_texture_index =
        register_single_texture(bindless_manager, "normal", mesh.normal_texture.as_deref())?;
    mesh.metallic_roughness_texture_index = register_single_texture(
        bindless_manager,
        "metallic_roughness",
        mesh.metallic_roughness_texture.as_deref(),
    )?;
    mesh.occlusion_texture_index = register_single_texture(
        bindless_manager,
        "occlusion",
        mesh.occlusion_texture.as_deref(),
    )?;
    mesh.emissive_texture_index = register_single_texture(
        bindless_manager,
        "emissive",
        mesh.emissive_texture.as_deref(),
    )?;
    Ok(())
}
