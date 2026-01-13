use crate::{
    renderer::{
        async_readback::AsyncReadbackManager,
        diagnostics::{
            DiagnosticsMode, DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler,
        },
        features::{
            AutoRotateFeature, DirectionalLight, FeatureFrameContext, FeatureManager,
            FeatureRenderContext, PointLight, RenderFeature, ShadowFeature, SpotLight,
        },
        forward_plus_integration::ForwardPlusIntegration,
        fullscreen_pass, hdr_framebuffer,
        hiz_pass::HiZPass,
        indirect_draw::IndirectDrawPass,
        instancing::{BatchKey, InstanceData, InstancingManager},
        model_renderer::{
            DrawContext, MaterialPushConstants, ModelRenderer, DRAW_PUSH_FRAGMENT_BYTES,
            DRAW_PUSH_VERTEX_BYTES,
        },
        motion_pass::MotionVectorPass,
        occlusion_culling::{CullBoundingBox, OcclusionCulling},
        pass_manager::{RenderPassManager, RenderingMode},
        resource_registry::{ResourceId, ResourceRegistry},
        resources,
        resources::uniform::{StorageBuffer, UniformBuffer},
        ssgi_pass::{SsgiPass, SsgiQuality},
        temporal_aa::TaaConfig,
        temporal_upscaling::{VsrPass, VsrQuality},
        hiz_pass::AdaptiveHiZManager,
        vram_budget, DepthBuffer, GBuffer, Material, MaterialHandle, MaterialManager, Mesh,
        PipelineCache, SkinnedVertex, Texture, TextureData, Transform,
    },
    vulkan::{self, Allocator, BindlessManager, CommandBufferContext},
    AshError, Result,
};

use ash::vk;
use bytemuck::Pod;
use glam::{Mat4, Vec3, Vec4};
use parking_lot::Mutex;
use rayon::prelude::*;
use resources::BufferPool;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::renderer::resources::buffer::BufferHandle;
use crate::renderer::resources::mesh::{MaterialDescriptor, MeshDescriptor};

#[derive(Default)]
pub struct CullingManager {
    pub shadow_casters: HashSet<u32>,
    pub shadow_receivers: HashSet<u32>,
    pub transparent_objects: HashSet<u32>,
    pub gpu_driven_objects: HashSet<u32>,
    pub traditional_objects: HashSet<u32>,
}

impl CullingManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(
        &mut self,
        draw_items: &[DrawItem],
        instancing_manager: &InstancingManager,
        _mesh_data: &[MeshData],
    ) {
        self.gpu_driven_objects.clear();
        self.traditional_objects.clear();
        self.shadow_casters.clear();
        self.shadow_receivers.clear();
        self.transparent_objects.clear();

        // Register GPU-driven objects (from instancing manager)
        for batch in instancing_manager.batches() {
            self.gpu_driven_objects.insert(batch.key.mesh_id);
        }

        // Register traditional objects (non-batched)
        for item in draw_items {
            let mesh_handle = item.mesh_id;

            if !self.gpu_driven_objects.contains(&mesh_handle) {
                self.traditional_objects.insert(mesh_handle);
            }

            // Track shadow casters/receivers
            if item.cast_shadows {
                self.shadow_casters.insert(mesh_handle);
            }
            if item.receive_shadows {
                self.shadow_receivers.insert(mesh_handle);
            }

            // Track which are transparent
            if item.material.is_transparent {
                self.transparent_objects.insert(mesh_handle);
            }
        }
    }

    pub fn is_shadow_caster(&self, mesh_id: u32) -> bool {
        self.shadow_casters.contains(&mesh_id)
    }

    pub fn is_transparent(&self, mesh_id: u32) -> bool {
        self.transparent_objects.contains(&mesh_id)
    }
}

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
    /// Whether this object should receive shadows
    pub receive_shadows: bool,
    /// Whether this object is transparent
    pub is_transparent: bool,
    /// Whether this object is hidden from rendering
    pub is_hidden: bool,
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
            receive_shadows: true,
            is_transparent: false,
            is_hidden: false,
        }
    }
}

struct RendererResources {
    uniform_buffers: Vec<UniformBuffer>,
    joint_matrices_buffer: Vec<resources::JointMatricesBuffer>,
    default_texture: Texture,
    black_texture: Texture,
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SampleShadingQuality {
    Disabled,           // Maximum performance
    Low,               // 25% samples
    #[default]
    Medium,            // 50% samples
    High,              // 75% samples
    Full,              // 100% samples
}

impl SampleShadingQuality {
    pub fn min_sample_shading(&self) -> f32 {
        match self {
            Self::Disabled => 0.0,
            Self::Low => 0.25,
            Self::Medium => 0.5,
            Self::High => 0.75,
            Self::Full => 1.0,
        }
    }

    pub fn enabled(&self) -> bool {
        *self != Self::Disabled
    }
}

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub msaa: MsaaPreset,
    pub sample_shading: SampleShadingQuality,
    pub watch_shaders: bool,
    pub specialization_constants: Vec<SpecializationOverride>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            msaa: MsaaPreset::Off,
            sample_shading: SampleShadingQuality::Disabled,
            watch_shaders: false,
            specialization_constants: Vec::new(),
        }
    }
}

impl PipelineConfig {
    fn multisample_config(&self) -> vulkan::MultisampleConfig {
        vulkan::MultisampleConfig {
            sample_count: self.msaa.sample_count(),
            enable_sample_shading: self.sample_shading.enabled(),
            min_sample_shading: self.sample_shading.min_sample_shading(),
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
    prev_view_proj: Mat4,
    _default_texture: Texture,
    _black_texture: Texture,
    model_renderer: ModelRenderer,
    draw_items: Vec<DrawItem>,
    swapchain: Option<vulkan::SwapchainWrapper>,
    render_pass: Option<vulkan::RenderPass>,
    render_pass_id: Option<ResourceId>,
    /// Dedicated render pass for HDR rendering (ensures format compatibility)
    hdr_render_pass: Option<vulkan::RenderPass>,
    hdr_render_pass_id: Option<ResourceId>,
    pipeline: Option<vulkan::Pipeline>,
    pipeline_id: Option<ResourceId>,
    depth_buffer: Option<DepthBuffer>,
    uniform_buffers: Vec<UniformBuffer>,
    material_storage_buffer: Option<StorageBuffer<resources::uniform::MaterialUniform>>,
    material_buffer_index: u32,
    pipeline_layout: Option<vulkan::PipelineLayout>,
    pipeline_layout_id: Option<ResourceId>,
    descriptors: Option<vulkan::DescriptorManager>,
    framebuffers: Vec<vulkan::Framebuffer>,
    framebuffer_ids: Vec<ResourceId>,
    start_time: Instant,
    // mesh: Option<Mesh>,    // DELETED: Legacy field
    // material: Material,    // DELETED: Legacy field
    // transform: Transform,  // DELETED: Legacy field
    mesh_data: Vec<MeshData>, // Indexed by mesh handle for O(1) access
    material_manager: MaterialManager,
    uploaded_material_indices: HashSet<u32>, // Track which materials are GPU-resident (UE5 pattern)
    swapchain_image_view_ids: Vec<ResourceId>,
    depth_buffer_id: Option<ResourceId>,
    frame_sync_ids: Vec<(ResourceId, ResourceId, ResourceId)>,
    old_swapchain_handles: Vec<vk::SwapchainKHR>,
    swapchain_cleanup_pending: bool,
    resize_pending: bool,
    pending_extent: Option<vk::Extent2D>,
    // Post-processing support
    msaa_preset: MsaaPreset,
    sample_shading: SampleShadingQuality,
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
    async_readback: Option<AsyncReadbackManager>,
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
    adaptive_hiz_manager: AdaptiveHiZManager,
    indirect_draw_pass: Option<IndirectDrawPass>,
    occlusion_culling: OcclusionCulling,
    // Temporal Super-Resolution
    vsr_pass: Option<VsrPass>,
    // Screen-Space Global Illumination
    ssgi_pass: Option<SsgiPass>,
    // Motion Vector Pass for TSR/TAA
    motion_pass: Option<MotionVectorPass>,
    motion_framebuffer: Option<vk::Framebuffer>,
    // G-Buffer for Normals and Motion Vectors
    gbuffer: Option<GBuffer>,
    // Pipeline optimization
    culling_manager: CullingManager,
    use_gpu_driven: bool,
    // Lighting
    scene_lighting: crate::renderer::features::SceneLighting,
    point_lights: Vec<PointLight>,
    directional_lights: Vec<DirectionalLight>,
    spot_lights: Vec<SpotLight>,
    debug_visualization_enabled: bool,
    // Post-processing descriptors
    post_descriptor_pool: vk::DescriptorPool,
    post_descriptor_sets: Vec<vk::DescriptorSet>,
    post_pipeline: Option<vulkan::Pipeline>,
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
    transform_system: resources::TransformSystem,
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

    // TAA Configuration
    pub taa_config: TaaConfig,

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
pub struct DrawItem {
    pub key: Arc<str>,
    pub mesh_id: u32,
    pub transform: Mat4,
    pub material: Material,
    pub material_handle: MaterialHandle,
    pub texture_flags: TexturePresenceFlags,
    pub texture_indices: [i32; 4], // base, normal, mr, occ
    pub emissive_index: i32,
    pub is_skinned: bool,
    pub joint_offset: u32,
    pub alpha_cutoff: f32,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    pub is_hidden: bool,
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
    pub is_hidden: bool,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            name: Arc::from(""),
            texture_indices: [-1, -1, -1, -1],
            emissive_index: -1,
            texture_flags: TexturePresenceFlags::default(),
            material_handle: MaterialHandle { index: 0, version: 0 },
            is_hidden: false,
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

pub struct MainPassParameters<'a> {
    pub cmd_ctx: &'a CommandBufferContext<'a>,
    pub frame_index: usize,
    pub scene_pipeline: vk::Pipeline,
    pub pipeline_layout_handle: vk::PipelineLayout,
    pub batch_offsets: &'a HashMap<BatchKey, u32>,
    pub view: Mat4,
    pub projection: Mat4,
    pub swapchain_extent: vk::Extent2D,
}

impl Renderer {
    fn sample_shading_config(&self) -> vulkan::MultisampleConfig {
        vulkan::MultisampleConfig {
            sample_count: self.msaa_preset.sample_count(),
            enable_sample_shading: self.sample_shading.enabled(),
            min_sample_shading: self.sample_shading.min_sample_shading(),
        }
    }

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
            let vram_budget = vram_budget::VramBudget::new(&dev_mem_props);

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
            let mut pipeline_cfg = renderer_config.pipeline.clone();
            
            // Hardware fallback for sample shading
            if !device.sample_rate_shading_supported && pipeline_cfg.sample_shading.enabled() {
                log::warn!("Sample rate shading requested but not supported by hardware. Falling back to disabled.");
                pipeline_cfg.sample_shading = SampleShadingQuality::Disabled;
            }

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

            let model_renderer =
                ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device.device));

            let mut descriptor_manager = vulkan::DescriptorManager::new(
                Arc::clone(&device.device),
                framebuffers.len() as u32,
                None,
            )?;

            let aspect = swapchain.extent.width as f32 / swapchain.extent.height as f32;
            let max_bones = 1024;
            let renderer_resources = Self::init_resources(
                &alloc,
                &device,
                command_manager.upload_command_pool_handle(),
                framebuffers.len(),
                aspect,
                max_bones,
            )?;
            let RendererResources {
                uniform_buffers,
                joint_matrices_buffer,
                default_texture,
                black_texture,
                material_storage_buffer,
                instance_buffers,
            } = renderer_resources;

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
                    &pipeline_cfg,
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

            // DELETED: Default cube creation. Renderer now starts empty.
            // let mut mesh = Mesh::create_cube();
            // ...

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
            // DELETED: Legacy texture registration


            // DELETED: material_manager (unused)
            // DELETED: Legacy mesh/material initialization



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

            forward_plus.init_pipeline(
                Arc::clone(&device.device), 
                ibl_sampler,
                depth_buffer.view()
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

            let pass_manager = RenderPassManager::new(RenderingMode::GPUDriven);

            let mesh_data: Vec<MeshData> = Vec::new();
            let material_manager = MaterialManager::new();

            let mut renderer = Self {
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
                prev_view_proj: Mat4::IDENTITY,
                _default_texture: default_texture,
                _black_texture: black_texture,
                model_renderer,
                draw_items: Vec::new(),
                swapchain: Some(swapchain),
                render_pass: Some(render_pass),
                render_pass_id: Some(render_pass_id),
                hdr_render_pass: None,
                hdr_render_pass_id: None,
                pipeline: Some(pipeline),
                pipeline_id: Some(pipeline_id),
                depth_buffer: Some(depth_buffer),
                // DELETED: Legacy fields
                // mesh: Some(mesh),
                // material,
                // transform,
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
                uploaded_material_indices: HashSet::new(),
                swapchain_image_view_ids,
                depth_buffer_id: Some(depth_buffer_id),
                frame_sync_ids,
                old_swapchain_handles: Vec::new(),
                swapchain_cleanup_pending: false,
                resize_pending: false,
                pending_extent: Some(swapchain_extent),
                msaa_preset: pipeline_cfg.msaa,
                sample_shading: pipeline_cfg.sample_shading,
                hdr_framebuffer: None,
                fullscreen_pass: None,
                tonemapping_enabled: true,
                tonemapping_exposure: 1.2,
                tonemapping_gamma: 2.2,
                bloom_enabled: true,
                bloom_intensity: 0.1,
                diagnostics: DiagnosticsState::default(),
                frame_profiler: FrameProfiler::new(),
                gpu_profiler: None,
                async_readback: None,
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
                motion_pass: None,
                motion_framebuffer: None,
                gbuffer: Some(gbuffer),
                culling_manager: CullingManager::new(),
                use_gpu_driven: pass_manager.use_gpu_driven(),
                scene_lighting: crate::renderer::features::SceneLighting::default(),
                point_lights: Vec::new(),
                directional_lights: Vec::new(),
                spot_lights: Vec::new(),
                debug_visualization_enabled: false,
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
                transform_system: resources::TransformSystem::new(),
                instance_buffer_indices,
                joint_buffer_indices,
                pass_manager,
                brdf_lut_pass: Some(brdf_lut_pass),
                irradiance_map: None,
                prefiltered_map: None,
                ibl_sampler,
                ibl_manager,
                allow_auto_material: renderer_config.allow_auto_material,
                strict_mode: renderer_config.strict_mode,
                readback_buffer,
                last_image_index: 0,
                taa_config: TaaConfig::default(),
                adaptive_hiz_manager: AdaptiveHiZManager::new(3.0), // Target 3ms for Hi-Z
            };

            // Initialize dummy IBL cubemaps to prevent descriptor mismatches (samplerCube vs sampler2D)
            let dummy_irradiance = resources::ImageHandle::create_cubemap(
                Arc::clone(&renderer.device.device),
                Arc::clone(&renderer.alloc),
                1,
                1,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                Some("Default_Irradiance_Dummy".to_string()),
            )?;
            let dummy_prefiltered = resources::ImageHandle::create_cubemap(
                Arc::clone(&renderer.device.device),
                Arc::clone(&renderer.alloc),
                1,
                1,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                Some("Default_Prefiltered_Dummy".to_string()),
            )?;
            
            // Clear dummy cubemaps to white to prevent permanently black sides
            renderer.device.execute_single_use(renderer.cmds.upload_command_pool_handle(), |cmd| {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                };
                
                let barrier_to_dst = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(dummy_irradiance.handle())
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                
                let barrier_to_dst_2 = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(dummy_prefiltered.handle())
                    .subresource_range(range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                
                {
                    renderer.device.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier_to_dst, barrier_to_dst_2],
                    );
                    
                    let clear_color = vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] };
                    renderer.device.device.cmd_clear_color_image(cmd, dummy_irradiance.handle(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, &clear_color, &[range]);
                    renderer.device.device.cmd_clear_color_image(cmd, dummy_prefiltered.handle(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, &clear_color, &[range]);
                    
                    let barrier_to_read = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(dummy_irradiance.handle())
                        .subresource_range(range)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                        
                    let barrier_to_read_2 = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(dummy_prefiltered.handle())
                        .subresource_range(range)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ);
                        
                    renderer.device.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier_to_read, barrier_to_read_2],
                    );
                }
            })?;
            
            renderer.irradiance_map = Some(dummy_irradiance);
            renderer.prefiltered_map = Some(dummy_prefiltered);

            if let Some(manager) = renderer.descriptors.as_ref() {
                for index in 0..manager.frame_set_count() {
                    // Initial Shadow Map binding
                    if let Some(shadow_map) = renderer.shadow_feature.shadow_map() {
                        manager.bind_shadow_map(index, shadow_map.depth_image_view, shadow_map.sampler)?;
                    }

                    // Initial IBL binding (dummy/default textures until baked)
                    let dummy_resources = crate::vulkan::IBLResources {
                        irradiance_view: renderer.irradiance_map.as_ref().unwrap().view(),
                        prefiltered_view: renderer.prefiltered_map.as_ref().unwrap().view(),
                        brdf_lut_view: renderer.brdf_lut_pass.as_ref().unwrap().get_lut_view().unwrap(),
                        skybox_view: renderer.irradiance_map.as_ref().unwrap().view(), // Fallback
                        sampler: renderer.ibl_sampler,
                    };
                    manager.bind_ibl_resources(index, &dummy_resources)?;
                }
            }

            // Initial Forward+ upload and binding
            if let Some(forward_plus) = renderer.forward_plus.as_mut() {
                forward_plus.upload_to_gpu(&renderer.alloc.vma, &renderer.device.device)?;
            }

            // Initialize motion vector pass
            renderer.init_motion_pass()?;
            
            // Initialize async readback manager
            renderer.init_async_readback()?;

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

    /// Initialize motion vector pass for TSR/TAA
    ///
    /// # Safety
    /// Must be called after GBuffer is initialized
    pub unsafe fn init_motion_pass(&mut self) -> Result<()> {
        if self.motion_pass.is_some() {
            return Ok(()); // Already initialized
        }

        let _gbuffer = self.gbuffer.as_ref().ok_or(AshError::VulkanError(
            "GBuffer must be initialized before motion pass".to_string(),
        ))?;

        let mut motion_pass = MotionVectorPass::new(Arc::clone(&self.device.device));
        
        // Initialize with G-Buffer motion format
        let motion_format = vk::Format::R16G16_SFLOAT;
        motion_pass.init(&self.device, motion_format)?;

        self.motion_pass = Some(motion_pass);
        
        // Create motion framebuffer
        let swapchain = self.swapchain.as_ref().ok_or(AshError::VulkanError(
            "Swapchain not initialized".to_string(),
        ))?;
        self.create_motion_framebuffer(swapchain.extent.width, swapchain.extent.height)?;
        
        log::info!("Motion vector pass initialized");

        Ok(())
    }

    /// Initialize async readback manager
    ///
    /// # Safety
    /// Device must be valid
    pub unsafe fn init_async_readback(&mut self) -> Result<()> {
        if self.async_readback.is_some() {
            return Ok(());
        }

        let mut manager = AsyncReadbackManager::new(Arc::clone(&self.device.device));
        manager.init(&self.device)?;

        self.async_readback = Some(manager);
        
        log::info!("Async readback system initialized and integrated");
        Ok(())
    }

    /// Create framebuffer for motion vector pass
    ///
    /// # Safety
    /// GBuffer and motion pass must be initialized
    unsafe fn create_motion_framebuffer(&mut self, width: u32, height: u32) -> Result<()> {
        let gbuffer = self.gbuffer.as_ref().ok_or(AshError::VulkanError(
            "GBuffer not initialized".to_string(),
        ))?;

        let motion_pass = self.motion_pass.as_ref().ok_or(AshError::VulkanError(
            "Motion pass not initialized".to_string(),
        ))?;

        // Destroy old framebuffer if exists
        if let Some(old_fb) = self.motion_framebuffer.take() {
            self.device.device.destroy_framebuffer(old_fb, None);
        }

        let attachments = [gbuffer.motion_view()];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(motion_pass.render_pass())
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);

        let framebuffer = self.device.device.create_framebuffer(&framebuffer_info, None)?;
        self.motion_framebuffer = Some(framebuffer);

        log::debug!("Motion framebuffer created ({width}x{height})");
        Ok(())
    }

    fn init_resources(
        alloc: &Arc<vulkan::Allocator>,
        device: &vulkan::VulkanDevice,
        command_pool: vk::CommandPool,
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

        // Create black texture for IBL fallback (no ambient light when IBL not loaded)
        let black_texture_data = TextureData::solid_color([0, 0, 0, 255]);
        let black_texture = unsafe {
            Texture::from_data(
                Arc::clone(alloc),
                Arc::clone(&device.device),
                command_pool,
                device.graphics_queue,
                &black_texture_data,
                vk::Format::R8G8B8A8_SRGB,
                Some("black_texture"),
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
            black_texture,
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
        )?;

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
        let push_constant_ranges = [
            vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                offset: 0,
                size: DRAW_PUSH_VERTEX_BYTES + DRAW_PUSH_FRAGMENT_BYTES,
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
            size: 144, // lightSpaceMatrix(64) + model(64) + 4*u32(16)
        };

        let shadow_push_range_frag = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 144,
            size: 4, // base_color_index
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
            .with_vertex_input(
                vec![crate::renderer::skinned_vertex::SkinnedVertex::binding_description()],
                crate::renderer::skinned_vertex::SkinnedVertex::attribute_descriptions().to_vec(),
            )
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
            .as_ref()
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
            log::debug!("DEBUG: Beginning post-processing render pass");
            self.device.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            log::debug!("DEBUG: Binding post-processing pipeline");
            self.device.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );

            log::debug!("DEBUG: Binding post-processing descriptor sets");
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

            log::debug!("DEBUG: Pushing post-processing constants");
            self.device.device.cmd_push_constants(
                command_buffer,
                pass.pipeline_layout(),
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            // Draw 3 vertices for a single fullscreen triangle
            log::debug!("DEBUG: Drawing fullscreen triangle");
            self.device.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            log::debug!("DEBUG: Ending post-processing render pass");
            self.device.device.cmd_end_render_pass(command_buffer);
        }

        Ok(())
    }

    /// Uploads pre-loaded IBL data to the GPU and binds it.
    pub fn upload_ibl(
        &mut self,
        header: &resources::ibl_asset::IblAssetHeader,
        cubemap_data: &[u8],
        irradiance_data: &[u8],
        prefiltered_data: &[u8],
    ) -> Result<()> {
        log::info!("Uploading IBL assets to GPU...");

        let (_env_cubemap, irradiance, prefiltered) = resources::ibl_asset::upload_ibl(
            Arc::clone(&self.device.device),
            Arc::clone(&self.alloc),
            self.cmds.upload_command_pool_handle(),
            self.device.graphics_queue,
            header,
            cubemap_data,
            irradiance_data,
            prefiltered_data,
        )?;

        // Update renderer state
        self.irradiance_map = Some(irradiance);
        self.prefiltered_map = Some(prefiltered);

        // Update descriptors
        if let Some(manager) = self.descriptors.as_ref() {
            let res = crate::vulkan::IBLResources {
                irradiance_view: self.irradiance_map.as_ref().unwrap().view(),
                prefiltered_view: self.prefiltered_map.as_ref().unwrap().view(),
                brdf_lut_view: self.brdf_lut_pass.as_ref().unwrap().get_lut_view().unwrap(),
                skybox_view: self.irradiance_map.as_ref().unwrap().view(), // Use irradiance as dummy skybox
                sampler: self.ibl_sampler,
            };
            for i in 0..manager.frame_set_count() {
                manager.bind_ibl_resources(i, &res)?;
            }
        }
        
        log::info!("✓ IBL assets uploaded and bound.");
        Ok(())
    }


    // DELETED: set_mesh() method. Use submit_render_commands instead.

// DELETED Ok(())

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

    // Legacy lighting methods removed for modern RAGE pipeline


    /// Update a point light at the specified index.
    pub fn update_light(&mut self, index: usize, light: PointLight) {
        if index >= self.point_lights.len() {
            self.point_lights.resize(index + 1, PointLight::default());
        }
        self.point_lights[index] = light;
        
        // Sync with Forward+ if available
        if let Some(forward_plus) = &mut self.forward_plus {
            forward_plus.update_lights(&self.point_lights, &self.directional_lights, &self.spot_lights);
            // Upload to GPU so lights are visible
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma, &self.device.device);
            }
        }
    }

    /// Convenience for setting view and projection at once
    pub fn set_view(&mut self, _eye: Vec3, _center: Vec3, _up: Vec3) {
        // We calculate the view matrix here.
        // The projection is usually handled by the camera, but we can store it or calculate it.
        // For simplicity, let's just make this a no-op that logs for now, or actually store it.
        // Wait, renderer doesn't have a view matrix field yet.
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

    /// Uploads a mesh to the GPU and returns its handle.
    /// This is the modern replacement for `set_mesh`.
    pub fn upload_mesh(&mut self, mut mesh: Mesh) -> Result<u32> {
        let handle = self.mesh_data.len() as u32;
        self.register_mesh_handle(handle, &mut mesh)?;
        Ok(handle)
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
                        is_transparent: props.base_color_factor[3] < 1.0,
                        texture_index: mesh.texture_index,
                        normal_texture_index: mesh.normal_texture_index,
                        metallic_roughness_texture_index: mesh.metallic_roughness_texture_index,
                        occlusion_texture_index: mesh.occlusion_texture_index,
                        emissive_texture_index: mesh.emissive_texture_index,
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
                is_hidden: false,
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

    /// Updates the GPU material buffer with a material at the specified index (AAA-grade direct streaming)
    /// Uses single-element writes instead of read-modify-write to avoid GPU stalls and race conditions.
    /// This must be called after registering the material to ensure the GPU sees the correct material.
    ///
    /// This follows modern game engine patterns (UE5, Unity) where material updates are streamed
    /// directly without reading back the entire buffer.
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
            
            // Enable texture access by mapping indices from the material
            let base_idx = material.texture_index.map(|i| i as i32).unwrap_or(-1);
            let normal_idx = material.normal_texture_index.map(|i| i as i32).unwrap_or(-1);
            let mr_idx = material.metallic_roughness_texture_index.map(|i| i as i32).unwrap_or(-1);
            let occ_idx = material.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1);
            let emissive_idx = material.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

            // Validate texture indices before upload
            if let Some(_bindless) = &self.bindless_manager {
                let max_res = vulkan::BindlessManager::DEFAULT_MAX_TEXTURES;
                resources::bindless_validator::BindlessValidator::validate_indices(
                    &[base_idx, normal_idx, mr_idx, occ_idx, emissive_idx],
                    max_res,
                )?;
            }

            mat_uniform.set_texture_indices(base_idx, normal_idx, mr_idx, occ_idx, emissive_idx, -1);

            // AAA PATTERN: Direct streaming write instead of read-modify-write
            // This avoids:
            // 1. GPU stalls from reading entire buffer back to CPU
            // 2. Race conditions where GPU reads old data while CPU writes
            // 3. Unnecessary flush of unmodified elements
            unsafe {
                buffer.write_element_at(handle as usize, &mat_uniform)?;
            }

            // Track material as GPU-resident (UE5 pattern)
            self.uploaded_material_indices.insert(handle);

            // Synchronization handled by memory barrier in render_frame()
            // No need for device_wait_idle - this was a workaround
            // UE5/RAGE pattern: explicit barriers only, no global waits

            // Log the material data we just uploaded for debugging
            log::warn!(
                "✓ Material[{}] Streamed -> color={:?}, metallic={:.2}, roughness={:.2}",
                handle,
                material.color,
                material.metallic,
                material.roughness
            );

            #[cfg(debug_assertions)]
            {
                // Verify material was uploaded correctly (Unity/UE5 pattern)
                if let Some(buffer) = self.material_storage_buffer.as_ref() {
                    let test_mat = unsafe { buffer.read_element_at(handle as usize) };
                    log::debug!(
                        "Material[{}] verification: base_color={:?}, metallic={:.2}, roughness={:.2}",
                        handle,
                        test_mat.base_color_factor,
                        test_mat.parameters.x,
                        test_mat.parameters.y
                    );
                }
            }

            Ok(())
        } else {
            Err(AshError::VulkanError("Material storage buffer not initialized".to_string()))
        }
    }

    /// Standardized material registration and upload helper.
    /// 
    /// This handles both registering the material with the manager and 
    /// uploading its data to the GPU in a single call.
    pub fn register_and_upload_material(&mut self, material: Material) -> Result<MaterialHandle> {
        let handle = self.material_manager.register_material(material.clone());
        self.upload_material_to_gpu(handle.index as u32, &material)?;
        Ok(handle)
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
            receive_shadows: true,
            is_transparent: false,
            is_hidden: false,
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
                                    mesh_id: command.mesh_handle,
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
                                    receive_shadows: command.receive_shadows,
                                    is_hidden: command.is_hidden,
                                });
                            } else {
                                let key = BatchKey::new(command.mesh_handle, material_handle);
                                let mut instance = InstanceData::from_matrix(command.transform);
                                if command.cast_shadows {
                                    instance.set_flag(crate::renderer::occlusion_culling::CULL_FLAG_CAST_SHADOWS, true);
                                }
                                if command.is_hidden {
                                    instance.set_flag(crate::renderer::occlusion_culling::CULL_FLAG_HIDDEN, true);
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
                            mesh_id: command.mesh_handle,
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
                            receive_shadows: command.receive_shadows,
                            is_hidden: command.is_hidden,
                        });
                    } else {
                        let key = BatchKey::new(command.mesh_handle, material_handle);
                        let instance = InstanceData::from_matrix(command.transform)
                            .with_cast_shadows(command.cast_shadows)
                            .with_receive_shadows(command.receive_shadows)
                            .with_hidden(command.is_hidden);
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

        log::info!("Baking environment cubemap...");
        let env_cubemap = self.ibl_manager.create_cubemap_from_equirect(
            &self.device,
            self.cmds.upload_command_pool_handle(),
            equirect.view(),
            equirect.sampler(),
            512, // Standard resolution
        )?;

        log::info!("Baking irradiance map...");
        let irradiance_map = self.ibl_manager.generate_irradiance(
            &self.device,
            self.cmds.upload_command_pool_handle(),
            env_cubemap.view(),
            equirect.sampler(),
        )?;

        log::info!("Baking prefiltered reflection map...");
        // 3. Generate Prefiltered Map
        let prefiltered_map = self.ibl_manager.generate_prefiltered(
            &self.device,
            self.cmds.upload_command_pool_handle(),
            env_cubemap.view(),
            equirect.sampler(),
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
        
        // CRITICAL FIX: Update Forward+ tile calculations for new screen size
        // Without this, tile buffer remains at old size -> crash or black screen
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.on_resize(swapchain_extent.width, swapchain_extent.height);
            log::info!("Forward+ resized for {}x{}", swapchain_extent.width, swapchain_extent.height);
        }
        
        self.recreate_descriptor_sets()?;
        // 7. Recreate pipeline.
        self.recreate_pipeline()?;

        // 8. Recreate post-processing pipeline (depends on swapchain extent)
        if self.fullscreen_pass.is_some() {
            self.recreate_post_pipeline()?;
        }

        log::info!("Swapchain recreation complete ({image_count} images)");
        Ok(())
    }

    fn cleanup_framebuffers(&mut self) {
        // Main pass framebuffers
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

        // --- Post-Processing Framebuffers (CRITICAL FIX: Prevent Resource Leak) ---
        // These are vulkan::Framebuffer objects that need to be dropped to destroy their handles.
        for framebuffer in self.post_framebuffers.drain(..) {
            drop(framebuffer);
        }
    }

    fn cleanup_render_pass(&mut self) {
        if let Some(render_pass_id) = self.render_pass_id.take() {
            if let Err(e) = self.resources.cleanup_resource(render_pass_id) {
                log::warn!("Failed to cleanup render pass: {e}");
            }
        }
        self.render_pass = None;

        // Cleanup HDR render pass
        if let Some(hdr_render_pass_id) = self.hdr_render_pass_id.take() {
            if let Err(e) = self.resources.cleanup_resource(hdr_render_pass_id) {
                log::warn!("Failed to cleanup HDR render pass: {e}");
            }
        }
        self.hdr_render_pass = None;
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
        let render_pass = if self.hdr_framebuffer.is_some() {
             self.hdr_render_pass
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("HDR Render pass missing".to_string()))?
                .handle()
        } else {
             self.render_pass
                .as_ref()
                .ok_or_else(|| AshError::VulkanError("Render pass missing".to_string()))?
                .handle()
        };
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

        let multisample_config = self.sample_shading_config();

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
            include_bytes!(concat!(env!("OUT_DIR"), "/vert.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;
        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.frag.spv")),
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
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.frag.spv")),
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
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )
            .build()?;

        let render_pass_id = self
            .resources
            .register_render_pass(render_pass.handle())
            .map_err(|e| AshError::VulkanError(format!("Failed to register render pass: {e}")))?;
        render_pass.mark_managed_by_registry();
        
        // If we are in HDR mode, this pass is the HDR pass.
        // If we are in Swapchain mode, it's the standard pass.
        // To fix format mismatches, we ensure we have a valid handle in the correct field.
        if render_to_hdr {
            self.hdr_render_pass = Some(render_pass);
            self.hdr_render_pass_id = Some(render_pass_id);
            // We also need a swapchain-compatible pass for post-processing/UI, 
            // but create_render_pass_and_framebuffers is usually called when swapchain changes.
            // For now, if render_pass is missing, we create a default one too.
            if self.render_pass.is_none() {
                 let sw_builder = vulkan::RenderPass::builder(Arc::clone(&self.device.device));
                 let sw_pass = sw_builder.with_swapchain_color(color_format)
                     .with_depth_attachment(depth_buffer.format(), vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                     .build()?;
                 let sw_id = self.resources.register_render_pass(sw_pass.handle())?;
                 
                 // CRITICAL FIX: Mark render pass as managed to prevent double-free
                 let mut pass = sw_pass;
                 pass.mark_managed_by_registry();
                 
                 self.render_pass = Some(pass);
                 self.render_pass_id = Some(sw_id);
            }
        } else {
            self.render_pass = Some(render_pass);
            self.render_pass_id = Some(render_pass_id);
        }

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
                    matrices.model = Mat4::IDENTITY;
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

            // CRITICAL FIX: Recreate bindless manager descriptor set
            // This was the root cause of black screens during resize
            if let Some(bindless) = self.bindless_manager.as_mut() {
                log::info!("Recreating bindless descriptor set after swapchain resize...");
                bindless.recreate(manager.allocator_mut())?;
            }

            // CRITICAL FIX: Re-bind Environment (Set 2) resources after recreation
            // Without this, shadow mapping and IBL break after resize
            for index in 0..manager.frame_set_count() {
                // Re-bind Shadow Map
                if let Some(shadow_map) = self.shadow_feature.shadow_map() {
                    manager.bind_shadow_map(index, shadow_map.depth_image_view, shadow_map.sampler)?;
                }

                // Re-bind IBL Resources if they exist
                if let (Some(irr), Some(pre), Some(brdf)) = (
                    &self.irradiance_map,
                    &self.prefiltered_map,
                    &self.brdf_lut_pass,
                ) {
                    let resources = crate::vulkan::IBLResources {
                    irradiance_view: irr.view(),
                    prefiltered_view: pre.view(),
                    brdf_lut_view: brdf.get_lut_view().unwrap_or(self._default_texture.view()),
                    skybox_view: irr.view(), // Fallback to irradiance map for now
                    sampler: self.ibl_sampler,
                };
                    manager.bind_ibl_resources(index, &resources)?;
                } else {
                    // Fallback: Bind BRDF LUT and black textures if IBL is not loaded yet
                    // Black texture ensures no ambient light is added when IBL is not loaded
                    let dummy_resources = crate::vulkan::IBLResources {
                    irradiance_view: self._black_texture.view(),
                    prefiltered_view: self._black_texture.view(),
                    brdf_lut_view: self.brdf_lut_pass.as_ref()
                        .and_then(|p| p.get_lut_view())
                        .unwrap_or(self._black_texture.view()),
                    skybox_view: self._black_texture.view(),
                    sampler: self.ibl_sampler,
                };
                    manager.bind_ibl_resources(index, &dummy_resources)?;
                }
            }

            // CRITICAL FIX: Update Forward+ Descriptors (Set 3)
            // ForwardPlusIntegration's pool is separate but its bindings need to be refreshed
            if let Some(forward_plus) = self.forward_plus.as_mut() {
                unsafe {
                    forward_plus.upload_to_gpu(&self.alloc.vma, &self.device.device)?;
                }
            }
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
    pub fn render_shadow_pass(
        &mut self,
        cmd_ctx: &CommandBufferContext,
        frame_index: usize,
        light_space_matrix: Mat4,
        batch_offsets: &HashMap<BatchKey, u32>,
    ) -> Result<()> {
        let (shadow_pipeline, shadow_layout) = if let (Some(pipeline), Some(layout)) = (
            self.shadow_pipeline.as_ref(),
            self.shadow_pipeline_layout.as_ref(),
        ) {
            (pipeline, layout)
        } else {
            return Ok(());
        };

        let shadow_map = if let Some(map) = self.shadow_feature.shadow_map() {
            map
        } else {
            return Ok(());
        };

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
        cmd_ctx.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, shadow_pipeline.pipeline);

        cmd_ctx.set_viewport(0, &[shadow_map.viewport()]);
        cmd_ctx.set_scissor(0, &[shadow_map.scissor()]);

        // Bind Descriptor Sets for Shadows
        if let Some(manager) = self.descriptors.as_ref() {
            if let Some(frame_set) = manager.frame_set(frame_index) {
                cmd_ctx.bind_descriptor_sets(
                    vk::PipelineBindPoint::GRAPHICS,
                    shadow_layout.handle(),
                    0,
                    &[frame_set],
                    &[],
                );
            }
        }

        if let Some(ref bindless) = self.bindless_manager {
            cmd_ctx.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                shadow_layout.handle(),
                1,
                &[bindless.descriptor_set()],
                &[],
            );
        }

        // --- Shadow Rendering Paths ---

        // METHOD A: GPU-Driven Instancing (Batched)
        // This path handles high-performance instanced rendering of static shadow casters.
        if self.use_gpu_driven {
            for batch in self.instancing_manager.shadow_batches() {
                let mesh_data = if let Some(m) = self.mesh_data.get(batch.key.mesh_id as usize) {
                    m
                } else {
                    continue;
                };

                if let Some(uploaded) = self.model_renderer.get(&mesh_data.name) {
                    if let Some(&first_instance) = batch_offsets.get(&batch.key) {
                        log::debug!("Mesh {} using shadow GPU-driven path", mesh_data.name);
                        let push = crate::renderer::model_renderer::ShadowPushConstants {
                            light_space_matrix: crate::renderer::model_renderer::Mat4Push::from(light_space_matrix),
                            model: crate::renderer::model_renderer::Mat4Push::from(glam::Mat4::IDENTITY),
                            joint_offset: 0,
                            use_instancing: 1,
                            instance_buffer_index: self.instance_buffer_indices[frame_index],
                            joint_buffer_index: 0,
                            base_color_index: mesh_data.texture_indices[0],
                            _padding: [0; 3],
                        };

                        unsafe {
                            self.model_renderer.draw_mesh_instanced_shadow(
                                cmd_ctx.handle(),
                                shadow_layout.handle(),
                                uploaded,
                                batch.count() as u32,
                                first_instance,
                                &push,
                            );
                        }
                    }
                }
            }
        }

        // METHOD B: Legacy / Direct Rendering (Non-batched)
        // This path handles skinned meshes, single instances, and as a fallback
        // for objects not currently managed by the GPU-driven system.
        for item in &self.draw_items {
            let mesh_index = item.mesh_id;
            
            // Skip if not a shadow caster
            if !self.culling_manager.is_shadow_caster(mesh_index) {
                continue;
            }

            // GPU-Driven Path: Skip if already handled by the high-performance path
            if self.use_gpu_driven && self.culling_manager.gpu_driven_objects.contains(&mesh_index) {
                log::debug!("Mesh {mesh_index} skipping shadow legacy path: handled by GPU-driven");
                continue;
            }

            log::debug!("Mesh {mesh_index} using shadow legacy path");

            if let Some(uploaded) = self.model_renderer.get(&item.key) {
                let push = crate::renderer::model_renderer::ShadowPushConstants {
                    light_space_matrix: crate::renderer::model_renderer::Mat4Push::from(light_space_matrix),
                    model: crate::renderer::model_renderer::Mat4Push::from(item.transform),
                    joint_offset: item.joint_offset,
                    use_instancing: 0,
                    instance_buffer_index: 0,
                    joint_buffer_index: if item.is_skinned { self.joint_buffer_indices[frame_index] } else { 0 },
                    base_color_index: item.texture_indices[0],
                    _padding: [0; 3],
                };

                unsafe {
                    self.model_renderer.draw_mesh_shadow(
                        cmd_ctx.handle(),
                        shadow_layout.handle(),
                        uploaded,
                        &push,
                    );
                }
            }
        }

        cmd_ctx.end_render_pass();
        Ok(())
    }

    pub fn render_main_pass(
        &mut self,
        params: &MainPassParameters,
    ) -> Result<()> {
        let cmd_ctx = params.cmd_ctx;
        let frame_index = params.frame_index;
        let scene_pipeline = params.scene_pipeline;
        let pipeline_layout_handle = params.pipeline_layout_handle;
        let batch_offsets = params.batch_offsets;
        let view = params.view;
        let projection = params.projection;
        let swapchain_extent = params.swapchain_extent;

        // --- Main Pass Rendering Paths ---

        // 1. Render all visible opaque instanced batches (High-performance Path)
        // GPU-Driven Path: Used for large numbers of static meshes with instancing.
        let mut current_object_offset = 0;
        for batch in self.instancing_manager.visible_batches() {
            if let Some(mesh_data) = self.mesh_data.get(batch.key.mesh_id as usize) {
                let mesh_key = &mesh_data.name;

                if let Some(uploaded) = self.model_renderer.get(mesh_key) {
                    log::debug!("Batch for mesh {mesh_key} using main GPU-driven/instanced path");

                    let material_push = MaterialPushConstants::new(batch.key.material_id)
                        .with_material_buffer_index(self.material_buffer_index)
                        .with_receive_shadows(batch.receives_shadows())
                        .with_debug_visualization(self.debug_visualization_enabled);

                    // --- GPU-Driven Indirect Path ---
                    if self.use_gpu_driven && self.indirect_draw_pass.is_some() {
                        let indirect = self.indirect_draw_pass.as_mut().unwrap();
                        // 1. Upload instances for this batch to the object buffer
                        unsafe {
                            indirect.upload_objects(
                                &self.alloc.vma,
                                &batch.instances,
                                current_object_offset,
                            )?;
                        }

                        // 2. Upload draw template for this mesh
                        let template = vk::DrawIndexedIndirectCommand {
                            index_count: uploaded.index_count(),
                            instance_count: 1,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: 0, // Set by compute shader
                        };
                        unsafe {
                            indirect.upload_templates(&self.alloc.vma, &[template], 0)?;
                        }

                        // 3. Run Culling Compute Shader
                        let bindless_manager =
                            self.bindless_manager.as_ref().expect("Bindless manager");
                        unsafe {
                            indirect.execute_culling(
                                cmd_ctx.handle(),
                                &self.occlusion_culling,
                                bindless_manager,
                                projection * view,
                                swapchain_extent.width,
                                swapchain_extent.height,
                                current_object_offset as u32,
                                batch.count() as u32,
                                0, // indirect_offset
                            )?;
                        }

                        // 4. Draw Indirect
                        let material_push = material_push.with_debug_path(1); // 1: GPU-Driven
                        let ctx = DrawContext {
                            command_buffer: cmd_ctx.handle(),
                            pipeline_layout: pipeline_layout_handle,
                            uploaded,
                            material: &material_push,
                            instance_buffer_index: indirect.object_buffer_index().unwrap_or(0),
                            joint_buffer_index: self.joint_buffer_indices[frame_index],
                        };
                        unsafe {
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
                        }

                        current_object_offset += batch.count();
                    } else {
                        // --- Direct Instanced Path Fallback ---
                        let material_push = material_push
                            .with_debug_path(2) // 2: Legacy/Direct
                            .with_debug_visualization(self.debug_visualization_enabled);
                        let ctx = DrawContext {
                            command_buffer: cmd_ctx.handle(),
                            pipeline_layout: pipeline_layout_handle,
                            uploaded,
                            material: &material_push,
                            instance_buffer_index: self.instance_buffer_indices[frame_index],
                            joint_buffer_index: self.joint_buffer_indices[frame_index],
                        };

                        let offset = batch_offsets.get(&batch.key).copied().unwrap_or(0);
                        unsafe {
                            self.model_renderer.draw_mesh_instanced(
                                &ctx,
                                batch.count() as u32,
                                offset,
                            );
                        }
                    }
                }
            }
        }

        // 2. Draw remaining opaque meshes (Legacy / Direct Path)
        // Legacy Path: Used for direct mesh rendering (skinned meshes, single instances, fallbacks).
        {
            let mut current_pipeline = scene_pipeline;
            for item in &self.draw_items {
                let mesh_index = item.mesh_id;
                
                // Skip if already handled by GPU-driven path
                if self.use_gpu_driven && self.culling_manager.gpu_driven_objects.contains(&mesh_index) {
                    log::debug!("Mesh {} skipping main legacy path: handled by GPU-driven", item.key);
                    continue;
                }

                // Skip if transparent (separate pass)
                if self.culling_manager.is_transparent(mesh_index) {
                    continue;
                }
                
                // Skip if hidden
                if item.is_hidden {
                    continue;
                }

                log::debug!("Mesh {} using main legacy path", item.key);

                if let Some(uploaded) = self.model_renderer.get(&item.key) {
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
                    
                    let material_handle = item.material_handle;
                    let material_push = MaterialPushConstants::new(material_handle)
                        .with_material_buffer_index(self.material_buffer_index)
                        .with_receive_shadows(item.receive_shadows)
                        .with_debug_path(2) // 2: Legacy (Direct path)
                        .with_debug_visualization(self.debug_visualization_enabled);

                    let ctx = DrawContext {
                        command_buffer: cmd_ctx.handle(),
                        pipeline_layout: pipeline_layout_handle,
                        uploaded,
                        material: &material_push,
                        instance_buffer_index: self.instance_buffer_indices[frame_index],
                        joint_buffer_index: self.joint_buffer_indices[frame_index],
                    };

                    unsafe {
                        self.model_renderer
                            .draw_mesh(&ctx, item.transform, item.joint_offset);
                    }
                }
            }
        }

        // 3. Draw all transparent meshes
        {
            let mut current_pipeline = scene_pipeline;
            
            // 3a. Draw transparent instanced batches
            for batch in self.instancing_manager.transparent_batches() {
                if let Some(mesh_data) = self.mesh_data.get(batch.key.mesh_id as usize) {
                    let mesh_key = &mesh_data.name;
                    if let Some(uploaded) = self.model_renderer.get(mesh_key) {
                        log::debug!("Transparent batch for mesh {mesh_key} using main legacy path");

                        let material_push = MaterialPushConstants::new(batch.key.material_id)
                            .with_material_buffer_index(self.material_buffer_index)
                            .with_receive_shadows(batch.receives_shadows())
                            .with_debug_path(2) // 2: Legacy (Direct path)
                            .with_debug_visualization(self.debug_visualization_enabled);
                        
                        let ctx = DrawContext {
                            command_buffer: cmd_ctx.handle(),
                            pipeline_layout: pipeline_layout_handle,
                            uploaded,
                            material: &material_push,
                            instance_buffer_index: self.instance_buffer_indices[frame_index],
                            joint_buffer_index: self.joint_buffer_indices[frame_index],
                        };

                        let offset = batch_offsets.get(&batch.key).copied().unwrap_or(0);
                        unsafe {
                            self.model_renderer.draw_mesh_instanced(
                                &ctx,
                                batch.count() as u32,
                                offset,
                            );
                        }
                    }
                }
            }

            // 3b. Draw transparent traditional meshes
            for item in &self.draw_items {
                let mesh_index = item.mesh_id;
                
                // Skip if not transparent
                if !self.culling_manager.is_transparent(mesh_index) {
                    continue;
                }

                // Skip if handled by GPU-driven
                if self.use_gpu_driven && self.culling_manager.gpu_driven_objects.contains(&mesh_index) {
                    continue;
                }
                
                if let Some(uploaded) = self.model_renderer.get(&item.key) {
                    log::debug!("Transparent mesh {} using main legacy path", item.key);

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

                    let material_handle = item.material_handle;
                    let material_push = MaterialPushConstants::new(material_handle)
                        .with_material_buffer_index(self.material_buffer_index)
                        .with_receive_shadows(item.receive_shadows)
                        .with_debug_path(2); // 2: Legacy (Direct path)

                    let ctx = DrawContext {
                        command_buffer: cmd_ctx.handle(),
                        pipeline_layout: pipeline_layout_handle,
                        uploaded,
                        material: &material_push,
                        instance_buffer_index: self.instance_buffer_indices[frame_index],
                        joint_buffer_index: self.joint_buffer_indices[frame_index],
                    };

                    unsafe {
                        self.model_renderer
                            .draw_mesh(&ctx, item.transform, item.joint_offset);
                    }
                }
            }
        }

        Ok(())
    }

    /// Set scene lighting configuration (RAGE)
    pub fn set_lighting(&mut self, lighting: &crate::renderer::features::SceneLighting) {
        self.scene_lighting = *lighting;
        
        // Sync shadow direction
        let direction = Vec4::from_array(lighting.directional.direction).truncate();
        self.shadow_feature.set_light_direction(direction);
    }

    pub fn render_frame(
        &mut self,
        view: Mat4,
        projection: Mat4,
        camera_pos: glam::Vec3,
        model_matrix: Option<Mat4>,
    ) -> Result<()> {
        self.transform_system.update();
        self.flush_old_swapchains();

        // Recycle per-frame descriptor pools (static pools are unaffected)
        if let Some(dm) = self.descriptors.as_mut() {
            dm.next_frame();
        }

        // Poll for async GPU readbacks
        if let Some(readback) = self.async_readback.as_mut() {
            unsafe {
                let _ = readback.poll_completed();
                readback.next_frame();
            }
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
            let main_render_pass = if self.hdr_framebuffer.is_some() {
                self.hdr_render_pass
                    .as_ref()
                    .map(|p| p.handle())
                    .ok_or_else(|| AshError::VulkanError("HDR render pass not available".to_string()))?
            } else {
                self.render_pass
                    .as_ref()
                    .map(|p| p.handle())
                    .ok_or_else(|| AshError::VulkanError("Render pass not available".to_string()))?
            };

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

            // DELETED: Transform overwrite bug (lines 4105-4108)
            // if let Some(item) = self.draw_items.get_mut(0) {
            //     item.transform = self.transform.model_matrix();
            // }

            // Build object registry once per frame
            self.culling_manager.build(&self.draw_items, &self.instancing_manager, &self.mesh_data);

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
                // Update adaptive quality based on previous frame's metrics
                if let Some(ref mut profiler) = self.gpu_profiler {
                    let timings = profiler.last_extended_timings();
                    if timings.valid {
                        let hiz_time_ms = timings.hiz_generate_ms as f64;
                        if let Some(new_quality) = self.adaptive_hiz_manager.update(hiz.quality(), hiz_time_ms) {
                            hiz.set_quality(new_quality);
                        }
                    }
                }

                // Hiz pyramid is built from previous frame's depth
                hiz.build_pyramid(command_buffer, depth_buffer.image())?;
                
                // Write timestamp after Hi-Z generation
                if let Some(ref profiler) = self.gpu_profiler {
                    profiler.write_timestamp(command_buffer, crate::renderer::diagnostics::TimingScope::HiZGenerateEnd);
                }
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

                // Create dummy transform for legacy feature compatibility
                let mut dummy_transform = resources::Transform::identity();

                let elapsed = self.start_time.elapsed().as_secs_f32();
                let mut feature_ctx = FeatureFrameContext {
                    device: self.device.device.as_ref(),
                    descriptor_manager: self.descriptors.as_ref(),
                    transform: &mut dummy_transform, // Use dummy
                    auto_rotate: false, // Auto-rotate now handled by examples
                    elapsed_seconds: elapsed,
                };
                self.features.before_frame(&mut feature_ctx);
                self.shadow_feature.before_frame(&mut feature_ctx);

                // Matrices provided via function arguments.
                let matrices = uniform_buffer.matrices_mut();
                // Use pre-calculated normal matrix for the provided model matrix
                let model = model_matrix.unwrap_or(Mat4::IDENTITY);
                let mut transform = resources::Transform::identity();
                transform.set_model(model);
                
                matrices.model = model;
                matrices.normal_matrix = Mat4::from_mat3(transform.normal_matrix());
                matrices.view = view;
                matrices.projection = jittered_projection;
                matrices.view_proj = jittered_projection * view;
                matrices.prev_view_proj = self.prev_view_proj;
                matrices.camera_pos = camera_pos.extend(1.0);
                matrices.set_lighting(&self.scene_lighting);

                // Set light-space matrix for shadow mapping
                let light_space_matrix = self.shadow_feature.light_space_matrix();
                matrices.set_light_space_matrix(light_space_matrix);
                // REDUNDANT OVERWRITE REMOVED: Using pre-calculated normal matrix from transform_system
                // matrices.normal_matrix = matrices.model.inverse().transpose();

                let view_proj = matrices.view_proj;
                uniform_buffer.update()?;
                self.prev_view_proj = view_proj;
            }

            // CRITICAL: Ensure all host-written buffers (including bindless storage buffers) are visible to GPU
            // This is required because examples might update buffers directly on the host.
            //
            // Synchronization Pattern (UE5/RAGE/Unity):
            // - HOST_WRITE: All CPU-side buffer writes complete
            // - SHADER_READ: Vertex/fragment/compute shaders can read all buffer types (uniform, storage, etc.)
            // - UNIFORM_READ: Explicit uniform buffer reads in all stages
            //
            // Note: Vulkan's SHADER_READ flag already covers storage buffer reads. There is no separate
            // SHADER_STORAGE_READ flag in Ash/Vulkan - SHADER_READ is the comprehensive flag for all shader reads.
            //
            // This barrier ensures GPU cache coherency for all buffer types before rendering begins.
            let global_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::UNIFORM_READ
                );

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

            let device_arc = Arc::clone(&self.device.device);
            let cmd_ctx = CommandBufferContext::new(device_arc.as_ref(), command_buffer);
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
            // --- Phase 8: Instance Data Preparation ---
            // 1. Prepare and upload all instances to the InstanceBuffer
            let mut all_instances = Vec::new();
            let mut batch_offsets = HashMap::new();
            {
                for batch in self.instancing_manager.batches() {
                    batch_offsets.insert(batch.key.clone(), all_instances.len() as u32);
                    all_instances.extend_from_slice(&batch.instances);
                }
            }

            if !all_instances.is_empty() {
                self.instance_buffer[frame_index].update(&all_instances)?;
            }

            // Shadow Pass
            let light_space_matrix = self.shadow_feature.light_space_matrix();
            self.render_shadow_pass(&cmd_ctx, frame_index, light_space_matrix, &batch_offsets)?;

            // --- Light Culling Compute Dispatch ---
            if let Some(ref mut fp_integration) = self.forward_plus {
                // Ensure pipeline is initialized before dispatching
                if fp_integration.is_enabled() {
                    // Update camera buffer with current view/projection matrices
                    // We use the NON-JITTERED projection for culling to match frustum
                    fp_integration.update_camera(
                        &self.alloc.vma,
                        &view.to_cols_array_2d(),
                        &projection.to_cols_array_2d(),
                        &camera_pos.extend(1.0).into(),
                    )?;

                    // Dispatch the compute shader
                    fp_integration.dispatch(
                        command_buffer,
                        &self.device.device,
                    );
                }
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0], // 0: Color (HDR or Swapchain)
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0], // 1: G-Buffer Normal
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0], // 2: G-Buffer Albedo (Clear to opaque black)
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0], // 3: G-Buffer Motion
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0, // 4: Depth
                        stencil: 0,
                    },
                },
            ];

            let (framebuffer_handle, hdr_attachment) = {
                let framebuffer = self.framebuffers.get(image_index as usize).ok_or_else(|| {
                    log::error!(
                        "Frame {}: Framebuffer index {} out of range (max: {})",
                        self.current_frame,
                        image_index,
                        self.framebuffers.len()
                    );
                    AshError::VulkanError("Framebuffer index out of range".into())
                })?;
                (framebuffer.handle(), framebuffer.attachments()[0])
            };

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(main_render_pass)
                .framebuffer(framebuffer_handle)
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

            let dummy_render_transform = Transform::identity(); // Local dummy for render ctx

            let render_ctx = FeatureRenderContext {
                device: self.device.device.as_ref(),
                descriptor_manager: self.descriptors.as_ref(),
                command_buffer,
                transform: &dummy_render_transform, // Use local dummy
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
                                skybox_view: irradiance.view(), // Fallback
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
                        log::debug!("DEBUG: Binding Forward+ descriptor set (Set 3)");
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
            let main_pass_params = MainPassParameters {
                cmd_ctx: &cmd_ctx,
                frame_index,
                scene_pipeline,
                pipeline_layout_handle,
                batch_offsets: &batch_offsets,
                view,
                projection: jittered_projection,
                swapchain_extent,
            };
            self.render_main_pass(&main_pass_params)?;

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
                // Read back metrics from previous frame (non-blocking)
                vsr.readback_metrics(command_buffer, &self.alloc.vma);

                let depth_buffer = self
                    .depth_buffer
                    .as_ref()
                    .ok_or_else(|| AshError::VulkanError("Depth buffer missing".to_string()))?;
                // Pass current color result (which is now correctly the HDR buffer if initialized)
                // and upsample to VSR history.
                vsr.upscale(
                    command_buffer,
                    hdr_attachment, // Index 0 is now HDR if active
                    depth_buffer.view(),
                    gbuffer.motion_view(),
                    jitter_uv,
                    &self.taa_config,
                )?;
                vsr.next_frame();
            }

            // --- Post-Processing (Tonemapping & Resolve) ---
            // CRITICAL FIX: Transition HDR buffer to SHADER_READ_ONLY for sampling in tonemapping shader
            if let Some(hdr) = &self.hdr_framebuffer {
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(hdr.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            // Resolve HDR target to swapchain (always needed even if tonemapping is disabled)
            log::debug!("DEBUG: About to call render_post_processing");
            self.render_post_processing(command_buffer, image_index as usize)?;
            log::debug!("DEBUG: render_post_processing completed successfully");

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

    // DELETED: transform(), transform_mut() accessors

    pub fn buffer_pool(&self) -> Arc<BufferPool> {
        Arc::clone(&self.buffer_pool)
    }

    // DELETED: Legacy accessor methods (mesh_mut, material, material_mut, refresh_draw_items)


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

    /// Enables or disables debug path visualization tints in the shader.
    pub fn set_debug_visualization(&mut self, enabled: bool) {
        self.debug_visualization_enabled = enabled;
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
            forward_plus.update_lights(lights, &[], &[]);
            // Upload to GPU
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma, &self.device.device);
            }
        }
    }

    /// Update directional lights for Forward+ rendering
    pub fn update_directional_lights(
        &mut self,
        lights: &[crate::renderer::features::DirectionalLight],
    ) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(&[], lights, &[]);
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma, &self.device.device);
            }
        }
    }

    /// Update spot lights for Forward+ rendering
    pub fn update_spot_lights(&mut self, lights: &[crate::renderer::features::SpotLight]) {
        if let Some(ref mut forward_plus) = self.forward_plus {
            forward_plus.update_lights(&[], &[], lights);
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma, &self.device.device);
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
            forward_plus.update_lights(point_lights, directional_lights, &[]);
            // SAFETY: `forward_plus.upload_to_gpu` manages its own internal buffers. We provide the VMA allocator which is valid.
            unsafe {
                let _ = forward_plus.upload_to_gpu(&self.alloc.vma, &self.device.device);
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
        
        // CRITICAL: Recreate main pipeline and framebuffers to use the NEW HDR render pass format
        self.recreate_swapchain_resources()?;
        
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
            vsr.output_view([
                crate::renderer::temporal_aa::SharpeningMode::Subtle,
                crate::renderer::temporal_aa::SharpeningMode::Moderate,
                crate::renderer::temporal_aa::SharpeningMode::Strong,
            ].contains(&self.taa_config.sharpening))
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
        
        // --- CRITICAL FIX: Bloom Placeholder ---
        // Binding the color view directly to bloom causes over-brightness because the tonemapping 
        // shader adds 'bloom' to the original color. Until a real bloom pass is implemented, 
        // we should bind a target that is logically black.
        // PREVIOUS BUG: let bloom_view = ssgi_view; // This caused flickering SSGI to appear as blinking glow
        let bloom_view = self._black_texture.view(); 
        
        log::debug!("Updating post-processing descriptors (HDR: {}, Bloom: {})", 
            hdr.is_some(), self.bloom_enabled);

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
            self.post_pipeline = Some(pipeline);
            // Pipeline cleanup is handled by RAII in vulkan::Pipeline wrapper.
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
        if let Some(ref mut profiler) = self.gpu_profiler {
            profiler.enabled = self.diagnostics.mode != DiagnosticsMode::Off;
        }
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

    /// Log quality reports for debug/profiling
    pub fn log_quality_reports(&self) {
        if let Some(hiz) = &self.hiz_pass {
            log::info!("{}", hiz.quality_report());
        }

        if let Some(vsr) = &self.vsr_pass {
            log::info!("{}", vsr.quality_report(self.taa_config.quality, self.taa_config.sharpening));
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

            // CRITICAL FIX: Explicitly drop post-processing resources before general resource cleanup.
            // This prevents access violations during shutdown if the window/surface is destroyed.
            // ORDER MATTERS: Pipeline depends on RenderPass (in FullscreenPass), so destroy Pipeline FIRST.
            self.post_pipeline = None;

            self.fullscreen_pass = None;
            self.hdr_framebuffer = None;
            
            self.cleanup_framebuffers(); // Drains post_framebuffers
            self.cleanup_render_pass();  // Drains hdr_render_pass
            self.cleanup_pipeline();

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
                fp.destroy(&self.alloc.vma, &self.device.device);
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

            // Cleanup async readback manager
            if let Some(mut readback) = self.async_readback.take() {
                readback.destroy();
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

            // DELETED: Legacy mesh cleanup

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
