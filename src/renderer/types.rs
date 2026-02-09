use ash::vk;
use bytemuck::Pod;
use glam::Mat4;
use std::sync::Arc;

use crate::renderer::resources::{Material, MaterialHandle, Mesh};
use crate::renderer::vcgs::CullBoundingBox;
use crate::vulkan;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DebugMode {
    #[default]
    None, // Final render
    Albedo,    // Visualize Albedo channel
    Normal,    // Visualize Normal channel
    Metallic,  // Visualize Metallic channel
    Roughness, // Visualize Roughness channel
    Lighting,  // Visualize Lighting only
}

#[derive(Clone, Debug)]
pub struct RenderCommand {
    /// Handle identifying the mesh to render
    pub mesh_handle: u32,
    /// Handle identifying the material to use
    pub material_handle: MaterialHandle,
    /// Transform matrix for positioning the mesh in world space
    pub transform: Mat4,
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
            cast_shadows: true,
            receive_shadows: true,
            is_transparent: false,
            is_hidden: false,
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

#[derive(Clone, Copy, Debug)]
pub struct GBufferIndices {
    pub depth_index: u32,
    pub motion_index: u32,
}

impl Default for GBufferIndices {
    fn default() -> Self {
        Self {
            depth_index: u32::MAX,
            motion_index: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SampleShadingQuality {
    Disabled, // Maximum performance
    Low,      // 25% samples
    #[default]
    Medium, // 50% samples
    High,     // 75% samples
    Full,     // 100% samples
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
    pub sample_shading: SampleShadingQuality,
    pub watch_shaders: bool,
    pub specialization_constants: Vec<SpecializationOverride>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_shading: SampleShadingQuality::Disabled,
            watch_shaders: false,
            specialization_constants: Vec::new(),
        }
    }
}

impl PipelineConfig {
    pub fn multisample_config(&self) -> vulkan::MultisampleConfig {
        vulkan::MultisampleConfig {
            sample_count: vk::SampleCountFlags::TYPE_1,
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

#[derive(Clone)]
pub struct DrawItem {
    pub key: Arc<str>,
    pub mesh_id: u32,
    pub transform: Mat4,
    pub material: Material,
    pub material_handle: MaterialHandle,
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
    pub bounds: CullBoundingBox,
    pub cluster_start_index: u32,
    pub cluster_count: u32,
}

impl Default for MeshData {
    fn default() -> Self {
        Self {
            name: Arc::from(""),
            texture_indices: [-1, -1, -1, -1],
            emissive_index: -1,
            texture_flags: TexturePresenceFlags::default(),
            material_handle: MaterialHandle { index: 0 },
            is_hidden: false,
            bounds: CullBoundingBox::default(),
            cluster_start_index: 0,
            cluster_count: 0,
        }
    }
}
