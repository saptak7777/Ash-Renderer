use crate::renderer::features::{DirectionalLight, PointLight, SceneLighting, SpotLight};
use crate::renderer::resources::DualHeapGeometryBuffer;
use crate::renderer::resources::TransformSystem;
use crate::renderer::vcgs::OcclusionCulling;
use crate::renderer::*;
use crate::vulkan::Allocator;
use crate::Result;
use std::collections::HashSet;
use std::sync::Arc;

/// Represents a 3D scene containing models, materials, and lights.
/// This decouples scene data from the Renderer logic.
pub struct Scene {
    pub model_renderer: ModelRenderer,
    pub material_manager: MaterialManager,
    pub occlusion_culling: OcclusionCulling,

    // Lighting Data
    pub point_lights: Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,
    pub spot_lights: Vec<SpotLight>,
    pub scene_lighting: SceneLighting, // IBL settings, etc.
    pub skybox_texture_index: u32,

    // Metadata & Tracking (Moved from Renderer)
    pub mesh_data: Vec<MeshData>,
    pub uploaded_material_indices: HashSet<u32>,
    pub transform_system: TransformSystem,
}

impl Scene {
    pub fn new(
        device: Arc<ash::Device>,
        alloc: Arc<Allocator>,
        geometry_buffer: Arc<DualHeapGeometryBuffer>,
    ) -> Result<Self> {
        let model_renderer =
            ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device), geometry_buffer);
        let transform_system = TransformSystem::new(Arc::clone(&device), Arc::clone(&alloc))?;

        Ok(Self {
            model_renderer,
            material_manager: MaterialManager::new(),
            occlusion_culling: OcclusionCulling::new(),
            point_lights: Vec::new(),
            directional_lights: Vec::new(),
            spot_lights: Vec::new(),
            scene_lighting: SceneLighting::default(),
            skybox_texture_index: 0,
            mesh_data: Vec::new(),
            uploaded_material_indices: HashSet::new(),
            transform_system,
        })
    }

    /// Register mesh metadata in the scene.
    pub fn register_mesh_metadata(&mut self, data: MeshData) {
        self.mesh_data.push(data);
    }

    /// Add a material to the scene and return its handle.
    /// Note: This registers with MaterialManager for dedup but assumes upload happens elsewhere
    /// or uses a scratch index until sync.
    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        // Use next_material_index from model_renderer as a hint
        let index = self.model_renderer.next_material_index;
        self.model_renderer.next_material_index += 1;
        self.material_manager.register_material(material, index)
    }

    /// Add a light to the scene.
    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }

    pub fn add_directional_light(&mut self, light: DirectionalLight) {
        self.directional_lights.push(light);
    }

    pub fn add_spot_light(&mut self, light: SpotLight) {
        self.spot_lights.push(light);
    }

    /// Set the lighting configuration for the scene.
    pub fn set_lighting(&mut self, lighting: SceneLighting) {
        self.scene_lighting = lighting;
    }
}
