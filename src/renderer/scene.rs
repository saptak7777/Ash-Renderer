use crate::renderer::features::{DirectionalLight, PointLight, SceneLighting, SpotLight};
use crate::renderer::resources::DualHeapGeometryBuffer;
use crate::renderer::vcgs::OcclusionCulling;
use crate::renderer::*;
use crate::vulkan::Allocator;
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
}

impl Scene {
    pub fn new(
        device: Arc<ash::Device>,
        alloc: Arc<Allocator>,
        geometry_buffer: Arc<DualHeapGeometryBuffer>,
    ) -> Self {
        let model_renderer =
            ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device), geometry_buffer);

        Self {
            model_renderer,
            material_manager: MaterialManager::new(),
            occlusion_culling: OcclusionCulling::new(),
            point_lights: Vec::new(),
            directional_lights: Vec::new(),
            spot_lights: Vec::new(),
            scene_lighting: SceneLighting::default(),
            skybox_texture_index: 0,
        }
    }

    /// Add a material to the scene and return its handle.
    pub fn add_material(&mut self, material: Material) -> MaterialHandle {
        self.material_manager.register_material(material)
    }

    /// Add a light to the scene.
    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }

    pub fn add_directional_light(&mut self, light: DirectionalLight) {
        self.directional_lights.push(light);
    }

    pub fn set_skybox(&mut self, texture_index: u32) {
        self.skybox_texture_index = texture_index;
    }

    /// Set the lighting configuration for the scene.
    pub fn set_lighting(&mut self, lighting: SceneLighting) {
        self.scene_lighting = lighting;
    }
}
