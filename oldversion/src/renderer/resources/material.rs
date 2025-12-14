use std::default::Default;

/// Material properties supporting a PBR workflow
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub color: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: [f32; 4],
    pub occlusion_strength: f32,
    pub normal_scale: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            color: [1.0, 1.0, 1.0, 1.0],
            roughness: 0.5,
            metallic: 0.0,
            emissive: [0.0, 0.0, 0.0, 1.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
        }
    }
}

impl Material {
    /// Creates a material with specific color
    pub fn with_color(name: impl Into<String>, color: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            color,
            roughness: 0.5,
            metallic: 0.0,
            emissive: [0.0, 0.0, 0.0, 1.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
        }
    }
}
