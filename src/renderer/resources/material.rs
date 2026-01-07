use std::collections::HashMap;
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
    pub alpha_cutoff: f32,
    pub tint_index: i32,
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
            alpha_cutoff: 0.1,
            tint_index: -1,
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
            alpha_cutoff: 0.1,
            tint_index: -1,
        }
    }
}

/// Quantized material key for stable hashing
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MaterialKey {
    color_r: u8,
    color_g: u8,
    color_b: u8,
    color_a: u8,
    metallic: u8,
    roughness: u8,
    emissive_r: u8,
    emissive_g: u8,
    emissive_b: u8,
    emissive_a: u8,
    occlusion_strength: u8,
    normal_scale: u8,
    alpha_cutoff: u8,
    tint_index: i32,
}

impl MaterialKey {
    pub fn from_material(material: &Material) -> Self {
        fn quantize(v: f32) -> u8 {
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        }
        Self {
            color_r: quantize(material.color[0]),
            color_g: quantize(material.color[1]),
            color_b: quantize(material.color[2]),
            color_a: quantize(material.color[3]),
            metallic: quantize(material.metallic),
            roughness: quantize(material.roughness),
            emissive_r: quantize(material.emissive[0]),
            emissive_g: quantize(material.emissive[1]),
            emissive_b: quantize(material.emissive[2]),
            emissive_a: quantize(material.emissive[3]),
            occlusion_strength: quantize(material.occlusion_strength),
            normal_scale: quantize(material.normal_scale),
            alpha_cutoff: quantize(material.alpha_cutoff),
            tint_index: material.tint_index,
        }
    }

    pub fn from_props(props: &crate::renderer::resources::mesh::MaterialProperties) -> Self {
        fn quantize(v: f32) -> u8 {
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        }
        Self {
            color_r: quantize(props.base_color_factor[0]),
            color_g: quantize(props.base_color_factor[1]),
            color_b: quantize(props.base_color_factor[2]),
            color_a: quantize(props.base_color_factor[3]),
            metallic: quantize(props.metallic_factor),
            roughness: quantize(props.roughness_factor),
            emissive_r: quantize(props.emissive_factor[0]),
            emissive_g: quantize(props.emissive_factor[1]),
            emissive_b: quantize(props.emissive_factor[2]),
            emissive_a: quantize(props.emissive_factor[3]),
            occlusion_strength: quantize(props.occlusion_strength),
            normal_scale: quantize(props.normal_scale),
            alpha_cutoff: quantize(props.alpha_cutoff),
            tint_index: -1,
        }
    }
}

/// A centralized registry for materials with O(1) deduplication
pub struct MaterialRegistry {
    materials: HashMap<u32, Material>,
    key_to_handle: HashMap<MaterialKey, u32>,
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialRegistry {
    pub fn new() -> Self {
        Self {
            materials: HashMap::new(),
            key_to_handle: HashMap::new(),
        }
    }

    /// Returns handle of existing material or registers new one
    pub fn get_or_register(&mut self, handle: u32, material: Material) -> u32 {
        let key = MaterialKey::from_material(&material);
        if let Some(&existing_handle) = self.key_to_handle.get(&key) {
            return existing_handle;
        }

        self.key_to_handle.insert(key, handle);
        self.materials.insert(handle, material);
        handle
    }

    pub fn get_handle_by_key(&self, key: MaterialKey) -> Option<u32> {
        self.key_to_handle.get(&key).copied()
    }

    pub fn insert(&mut self, handle: u32, material: Material) {
        let key = MaterialKey::from_material(&material);
        self.key_to_handle.insert(key, handle);
        self.materials.insert(handle, material);
    }

    pub fn get(&self, handle: u32) -> Option<&Material> {
        self.materials.get(&handle)
    }

    pub fn get_mut(&mut self, handle: u32) -> Option<&mut Material> {
        self.materials.get_mut(&handle)
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<u32, Material> {
        self.materials.iter()
    }

    pub fn clear(&mut self) {
        self.materials.clear();
        self.key_to_handle.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_key_stability() {
        let mat1 = Material {
            name: "mat1".to_string(),
            color: [0.5, 0.5, 0.5, 1.0],
            ..Default::default()
        };
        let mat2 = Material {
            name: "mat2".to_string(),
            color: [0.5, 0.5, 0.5, 1.0], // Identical properties, different name
            ..Default::default()
        };

        let key1 = MaterialKey::from_material(&mat1);
        let key2 = MaterialKey::from_material(&mat2);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_material_deduplication() {
        let mut registry = MaterialRegistry::new();
        let mat1 = Material {
            color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.5,
            roughness: 0.2,
            ..Default::default()
        };
        let mat2 = mat1.clone();

        let handle1 = 1;
        let handle2 = 2;

        let result_h1 = registry.get_or_register(handle1, mat1);
        let result_h2 = registry.get_or_register(handle2, mat2);

        assert_eq!(result_h1, handle1);
        assert_eq!(result_h2, handle1); // Should reuse handle1
        assert_eq!(registry.materials.len(), 1);
    }

    #[test]
    fn test_quantization_variance() {
        let mut registry = MaterialRegistry::new();
        let mat1 = Material {
            color: [0.5, 0.5, 0.5, 1.0],
            ..Default::default()
        };
        // Very minor difference that should still be quantized to same value (0.5001 -> 0.5)
        let mat2 = Material {
            color: [0.5001, 0.5, 0.5, 1.0],
            ..Default::default()
        };

        let h1 = registry.get_or_register(1, mat1);
        let h2 = registry.get_or_register(2, mat2);

        assert_eq!(h1, h2);
    }
}
