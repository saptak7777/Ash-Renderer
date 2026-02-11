use crate::renderer::resources::uniform::MaterialUniform;
use std::collections::{HashMap, HashSet};
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
    pub is_transparent: bool,
    // Texture indices (None = -1)
    pub texture_index: Option<u32>,
    pub normal_texture_index: Option<u32>,
    pub metallic_roughness_texture_index: Option<u32>,
    pub occlusion_texture_index: Option<u32>,
    pub emissive_texture_index: Option<u32>,
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
            is_transparent: false,
            texture_index: None,
            normal_texture_index: None,
            metallic_roughness_texture_index: None,
            occlusion_texture_index: None,
            emissive_texture_index: None,
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
            is_transparent: color[3] < 1.0,
            texture_index: None,
            normal_texture_index: None,
            metallic_roughness_texture_index: None,
            occlusion_texture_index: None,
            emissive_texture_index: None,
        }
    }

    /// Converts the material into its GPU-friendly uniform representation.
    pub fn to_uniform(&self) -> MaterialUniform {
        let mut mat_uniform = MaterialUniform::default();
        mat_uniform.set_base_color_factor(glam::Vec4::from_array(self.color));
        mat_uniform.set_emissive_factor(glam::Vec4::from_array(self.emissive));
        mat_uniform.set_metallic_roughness(self.metallic, self.roughness);
        mat_uniform.set_occlusion_strength(self.occlusion_strength);
        mat_uniform.set_normal_scale(self.normal_scale);
        mat_uniform.set_alpha_cutoff(self.alpha_cutoff);

        let base_idx = self.texture_index.unwrap_or(u32::MAX) as i32;
        let normal_idx = self.normal_texture_index.unwrap_or(u32::MAX) as i32;
        let mr_idx = self.metallic_roughness_texture_index.unwrap_or(u32::MAX) as i32;
        let occ_idx = self.occlusion_texture_index.unwrap_or(u32::MAX) as i32;
        let emissive_idx = self.emissive_texture_index.unwrap_or(u32::MAX) as i32;

        mat_uniform.set_texture_indices(
            base_idx,
            normal_idx,
            mr_idx,
            occ_idx,
            emissive_idx,
            self.tint_index,
        );
        mat_uniform
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
    is_transparent: bool,
    // Texture indices for unique identification
    texture_index: i32,
    normal_texture_index: i32,
    metallic_roughness_texture_index: i32,
    occlusion_texture_index: i32,
    emissive_texture_index: i32,
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
            is_transparent: material.is_transparent,
            texture_index: material.texture_index.map(|i| i as i32).unwrap_or(-1),
            normal_texture_index: material
                .normal_texture_index
                .map(|i| i as i32)
                .unwrap_or(-1),
            metallic_roughness_texture_index: material
                .metallic_roughness_texture_index
                .map(|i| i as i32)
                .unwrap_or(-1),
            occlusion_texture_index: material
                .occlusion_texture_index
                .map(|i| i as i32)
                .unwrap_or(-1),
            emissive_texture_index: material
                .emissive_texture_index
                .map(|i| i as i32)
                .unwrap_or(-1),
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
            is_transparent: props.base_color_factor[3] < 1.0,
            texture_index: -1,
            normal_texture_index: -1,
            metallic_roughness_texture_index: -1,
            occlusion_texture_index: -1,
            emissive_texture_index: -1,
        }
    }
}

use bytemuck::{Pod, Zeroable};

/// Material handle (Type-safe u32 wrapper for Bindless Indexing)
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, Pod, Zeroable)]
pub struct MaterialHandle {
    pub index: u32,
}

impl MaterialHandle {
    pub fn null() -> Self {
        Self { index: 0 }
    }

    pub fn is_null(&self) -> bool {
        self.index == 0
    }

    pub fn is_valid(&self, _manager: &MaterialManager) -> bool {
        // Simple index check for bindless (bounds check)
        self.index < 1024
    }

    pub fn get<'a>(&self, manager: &'a MaterialManager) -> Option<&'a Material> {
        if (self.index as usize) < manager.materials.len() {
            Some(&manager.materials[self.index as usize])
        } else {
            None
        }
    }
}

pub struct MaterialManager {
    materials: Vec<Material>,
    default_material: MaterialHandle,
    key_to_handle: HashMap<MaterialKey, MaterialHandle>,
}

impl MaterialManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn material_count(&self) -> u32 {
        self.materials.len() as u32
    }

    pub fn get_handle_by_name(&self, name: &str) -> Option<MaterialHandle> {
        self.materials
            .iter()
            .enumerate()
            .find(|(_, m)| m.name == name)
            .map(|(i, _)| MaterialHandle { index: i as u32 })
    }

    /// Returns an iterator over materials that have not been synced to the GPU yet.
    pub fn iter_unsynced<'a>(
        &'a self,
        uploaded_indices: &'a HashSet<u32>,
    ) -> impl Iterator<Item = (u32, &'a Material)> + 'a {
        self.materials
            .iter()
            .enumerate()
            .filter_map(move |(id, mat)| {
                let id = id as u32;
                if !uploaded_indices.contains(&id) {
                    Some((id, mat))
                } else {
                    None
                }
            })
    }

    pub fn register_material(&mut self, material: &Material) -> MaterialHandle {
        let key = MaterialKey::from_material(material);
        if let Some(&existing_handle) = self.key_to_handle.get(&key) {
            return existing_handle;
        }

        let index = self.materials.len() as u32;
        self.materials.push(material.clone());
        let handle = MaterialHandle { index };
        self.key_to_handle.insert(key, handle);

        log::debug!("Material registered: {} at slot {}", material.name, index);

        handle
    }

    pub fn get_default_material(&self) -> &Material {
        &self.materials[self.default_material.index as usize]
    }

    pub fn default_material(&self) -> MaterialHandle {
        self.default_material
    }

    pub fn get_material(&self, handle: MaterialHandle) -> &Material {
        if self.is_handle_valid(handle) {
            &self.materials[handle.index as usize]
        } else {
            &self.materials[self.default_material.index as usize]
        }
    }

    pub fn is_handle_valid(&self, handle: MaterialHandle) -> bool {
        (handle.index as usize) < self.materials.len()
    }

    pub fn get(&self, handle: MaterialHandle) -> Option<&Material> {
        handle.get(self)
    }

    pub fn get_mut(&mut self, handle: MaterialHandle) -> Option<&mut Material> {
        if self.is_handle_valid(handle) {
            Some(&mut self.materials[handle.index as usize])
        } else {
            None
        }
    }
}

impl Default for MaterialManager {
    fn default() -> Self {
        let mut manager = Self {
            materials: Vec::new(),
            default_material: MaterialHandle { index: 0 },
            key_to_handle: HashMap::new(),
        };

        // Register default material at index 0
        let default_mat = Material::default();
        manager.register_material(&default_mat);

        manager
    }
}

/// A centralized registry for materials with O(1) deduplication
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
        let mut manager = MaterialManager::new();
        let mat1 = Material {
            color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.5,
            roughness: 0.2,
            ..Default::default()
        };
        let mat2 = mat1.clone();

        let handle1 = manager.register_material(&mat1);
        let handle2 = manager.register_material(&mat2);

        assert_eq!(handle1, handle2); // Should reuse the same handle
                                      // Default material at 0, unique mat1 at 1. mat2 is deduplicated to 1.
        assert_eq!(manager.materials.len(), 2);
    }

    #[test]
    fn test_quantization_variance() {
        let mut manager = MaterialManager::new();
        let mat1 = Material {
            color: [0.5, 0.5, 0.5, 1.0],
            ..Default::default()
        };
        // Very minor difference that should still be quantized to same value (0.5001 -> 0.5)
        let mat2 = Material {
            color: [0.5001, 0.5, 0.5, 1.0],
            ..Default::default()
        };

        let h1 = manager.register_material(&mat1);
        let h2 = manager.register_material(&mat2);

        assert_eq!(h1, h2);
    }
}
