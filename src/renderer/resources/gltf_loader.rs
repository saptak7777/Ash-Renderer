use crate::renderer::resources::mesh::{MaterialProperties, Mesh};
use crate::renderer::resources::Vertex as RendererVertex;
use crate::Result;
use archetype_asset::ModelLoader;
use std::path::Path;

/// Utility to load GLTF models and bridge them to the renderer's data structures.
/// This fulfills the "Dumb Pipe" philosophy by keeping loading logic out of the core renderer
/// while still providing a convenient helper for applications and examples.
#[cfg(feature = "gltf_loading")]
#[allow(clippy::field_reassign_with_default)]
pub fn load_model(path: impl AsRef<Path>) -> Result<Vec<Mesh>> {
    let path = path.as_ref();
    let glb_data = std::fs::read(path)
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to read GLB: {e}")))?;

    let loader = ModelLoader::new();

    // We use block_on here because the examples are currently synchronous.
    // In a real async application, you would use the loader directly.
    let model = futures::executor::block_on(loader.load_gltf_optimized(&glb_data))
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to load GLTF: {e}")))?;

    let mut meshes = Vec::new();

    for (i, mesh_asset) in model.meshes.into_iter().enumerate() {
        // Map archetype_asset vertices (16 floats) to ash_renderer vertices (15 floats)
        let renderer_vertices: Vec<RendererVertex> = mesh_asset
            .vertices()
            .vertices
            .chunks_exact(16)
            .map(|v| {
                RendererVertex {
                    position: [v[0], v[1], v[2]],
                    normal: [v[3], v[4], v[5]],
                    uv: [v[6], v[7]],
                    color: [v[12], v[13], v[14]], // RGB from RGBA
                    tangent: [v[8], v[9], v[10], v[11]],
                }
            })
            .collect();

        let mut mesh = Mesh::default();
        mesh.name = mesh_asset
            .name
            .clone()
            .unwrap_or_else(|| format!("Mesh_{i}"))
            .into();
        mesh.vertices = renderer_vertices;
        mesh.indices = Some(mesh_asset.vertices().indices.clone());

        // Map material properties
        if let Some(mat_idx) = mesh_asset.material_index {
            if mat_idx < model.materials.len() {
                let mat_asset = &model.materials[mat_idx];
                mesh.material_properties = Some(MaterialProperties {
                    base_color_factor: mat_asset.base_color_factor,
                    metallic_factor: mat_asset.metallic_factor,
                    roughness_factor: mat_asset.roughness_factor,
                    emissive_factor: [
                        mat_asset.emissive_factor[0],
                        mat_asset.emissive_factor[1],
                        mat_asset.emissive_factor[2],
                        1.0,
                    ],
                    occlusion_strength: mat_asset.occlusion_strength,
                    normal_scale: mat_asset.normal_scale,
                    alpha_cutoff: mat_asset.alpha_cutoff,
                });
            }
        }

        meshes.push(mesh);
    }

    Ok(meshes)
}
