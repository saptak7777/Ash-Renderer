//! GLTF Model Loading
//!
//! Provides utilities to load GLTF/GLB models directly using the `gltf` crate.
//! This module follows the "Dumb Pipe" philosophy by keeping loading logic
//! separate from the core renderer.

use crate::renderer::resources::mesh::{MaterialProperties, Mesh, Vertex};
use crate::renderer::resources::texture::TextureData;
use crate::{AshError, Result};
use std::path::Path;

/// Loads a GLTF or GLB model from the filesystem.
///
/// Returns a vector of meshes with embedded material properties and texture data.
/// Each mesh corresponds to a primitive in the GLTF file.
///
/// # Errors
/// Returns `AshError` if:
/// - File cannot be read
/// - GLTF parsing fails
/// - Buffer data is missing or invalid
/// - Accessor data is malformed
#[cfg(feature = "gltf_loading")]
pub fn load_model(path: impl AsRef<Path>) -> Result<Vec<Mesh>> {
    let path = path.as_ref();

    // Load GLTF document
    let (document, buffers, images) = gltf::import(path)
        .map_err(|e| AshError::VulkanError(format!("Failed to load GLTF: {e}")))?;

    let mut meshes = Vec::new();

    for (mesh_idx, gltf_mesh) in document.meshes().enumerate() {
        for (prim_idx, primitive) in gltf_mesh.primitives().enumerate() {
            let reader =
                primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &data[..]));

            // Extract vertex data
            let positions = reader
                .read_positions()
                .ok_or_else(|| AshError::VulkanError("Missing position attribute".into()))?
                .collect::<Vec<_>>();

            let normals = reader
                .read_normals()
                .map(|iter| iter.collect::<Vec<_>>())
                .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);

            let uvs = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect::<Vec<_>>())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let colors = reader
                .read_colors(0)
                .map(|iter| iter.into_rgb_f32().collect::<Vec<_>>())
                .unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);

            let tangents = reader
                .read_tangents()
                .map(|iter| iter.collect::<Vec<_>>())
                // Industry standard: Use [1,0,0,1] as default tangent if missing.
                // This ensures unit length and common perpendicularity to [0,0,1] normals.
                // Preventing degenerate TBN matrices while avoiding heavy CPU re-calculation.
                .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]);

            // Build vertices
            let vertices: Vec<Vertex> = (0..positions.len())
                .map(|i| Vertex {
                    position: positions[i],
                    normal: normals[i],
                    uv: uvs[i],
                    color: colors[i],
                    tangent: tangents[i],
                    _padding: 0,
                })
                .collect();

            // Extract indices
            let indices = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect::<Vec<_>>());

            // Extract material properties
            let material_properties = primitive.material().index().and_then(|mat_idx| {
                document.materials().nth(mat_idx).map(|mat| {
                    let pbr = mat.pbr_metallic_roughness();
                    MaterialProperties {
                        base_color_factor: pbr.base_color_factor(),
                        metallic_factor: pbr.metallic_factor(),
                        roughness_factor: pbr.roughness_factor(),
                        emissive_factor: {
                            let e = mat.emissive_factor();
                            [e[0], e[1], e[2], 1.0]
                        },
                        occlusion_strength: mat
                            .occlusion_texture()
                            .map(|t| t.strength())
                            .unwrap_or(1.0),
                        normal_scale: mat.normal_texture().map(|t| t.scale()).unwrap_or(1.0),
                        alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
                        flags: if mat.alpha_mode() != gltf::material::AlphaMode::Opaque {
                            crate::renderer::resources::uniform::MATERIAL_FLAG_ALPHA_TESTED
                        } else {
                            0
                        },
                    }
                })
            });

            // Load textures
            let base_color_texture = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .and_then(|info| load_texture_data(&images, info.texture().source().index()));

            let normal_texture = primitive
                .material()
                .normal_texture()
                .and_then(|info| load_texture_data(&images, info.texture().source().index()));

            let metallic_roughness_texture = primitive
                .material()
                .pbr_metallic_roughness()
                .metallic_roughness_texture()
                .and_then(|info| load_texture_data(&images, info.texture().source().index()));

            let occlusion_texture = primitive
                .material()
                .occlusion_texture()
                .and_then(|info| load_texture_data(&images, info.texture().source().index()));

            let emissive_texture = primitive
                .material()
                .emissive_texture()
                .and_then(|info| load_texture_data(&images, info.texture().source().index()));

            // Create mesh
            let mesh_name = gltf_mesh
                .name()
                .map(|n| format!("{n}_prim{prim_idx}"))
                .unwrap_or_else(|| format!("Mesh{mesh_idx}_prim{prim_idx}"));

            let mesh = Mesh {
                name: mesh_name.into(),
                vertices,
                indices,
                texture_data: base_color_texture,
                normal_texture_data: normal_texture,
                metallic_roughness_texture_data: metallic_roughness_texture,
                occlusion_texture_data: occlusion_texture,
                emissive_texture_data: emissive_texture,
                material_properties,
                ..Default::default()
            };

            meshes.push(mesh);
        }
    }

    log::info!("Loaded {count} meshes from {path:?}", count = meshes.len());
    Ok(meshes)
}

/// Converts a GLTF image to TextureData (RGBA8).
fn load_texture_data(images: &[gltf::image::Data], index: usize) -> Option<TextureData> {
    let image = images.get(index)?;

    // Convert to RGBA8 if needed
    let pixels = match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            // Convert RGB to RGBA
            image
                .pixels
                .chunks_exact(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                .collect()
        }
        gltf::image::Format::R8G8 => {
            // Convert RG to RGBA (useful for normal maps)
            image
                .pixels
                .chunks_exact(2)
                .flat_map(|rg| [rg[0], rg[1], 0, 255])
                .collect()
        }
        gltf::image::Format::R8 => {
            // Convert R to RGBA
            image.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect()
        }
        _ => {
            log::warn!(
                "Unsupported texture format: {format:?}",
                format = image.format
            );
            return None;
        }
    };

    TextureData::new(image.width, image.height, pixels).ok()
}

#[cfg(all(test, feature = "gltf_loading"))]
mod tests {
    use super::*;

    #[test]
    fn test_load_texture_data_rgba() {
        let image = gltf::image::Data {
            pixels: vec![255, 0, 0, 255, 0, 255, 0, 255],
            format: gltf::image::Format::R8G8B8A8,
            width: 2,
            height: 1,
        };
        let images = vec![image];

        let texture = load_texture_data(&images, 0).expect("Should load texture");
        assert_eq!(texture.width, 2);
        assert_eq!(texture.height, 1);
        assert_eq!(texture.pixels.len(), 8);
    }

    #[test]
    fn test_load_texture_data_rgb() {
        let image = gltf::image::Data {
            pixels: vec![255, 0, 0, 0, 255, 0],
            format: gltf::image::Format::R8G8B8,
            width: 2,
            height: 1,
        };
        let images = vec![image];

        let texture = load_texture_data(&images, 0).expect("Should load texture");
        assert_eq!(texture.width, 2);
        assert_eq!(texture.height, 1);
        assert_eq!(texture.pixels.len(), 8); // Converted to RGBA
        assert_eq!(texture.pixels[3], 255); // Alpha should be 255
    }
}
