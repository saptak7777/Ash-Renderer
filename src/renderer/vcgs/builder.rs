//! # VCGS (Virtual Clustered Geometry System) Builder
//!
//! This module provides the logic for converting standard [`Mesh`] data into a
//! Virtual Clustered Geometry DAG. It handles:
//! - Mesh simplification and clustering via `meshopt`.
//! - Generation of cluster hierarchies for continuous LOD.
//! - Building the GPU-ready DAG structure.
//!
//! The primary entry point is [`build_mesh_dag`].

use crate::renderer::resources::mesh::Mesh;
use crate::renderer::resources::mesh::MeshCluster;
use bytemuck;
use rayon::prelude::*;

/// Builds a cluster DAG for a mesh using meshopt (VCGS V2).
pub fn build_mesh_dag(mesh: &mut Mesh) {
    if mesh.indices.is_none() {
        return;
    }
    // Base indices (Level 0 input)
    let indices = mesh.indices.as_ref().unwrap().clone();
    let vertices = &mesh.vertices;

    log::info!(
        "Building VCGS V2 DAG for '{}' ({} tris)...",
        mesh.name,
        indices.len() / 3
    );

    // Prepare position buffer for meshopt (flattened f32)
    let positions: Vec<f32> = vertices.iter().flat_map(|v| v.position).collect();

    // ------------------------------------------------------------------------
    // Step 1: Generate Leaf Meshlets (Level 0)
    // ------------------------------------------------------------------------
    let max_vertices = 64;
    let max_triangles = 124;
    let cone_weight = 0.0;

    // Using meshopt crate (0.6.2)
    // Wrap vertices in adapter (stride = 12 bytes = 3 floats)
    let positions_bytes: &[u8] = bytemuck::cast_slice(&positions);
    let vertex_adapter = meshopt::VertexDataAdapter::new(positions_bytes, 12, 0).unwrap();
    let meshlets = meshopt::build_meshlets(
        &indices,
        &vertex_adapter,
        max_vertices,
        max_triangles,
        cone_weight,
    );

    let mut clusters = Vec::with_capacity(meshlets.len() * 2); // Reserve space for parents
    let mut new_global_indices = Vec::with_capacity(indices.len() * 2); // Global index buffer

    // Helper to process meshlets result and append to global state
    fn append_meshlets(
        meshlets: &meshopt::Meshlets,
        original_vertices: &[crate::renderer::Vertex],
        clusters: &mut Vec<MeshCluster>,
        global_indices: &mut Vec<u32>,
        error_metric: f32,
    ) -> std::ops::Range<usize> {
        let start_cluster_idx = clusters.len();

        for meshlet in &meshlets.meshlets {
            let first_index = global_indices.len() as u32;

            // Reconstruct triangles
            for i in 0..meshlet.triangle_count {
                let t_idx = meshlet.triangle_offset as usize + i as usize;
                let v0_local = meshlets.triangles[t_idx * 3 + 0];
                let v1_local = meshlets.triangles[t_idx * 3 + 1];
                let v2_local = meshlets.triangles[t_idx * 3 + 2];

                let v0 = meshlets.vertices[meshlet.vertex_offset as usize + v0_local as usize];
                let v1 = meshlets.vertices[meshlet.vertex_offset as usize + v1_local as usize];
                let v2 = meshlets.vertices[meshlet.vertex_offset as usize + v2_local as usize];

                global_indices.push(v0);
                global_indices.push(v1);
                global_indices.push(v2);
            }

            // Compute bounds
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];

            for i in 0..meshlet.vertex_count {
                let v_idx = meshlets.vertices[meshlet.vertex_offset as usize + i as usize];
                let v = original_vertices[v_idx as usize];
                for k in 0..3 {
                    min[k] = min[k].min(v.position[k]);
                    max[k] = max[k].max(v.position[k]);
                }
            }

            let center = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];

            let mut radius_sq = 0.0f32;
            for i in 0..meshlet.vertex_count {
                // Bounds Check: Prevent panic on bad data
                if (meshlet.vertex_offset as usize + i as usize) >= meshlets.vertices.len() {
                    continue;
                }

                let v_idx = meshlets.vertices[meshlet.vertex_offset as usize + i as usize];

                if (v_idx as usize) >= original_vertices.len() {
                    // Start of panic prevention
                    continue;
                }

                let v = original_vertices[v_idx as usize];
                let d = [
                    v.position[0] - center[0],
                    v.position[1] - center[1],
                    v.position[2] - center[2],
                ];
                radius_sq = radius_sq.max(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            }

            clusters.push(MeshCluster {
                bounds_center: center,
                error_metric,
                bounds_radius: radius_sq.sqrt(),
                parent_index: u32::MAX,
                first_index: first_index as u32,
                index_count: meshlet.triangle_count as u32 * 3,
            });
        }

        start_cluster_idx..clusters.len()
    }

    // Initial Level 0 (Error 0.0)
    let mut current_range = append_meshlets(
        &meshlets,
        vertices,
        &mut clusters,
        &mut new_global_indices,
        0.0,
    );
    let mut level = 0;

    // ------------------------------------------------------------------------
    // Step 2: DAG Reduction Loop
    // ------------------------------------------------------------------------
    while current_range.len() > 1 {
        level += 1;
        let start_idx = current_range.start;
        let end_idx = current_range.end;

        // A. SPATIAL SORT (Morton Codes) - Parallelized
        let (min, max) = compute_cluster_bounds(&clusters[start_idx..end_idx]);
        let range_x = (max[0] - min[0]).max(0.001);
        let range_y = (max[1] - min[1]).max(0.001);
        let range_z = (max[2] - min[2]).max(0.001);

        // Use Rayon for parallel Morton Code calculation
        let mut sortable: Vec<(u32, usize)> = (start_idx..end_idx)
            .into_par_iter()
            .map(|i| {
                let c = &clusters[i];
                let nx = (c.bounds_center[0] - min[0]) / range_x;
                let ny = (c.bounds_center[1] - min[1]) / range_y;
                let nz = (c.bounds_center[2] - min[2]) / range_z;
                (morton_3d(nx, ny, nz), i)
            })
            .collect();

        // Parallel Sort
        sortable.par_sort_by_key(|k| k.0);

        // B. GROUP & MERGE & SIMPLIFY
        let next_level_start = clusters.len();

        for chunk in sortable.chunks(4) {
            let mut merged_indices = Vec::new();
            for &(_, cluster_idx) in chunk {
                let c = &clusters[cluster_idx];
                let range = c.first_index as usize..(c.first_index + c.index_count) as usize;
                merged_indices.extend_from_slice(&new_global_indices[range]);
            }

            // C. SIMPLIFY
            // V2 Metrics: Exponential error growth (Geometric Error)
            let target_error = 1e-2f32 * 2.0f32.powi(level - 1);
            let target_count = 124 * 3;

            let positions_bytes: &[u8] = bytemuck::cast_slice(&positions);
            let vertex_adapter = meshopt::VertexDataAdapter::new(positions_bytes, 12, 0).unwrap();

            let options = meshopt::SimplifyOptions::LockBorder;

            // Signature: simplify(indices, vertices, target_count, target_error, options, result_error)
            let simplified = meshopt::simplify(
                &merged_indices,
                &vertex_adapter,
                target_count,
                target_error,
                options,
                None, // result_error
            );

            if simplified.is_empty() {
                continue;
            }

            // D. RE-CLUSTER (Build Parent Meshlets)
            let positions_bytes: &[u8] = bytemuck::cast_slice(&positions);
            let vertex_adapter = meshopt::VertexDataAdapter::new(positions_bytes, 12, 0).unwrap();
            let parent_meshlets = meshopt::build_meshlets(
                &simplified,
                &vertex_adapter,
                max_vertices,
                max_triangles,
                cone_weight,
            );

            // E. APPEND & LINK
            let parent_range = append_meshlets(
                &parent_meshlets,
                vertices,
                &mut clusters,
                &mut new_global_indices,
                target_error,
            );

            if !parent_range.is_empty() {
                let primary_parent = parent_range.start as u32;

                for &(_, child_idx) in chunk {
                    clusters[child_idx].parent_index = primary_parent;
                }
            }
        }

        let next_level_end = clusters.len();

        if next_level_end == next_level_start {
            break;
        }

        current_range = next_level_start..next_level_end;
    }

    // Replace indices and clusters
    mesh.indices = Some(new_global_indices);
    mesh.clusters = clusters;

    log::info!(
        "DAG V2 Build Complete: {} Total Clusters, {} Levels",
        mesh.clusters.len(),
        level
    );
}

// ----------------------------------------------------------------------------
// Spacial Sorting Helpers (Morton Codes)
// ----------------------------------------------------------------------------

fn expand_bits(mut v: u32) -> u32 {
    v = (v * 0x00010001) & 0xFF0000FF;
    v = (v * 0x00000101) & 0x0F00F00F;
    v = (v * 0x00000011) & 0xC30C30C3;
    v = (v * 0x00000005) & 0x49249249;
    v
}

fn morton_3d(x: f32, y: f32, z: f32) -> u32 {
    let x = (x * 1023.0).clamp(0.0, 1023.0) as u32;
    let y = (y * 1023.0).clamp(0.0, 1023.0) as u32;
    let z = (z * 1023.0).clamp(0.0, 1023.0) as u32;

    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

fn compute_cluster_bounds(clusters: &[MeshCluster]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for c in clusters {
        let r = c.bounds_radius;
        for k in 0..3 {
            min[k] = min[k].min(c.bounds_center[k] - r);
            max[k] = max[k].max(c.bounds_center[k] + r);
        }
    }

    (min, max)
}
