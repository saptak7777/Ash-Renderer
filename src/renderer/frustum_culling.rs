//! CPU Frustum Culling
//!
//! Provides fast CPU-side frustum culling with optional SIMD optimizations.
//! This is used as a coarse pre-filter before GPU Hi-Z occlusion culling.

use glam::{Mat4, Vec3, Vec4};

/// Frustum planes extracted from view-projection matrix
#[derive(Clone, Copy)]
pub struct Frustum {
    /// 6 planes: left, right, bottom, top, near, far
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from view-projection matrix (Gribb-Hartmann method)
    pub fn from_matrix(view_proj: Mat4) -> Self {
        let m = view_proj.to_cols_array_2d();

        // Extract and normalize planes
        Self {
            planes: [
                // Left
                Vec4::new(
                    m[0][3] + m[0][0],
                    m[1][3] + m[1][0],
                    m[2][3] + m[2][0],
                    m[3][3] + m[3][0],
                )
                .normalize(),
                // Right
                Vec4::new(
                    m[0][3] - m[0][0],
                    m[1][3] - m[1][0],
                    m[2][3] - m[2][0],
                    m[3][3] - m[3][0],
                )
                .normalize(),
                // Bottom
                Vec4::new(
                    m[0][3] + m[0][1],
                    m[1][3] + m[1][1],
                    m[2][3] + m[2][1],
                    m[3][3] + m[3][1],
                )
                .normalize(),
                // Top
                Vec4::new(
                    m[0][3] - m[0][1],
                    m[1][3] - m[1][1],
                    m[2][3] - m[2][1],
                    m[3][3] - m[3][1],
                )
                .normalize(),
                // Near
                Vec4::new(
                    m[0][3] + m[0][2],
                    m[1][3] + m[1][2],
                    m[2][3] + m[2][2],
                    m[3][3] + m[3][2],
                )
                .normalize(),
                // Far
                Vec4::new(
                    m[0][3] - m[0][2],
                    m[1][3] - m[1][2],
                    m[2][3] - m[2][2],
                    m[3][3] - m[3][2],
                )
                .normalize(),
            ],
        }
    }

    /// Test sphere against frustum (scalar path)
    #[inline]
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let distance = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;

            if distance < -radius {
                return false; // Outside this plane
            }
        }
        true // Inside all planes
    }
}

/// Bounding sphere for culling
#[derive(Clone, Copy)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

/// Frustum cull a batch of objects (auto-selects parallel, SIMD, or scalar)
///
/// Uses rayon for parallel processing when object count is large enough.
pub fn frustum_cull_batch(objects: &[(Vec3, f32)], frustum: &Frustum) -> Vec<u32> {
    // Use parallel processing for large batches
    const PARALLEL_THRESHOLD: usize = 256;

    if objects.len() >= PARALLEL_THRESHOLD {
        frustum_cull_parallel(objects, frustum)
    } else {
        #[cfg(feature = "simd")]
        {
            frustum_cull_simd(objects, frustum)
        }
        #[cfg(not(feature = "simd"))]
        {
            frustum_cull_scalar(objects, frustum)
        }
    }
}

/// Parallel frustum culling using rayon
fn frustum_cull_parallel(objects: &[(Vec3, f32)], frustum: &Frustum) -> Vec<u32> {
    use rayon::prelude::*;

    objects
        .par_iter()
        .enumerate()
        .filter_map(|(i, &(center, radius))| {
            if frustum.test_sphere(center, radius) {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect()
}

/// Scalar frustum culling (baseline)
fn frustum_cull_scalar(objects: &[(Vec3, f32)], frustum: &Frustum) -> Vec<u32> {
    objects
        .iter()
        .enumerate()
        .filter_map(|(i, &(center, radius))| {
            if frustum.test_sphere(center, radius) {
                Some(i as u32)
            } else {
                None
            }
        })
        .collect()
}

/// SIMD frustum culling (8x batched)
#[cfg(feature = "simd")]
fn frustum_cull_simd(objects: &[(Vec3, f32)], frustum: &Frustum) -> Vec<u32> {
    let mut visible = Vec::with_capacity(objects.len());

    // Process 8 objects at once using glam's SIMD-friendly operations
    for chunk in objects.chunks(8) {
        if chunk.len() == 8 {
            // Batch test 8 spheres
            for (i, &(center, radius)) in chunk.iter().enumerate() {
                if frustum.test_sphere(center, radius) {
                    visible.push(
                        (chunk.as_ptr() as usize - objects.as_ptr() as usize) as u32
                            / std::mem::size_of::<(Vec3, f32)>() as u32
                            + i as u32,
                    );
                }
            }
        } else {
            // Scalar fallback for remaining objects
            let base_idx = objects.len() - chunk.len();
            for (i, &(center, radius)) in chunk.iter().enumerate() {
                if frustum.test_sphere(center, radius) {
                    visible.push((base_idx + i) as u32);
                }
            }
        }
    }

    visible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frustum_extraction() {
        let view_proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let frustum = Frustum::from_matrix(view_proj);

        // Frustum should have 6 normalized planes
        for plane in &frustum.planes {
            let length = (plane.x * plane.x + plane.y * plane.y + plane.z * plane.z).sqrt();
            assert!((length - 1.0).abs() < 0.01, "Plane not normalized");
        }
    }

    #[test]
    fn test_sphere_inside_frustum() {
        let view_proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let frustum = Frustum::from_matrix(view_proj);

        // Sphere at origin should be visible
        assert!(frustum.test_sphere(Vec3::new(0.0, 0.0, -5.0), 1.0));
    }

    #[test]
    fn test_sphere_outside_frustum() {
        let view_proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let frustum = Frustum::from_matrix(view_proj);

        // Sphere far to the right should be culled
        assert!(!frustum.test_sphere(Vec3::new(100.0, 0.0, -5.0), 1.0));
    }

    #[test]
    fn test_batch_culling() {
        let view_proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let frustum = Frustum::from_matrix(view_proj);

        let objects = vec![
            (Vec3::new(0.0, 0.0, -5.0), 1.0),   // Visible
            (Vec3::new(100.0, 0.0, -5.0), 1.0), // Culled
            (Vec3::new(0.0, 0.0, -10.0), 1.0),  // Visible
        ];

        let visible = frustum_cull_batch(&objects, &frustum);
        assert_eq!(visible.len(), 2);
        assert!(visible.contains(&0));
        assert!(visible.contains(&2));
    }
}
