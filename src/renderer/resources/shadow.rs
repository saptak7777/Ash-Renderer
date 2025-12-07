//! Cascaded Shadow Maps (CSM) for directional lights
//!
//! Provides high-quality shadows at all distances by splitting the view frustum
//! into multiple cascades, each with its own shadow map resolution.
//!
//! # Architecture
//! - 4 cascades by default (configurable)
//! - Logarithmic/practical split scheme
//! - Per-cascade light-space matrices
//! - Cascade blending for smooth transitions

use glam::{Mat4, Vec3, Vec4};

/// Maximum number of cascades
pub const MAX_CASCADES: usize = 4;

/// CSM configuration
#[derive(Debug, Clone)]
pub struct CsmConfig {
    /// Number of cascades (1-4)
    pub cascade_count: u32,
    /// Shadow map resolution per cascade
    pub resolution: u32,
    /// Lambda for split calculation (0 = linear, 1 = logarithmic, 0.5 = practical)
    pub split_lambda: f32,
    /// Depth bias per cascade (multiplied by cascade index)
    pub depth_bias: f32,
    /// Enable cascade blending
    pub blend_cascades: bool,
    /// Blend region size (in NDC)
    pub blend_size: f32,
    /// Enable shadows
    pub enabled: bool,
}

impl Default for CsmConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            resolution: 2048,
            split_lambda: 0.75, // Practical split
            depth_bias: 0.0005,
            blend_cascades: true,
            blend_size: 0.1,
            enabled: true,
        }
    }
}

impl CsmConfig {
    /// High quality preset
    pub fn high_quality() -> Self {
        Self {
            cascade_count: 4,
            resolution: 4096,
            split_lambda: 0.8,
            depth_bias: 0.0003,
            blend_cascades: true,
            blend_size: 0.15,
            enabled: true,
        }
    }

    /// Performance preset
    pub fn performance() -> Self {
        Self {
            cascade_count: 2,
            resolution: 1024,
            split_lambda: 0.5,
            depth_bias: 0.001,
            blend_cascades: false,
            blend_size: 0.0,
            enabled: true,
        }
    }
}

/// Cascade split distances and matrices
#[derive(Debug, Clone, Copy, Default)]
pub struct CascadeData {
    /// Near split distance in view space
    pub near: f32,
    /// Far split distance in view space
    pub far: f32,
    /// Light-space matrix for this cascade
    pub light_space_matrix: Mat4,
}

/// GPU-ready cascade data (matches shader layout)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCascadeData {
    /// Light-space matrices [4]
    pub light_matrices: [[f32; 16]; MAX_CASCADES],
    /// Split distances (x, y, z, w = cascade 0-3 far planes)
    pub split_distances: [f32; 4],
    /// Cascade count, blend enabled, blend size, padding
    pub params: [f32; 4],
}

impl Default for GpuCascadeData {
    fn default() -> Self {
        Self {
            light_matrices: [[0.0; 16]; MAX_CASCADES],
            split_distances: [0.0; 4],
            params: [4.0, 1.0, 0.1, 0.0],
        }
    }
}

/// Cascaded Shadow Map manager
pub struct CascadedShadowMap {
    /// Configuration
    config: CsmConfig,
    /// Per-cascade data
    cascades: [CascadeData; MAX_CASCADES],
    /// Cached GPU data
    gpu_data: GpuCascadeData,
}

impl CascadedShadowMap {
    /// Create a new CSM manager
    pub fn new(config: CsmConfig) -> Self {
        Self {
            config,
            cascades: [CascadeData::default(); MAX_CASCADES],
            gpu_data: GpuCascadeData::default(),
        }
    }

    /// Calculate cascade splits using the practical split scheme
    ///
    /// # Arguments
    /// * `near` - Camera near plane
    /// * `far` - Camera far plane (or max shadow distance)
    fn calculate_splits(&self, near: f32, far: f32) -> [f32; MAX_CASCADES + 1] {
        let mut splits = [0.0f32; MAX_CASCADES + 1];
        let count = self.config.cascade_count as usize;
        let lambda = self.config.split_lambda;

        splits[0] = near;

        for (i, split) in splits.iter_mut().enumerate().skip(1).take(count) {
            let p = i as f32 / count as f32;

            // Logarithmic split
            let log_split = near * (far / near).powf(p);

            // Linear split
            let linear_split = near + (far - near) * p;

            // Practical split (blend between log and linear)
            *split = lambda * log_split + (1.0 - lambda) * linear_split;
        }

        // Fill remaining splits with far plane
        for split in splits.iter_mut().skip(count + 1) {
            *split = far;
        }

        splits
    }

    /// Calculate frustum corners in world space for a split
    fn calculate_frustum_corners(
        view: &Mat4,
        proj: &Mat4,
        near_split: f32,
        far_split: f32,
    ) -> [Vec3; 8] {
        let inv_vp = (*proj * *view).inverse();

        // Scale near/far based on projection
        let proj_near = proj.w_axis.z / (proj.z_axis.z - 1.0);
        let proj_far = proj.w_axis.z / (proj.z_axis.z + 1.0);

        let near_ndc = (near_split - proj_near) / (proj_far - proj_near) * 2.0 - 1.0;
        let far_ndc = (far_split - proj_near) / (proj_far - proj_near) * 2.0 - 1.0;

        let corners_ndc = [
            // Near plane
            Vec4::new(-1.0, -1.0, near_ndc.max(-1.0), 1.0),
            Vec4::new(1.0, -1.0, near_ndc.max(-1.0), 1.0),
            Vec4::new(1.0, 1.0, near_ndc.max(-1.0), 1.0),
            Vec4::new(-1.0, 1.0, near_ndc.max(-1.0), 1.0),
            // Far plane
            Vec4::new(-1.0, -1.0, far_ndc.min(1.0), 1.0),
            Vec4::new(1.0, -1.0, far_ndc.min(1.0), 1.0),
            Vec4::new(1.0, 1.0, far_ndc.min(1.0), 1.0),
            Vec4::new(-1.0, 1.0, far_ndc.min(1.0), 1.0),
        ];

        let mut corners = [Vec3::ZERO; 8];
        for (i, ndc) in corners_ndc.iter().enumerate() {
            let world = inv_vp * *ndc;
            corners[i] = world.truncate() / world.w;
        }

        corners
    }

    /// Update cascade matrices for current camera
    ///
    /// # Arguments
    /// * `camera_view` - Camera view matrix
    /// * `camera_proj` - Camera projection matrix
    /// * `light_dir` - Normalized light direction (pointing toward light)
    /// * `shadow_distance` - Maximum shadow rendering distance
    pub fn update(
        &mut self,
        camera_view: &Mat4,
        camera_proj: &Mat4,
        light_dir: Vec3,
        shadow_distance: f32,
    ) {
        let near = 0.1; // Camera near plane
        let far = shadow_distance;

        let splits = self.calculate_splits(near, far);
        let count = self.config.cascade_count as usize;

        for i in 0..count {
            let cascade_near = splits[i];
            let cascade_far = splits[i + 1];

            // Get frustum corners for this cascade
            let corners = Self::calculate_frustum_corners(
                camera_view,
                camera_proj,
                cascade_near,
                cascade_far,
            );

            // Calculate frustum center
            let center: Vec3 = corners.iter().copied().sum::<Vec3>() / 8.0;

            // Calculate bounding sphere radius
            let mut max_dist = 0.0f32;
            for corner in &corners {
                max_dist = max_dist.max((*corner - center).length());
            }

            // Round to texel size for stability
            let texels_per_unit = self.config.resolution as f32 / (max_dist * 2.0);
            let max_dist = (max_dist * texels_per_unit).ceil() / texels_per_unit;

            // Light view matrix (looking along light direction)
            let light_pos = center - light_dir.normalize() * max_dist;
            let light_view = Mat4::look_at_rh(light_pos, center, Vec3::Y);

            // Orthographic projection sized to fit the cascade
            let light_proj = Mat4::orthographic_rh(
                -max_dist,
                max_dist,
                -max_dist,
                max_dist,
                0.0,
                max_dist * 2.0,
            );

            self.cascades[i] = CascadeData {
                near: cascade_near,
                far: cascade_far,
                light_space_matrix: light_proj * light_view,
            };
        }

        // Update GPU data
        self.update_gpu_data();
    }

    /// Update cached GPU data
    fn update_gpu_data(&mut self) {
        let count = self.config.cascade_count as usize;

        for i in 0..MAX_CASCADES {
            if i < count {
                let mat = self.cascades[i].light_space_matrix;
                self.gpu_data.light_matrices[i] = mat.to_cols_array();
                self.gpu_data.split_distances[i] = self.cascades[i].far;
            } else {
                self.gpu_data.light_matrices[i] = [0.0; 16];
                self.gpu_data.split_distances[i] = f32::MAX;
            }
        }

        self.gpu_data.params = [
            self.config.cascade_count as f32,
            if self.config.blend_cascades { 1.0 } else { 0.0 },
            self.config.blend_size,
            0.0,
        ];
    }

    /// Get GPU-ready cascade data
    pub fn gpu_data(&self) -> &GpuCascadeData {
        &self.gpu_data
    }

    /// Get cascade data by index
    pub fn cascade(&self, index: usize) -> Option<&CascadeData> {
        if index < self.config.cascade_count as usize {
            Some(&self.cascades[index])
        } else {
            None
        }
    }

    /// Get number of active cascades
    pub fn cascade_count(&self) -> usize {
        self.config.cascade_count as usize
    }

    /// Get config
    pub fn config(&self) -> &CsmConfig {
        &self.config
    }

    /// Get mutable config
    pub fn config_mut(&mut self) -> &mut CsmConfig {
        &mut self.config
    }

    /// Get light-space matrix for a cascade
    pub fn light_matrix(&self, cascade: usize) -> Mat4 {
        self.cascades
            .get(cascade)
            .map(|c| c.light_space_matrix)
            .unwrap_or(Mat4::IDENTITY)
    }

    /// Get all cascade matrices as a slice
    pub fn light_matrices(&self) -> &[CascadeData; MAX_CASCADES] {
        &self.cascades
    }
}

impl Default for CascadedShadowMap {
    fn default() -> Self {
        Self::new(CsmConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_calculation() {
        let csm = CascadedShadowMap::new(CsmConfig::default());
        let splits = csm.calculate_splits(0.1, 100.0);

        assert_eq!(splits[0], 0.1);
        assert!(splits[1] < splits[2]);
        assert!(splits[2] < splits[3]);
        assert!(splits[3] < splits[4]);
        assert!(splits[4] <= 100.0);
    }

    #[test]
    fn test_cascade_update() {
        let mut csm = CascadedShadowMap::new(CsmConfig::default());

        let view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let light_dir = Vec3::new(-0.5, -1.0, -0.3).normalize();

        csm.update(&view, &proj, light_dir, 50.0);

        // Check matrices are valid
        for i in 0..4 {
            let mat = csm.light_matrix(i);
            assert_ne!(mat, Mat4::IDENTITY);
        }

        // Check GPU data
        let gpu = csm.gpu_data();
        assert_eq!(gpu.params[0], 4.0); // 4 cascades
    }

    #[test]
    fn test_config_presets() {
        let hq = CsmConfig::high_quality();
        assert_eq!(hq.resolution, 4096);

        let perf = CsmConfig::performance();
        assert_eq!(perf.cascade_count, 2);
    }
}
