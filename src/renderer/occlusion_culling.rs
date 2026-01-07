//! GPU Occlusion Culling System
//!
//! Implements hierarchical-Z (Hi-Z) based occlusion culling using compute shaders.
//! Objects are tested against the Hi-Z pyramid to determine visibility before rendering.
//!
//! # Architecture
//! 1. Build Hi-Z pyramid from depth buffer (mip chain)
//! 2. Test object bounding boxes against Hi-Z
//! 3. Generate indirect draw commands for visible objects
//!
//! # Performance
//! - Reduces draw calls by 30-70% in complex scenes
//! - GPU-driven, no CPU readback required

use glam::{Mat4, Vec3};

/// Maximum number of objects that can be culled per frame
pub const MAX_CULLABLE_OBJECTS: usize = 65536;

/// Hi-Z pyramid levels (1024 -> 1 = 10 levels)
pub const HIZ_LEVELS: usize = 10;

/// Object bounding box for culling (GPU layout)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullBoundingBox {
    /// Center position (xyz) + padding (w)
    pub center: [f32; 4],
    /// Extents (half-sizes xyz) + padding (w)
    pub extents: [f32; 4],
}

/// Cluster bounding sphere (GPU layout)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullBoundingSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

impl CullBoundingBox {
    /// Create from min/max bounds
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        let center = (min + max) * 0.5;
        let extents = (max - min) * 0.5;
        Self {
            center: [center.x, center.y, center.z, 0.0],
            extents: [extents.x, extents.y, extents.z, 0.0],
        }
    }

    /// Create from center and extents
    pub fn new(center: Vec3, extents: Vec3) -> Self {
        Self {
            center: [center.x, center.y, center.z, 0.0],
            extents: [extents.x, extents.y, extents.z, 0.0],
        }
    }

    /// Get AABB corners
    pub fn corners(&self) -> [Vec3; 8] {
        let center = Vec3::new(self.center[0], self.center[1], self.center[2]);
        let extents = Vec3::new(self.extents[0], self.extents[1], self.extents[2]);
        [
            center + Vec3::new(-extents.x, -extents.y, -extents.z),
            center + Vec3::new(extents.x, -extents.y, -extents.z),
            center + Vec3::new(extents.x, extents.y, -extents.z),
            center + Vec3::new(-extents.x, extents.y, -extents.z),
            center + Vec3::new(-extents.x, -extents.y, extents.z),
            center + Vec3::new(extents.x, -extents.y, extents.z),
            center + Vec3::new(extents.x, extents.y, extents.z),
            center + Vec3::new(-extents.x, extents.y, extents.z),
        ]
    }
}

/// Per-object culling data
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullObjectData {
    /// Bounding box (for object-level) or sphere (for cluster-level)
    pub bounds: CullBoundingBox,
    /// Model matrix row 0
    pub model_row0: [f32; 4],
    /// Model matrix row 1
    pub model_row1: [f32; 4],
    /// Model matrix row 2
    pub model_row2: [f32; 4],
    /// Model matrix row 3
    pub model_row3: [f32; 4],
    /// Draw command index (index into template buffer)
    pub draw_index: u32,
    /// Override first index
    pub first_index: u32,
    /// Override index count
    pub index_count: u32,
    /// Override vertex offset
    pub vertex_offset: i32,
    /// Instance color multiplier (RGBA)
    pub color: [f32; 4],
    /// Custom data (user-defined)
    pub custom: [f32; 4],
    /// Cluster offset in global cluster buffer
    pub cluster_offset: u32,
    /// Number of clusters for this object
    pub cluster_count: u32,
    /// Culling flags (e.g., enabled, shadow-caster)
    pub flags: u32,
    /// Padding for 16-byte alignment
    pub _padding: u32,
}

impl CullObjectData {
    /// Create culling data for an object
    pub fn new(bounds: CullBoundingBox, model: Mat4, draw_index: u32) -> Self {
        let cols = model.to_cols_array_2d();
        Self {
            bounds,
            model_row0: cols[0],
            model_row1: cols[1],
            model_row2: cols[2],
            model_row3: cols[3],
            draw_index,
            first_index: 0, // 0 = use template
            index_count: 0, // 0 = use template
            vertex_offset: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            custom: [0.0; 4],
            cluster_offset: 0,
            cluster_count: 0,
            flags: 1,
            _padding: 0,
        }
    }

    /// Create culling data for a cluster
    pub fn for_cluster(
        center: [f32; 3],
        radius: f32,
        model: Mat4,
        draw_index: u32,
        first_index: u32,
        index_count: u32,
    ) -> Self {
        let cols = model.to_cols_array_2d();
        // Pack sphere into CullBoundingBox for unified data structure
        let bounds = CullBoundingBox {
            center: [center[0], center[1], center[2], 1.0], // w=1 means sphere mode
            extents: [radius, radius, radius, 0.0],
        };

        Self {
            bounds,
            model_row0: cols[0],
            model_row1: cols[1],
            model_row2: cols[2],
            model_row3: cols[3],
            draw_index,
            first_index,
            index_count,
            vertex_offset: 0,
            color: [1.0, 1.0, 1.0, 1.0],
            custom: [0.0; 4],
            cluster_offset: 0,
            cluster_count: 0,
            flags: 1,
            _padding: 0,
        }
    }

    /// Set custom data
    pub fn with_custom(mut self, custom: [f32; 4]) -> Self {
        self.custom = custom;
        self
    }

    /// Set color
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Get position from matrix
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.model_row3[0], self.model_row3[1], self.model_row3[2])
    }

    /// Create from matrix with default bounds/index
    pub fn from_matrix(model: Mat4) -> Self {
        Self::new(CullBoundingBox::default(), model, 0)
    }
}

/// Indirect draw command (matches VkDrawIndexedIndirectCommand)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndirectDrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

/// Occlusion culling push constants
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullingPushConstants {
    /// View-projection matrix (column-major)
    pub view_proj: [[f32; 4]; 4],
    /// Screen dimensions (width, height, 1/width, 1/height)
    pub screen_params: [f32; 4],
    /// Number of objects
    pub object_count: u32,
    /// Hi-Z pyramid levels
    pub hiz_levels: u32,
    /// Base object index
    pub base_index: u32,
    /// Indirect command start index
    pub indirect_start: u32,
    pub object_buffer_index: u32,
}

impl Default for CullingPushConstants {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            screen_params: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            object_count: 0,
            hiz_levels: HIZ_LEVELS as u32,
            base_index: 0,
            indirect_start: 0,
            object_buffer_index: 0,
        }
    }
}

/// Culling performance metrics
#[derive(Debug, Clone, Default)]
pub struct CullStats {
    pub total: u32,
    pub visible: u32,
    pub occlusion_culled: u32,
    pub draws_saved: u32,
}

impl CullStats {
    pub fn format(&self) -> String {
        let rate = if self.total > 0 {
            (self.occlusion_culled as f32 / self.total as f32) * 100.0
        } else {
            0.0
        };
        format!(
            "Culling: {}/{} visible ({:.1}% culled)",
            self.visible, self.total, rate
        )
    }
}

/// GPU-driven culling manager
pub struct OcclusionCulling {
    pub enabled: bool,
    pub frustum_only: bool,
    objects: Vec<CullObjectData>,
    stats: CullStats,
}

impl OcclusionCulling {
    pub fn new() -> Self {
        Self {
            enabled: true,
            frustum_only: false,
            objects: Vec::with_capacity(1024),
            stats: CullStats::default(),
        }
    }

    /// Enable/disable occlusion culling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Is culling enabled?
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set frustum-only mode (skip Hi-Z)
    pub fn set_frustum_only(&mut self, frustum_only: bool) {
        self.frustum_only = frustum_only;
    }

    pub fn begin_frame(&mut self) {
        self.objects.clear();
        // Stats are reset per frame
        self.stats = CullStats::default();
    }

    pub fn push_clusters(
        &mut self,
        bounds: CullBoundingBox,
        model: Mat4,
        draw_index: u32,
        clusters: &[crate::renderer::resources::mesh::MeshCluster],
    ) {
        if clusters.is_empty() {
            // Assume caller handles capacity for hot path performance
            self.objects
                .push(CullObjectData::new(bounds, model, draw_index));
        } else {
            for cluster in clusters {
                self.objects.push(CullObjectData::for_cluster(
                    cluster.bounds_center,
                    cluster.bounds_radius,
                    model,
                    draw_index,
                    cluster.first_index,
                    cluster.index_count,
                ));
            }
        }
    }

    pub fn add_object(&mut self, bounds: CullBoundingBox, model: Mat4, draw_index: u32) {
        self.push_clusters(bounds, model, draw_index, &[]);
    }

    /// Get object data for GPU upload
    pub fn object_data(&self) -> &[CullObjectData] {
        &self.objects
    }

    /// Get number of objects
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Create push constants
    pub fn push_constants(&self, view_proj: Mat4, width: u32, height: u32) -> CullingPushConstants {
        CullingPushConstants {
            view_proj: view_proj.to_cols_array_2d(),
            screen_params: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            object_count: self.objects.len() as u32,
            hiz_levels: HIZ_LEVELS as u32,
            base_index: 0,
            indirect_start: 0,
            object_buffer_index: 0,
        }
    }

    pub fn update_stats(&mut self, visible_count: u32) {
        self.stats.total = self.objects.len() as u32;
        self.stats.visible = visible_count;
        self.stats.occlusion_culled = self.stats.total.saturating_sub(visible_count);
    }

    pub fn stats(&self) -> &CullStats {
        &self.stats
    }
}

impl Default for OcclusionCulling {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_from_min_max() {
        let bb =
            CullBoundingBox::from_min_max(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(bb.center[0], 0.0);
        assert_eq!(bb.extents[0], 1.0);
        assert_eq!(bb.extents[1], 2.0);
    }

    #[test]
    fn test_occlusion_stats() {
        let stats = CullStats {
            total: 100,
            visible: 30,
            occlusion_culled: 70,
            ..Default::default()
        };
        // Just verify it doesn't crash
        let _ = stats.format();
    }

    #[test]
    fn test_cull_object_size() {
        // Ensure GPU-friendly alignment
        assert_eq!(std::mem::size_of::<CullObjectData>() % 16, 0);
    }
}
