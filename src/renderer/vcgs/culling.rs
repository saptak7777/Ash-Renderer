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

/// Culling flags (bits)
pub const CULL_FLAG_ENABLED: u32 = 1 << 0;
pub const CULL_FLAG_CAST_SHADOWS: u32 = 1 << 1;
pub const CULL_FLAG_TRANSPARENT: u32 = 1 << 2;
pub const CULL_FLAG_RECEIVE_SHADOWS: u32 = 1 << 3;
pub const CULL_FLAG_HIDDEN: u32 = 1 << 4;

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
    /// VCGS: Parent cluster index (u32::MAX if root)
    pub parent_index: u32,
    /// VCGS: Error metric for LOD selection
    pub error_metric: f32,
    /// Culling flags (e.g., enabled, shadow-caster)
    pub flags: u32,
    /// Material handle/index for BDA material pulling
    pub material_index: u32,
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
            parent_index: u32::MAX,
            error_metric: 0.0,
            flags: 1,
            material_index: 0,
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
        parent_index: u32,
        error_metric: f32,
        material_index: u32,
        vertex_offset: i32,
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
            vertex_offset,
            color: [1.0, 1.0, 1.0, 1.0],
            custom: [0.0; 4],
            parent_index,
            error_metric,
            flags: 1,
            material_index,
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

    /// Set cast shadows flag
    pub fn with_cast_shadows(mut self, enabled: bool) -> Self {
        self.set_flag(CULL_FLAG_CAST_SHADOWS, enabled);
        self
    }

    /// Set transparent flag
    pub fn with_transparent(mut self, enabled: bool) -> Self {
        self.set_flag(CULL_FLAG_TRANSPARENT, enabled);
        self
    }

    /// Set receive shadows flag
    pub fn with_receive_shadows(mut self, enabled: bool) -> Self {
        self.set_flag(CULL_FLAG_RECEIVE_SHADOWS, enabled);
        self
    }

    /// Set hidden flag
    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.set_flag(CULL_FLAG_HIDDEN, hidden);
        self
    }

    /// Set material index
    pub fn with_material_index(mut self, material_index: u32) -> Self {
        self.material_index = material_index;
        self
    }

    /// Set index count
    pub fn with_index_count(mut self, index_count: u32) -> Self {
        self.index_count = index_count;
        self
    }

    /// Set first index
    pub fn with_first_index(mut self, first_index: u32) -> Self {
        self.first_index = first_index;
        self
    }

    /// Set vertex offset
    pub fn with_vertex_offset(mut self, vertex_offset: i32) -> Self {
        self.vertex_offset = vertex_offset;
        self
    }

    /// Set bounding box
    pub fn with_bounds(mut self, bounds: CullBoundingBox) -> Self {
        self.bounds = bounds;
        self
    }

    /// Check if transparent
    pub fn is_transparent(&self) -> bool {
        self.has_flag(CULL_FLAG_TRANSPARENT)
    }

    /// Get position from matrix
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.model_row3[0], self.model_row3[1], self.model_row3[2])
    }

    /// Get model matrix
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_cols_array_2d(&[
            self.model_row0,
            self.model_row1,
            self.model_row2,
            self.model_row3,
        ])
    }

    /// Calculate world-space radius (conservative)
    pub fn world_radius(&self) -> f32 {
        let extents = Vec3::new(
            self.bounds.extents[0],
            self.bounds.extents[1],
            self.bounds.extents[2],
        );
        let local_radius = extents.length();

        let scale_x =
            Vec3::new(self.model_row0[0], self.model_row0[1], self.model_row0[2]).length();
        let scale_y =
            Vec3::new(self.model_row1[0], self.model_row1[1], self.model_row1[2]).length();
        let scale_z =
            Vec3::new(self.model_row2[0], self.model_row2[1], self.model_row2[2]).length();
        let max_scale = scale_x.max(scale_y).max(scale_z);

        local_radius * max_scale
    }

    /// Create from matrix with default bounds/index
    pub fn from_matrix(model: Mat4) -> Self {
        Self::new(CullBoundingBox::default(), model, 0)
    }

    pub fn set_flag(&mut self, flag: u32, enabled: bool) {
        if enabled {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }

    pub fn has_flag(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }
}

/// Indirect draw command (matches VkDrawIndirectCommand)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndirectDrawCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
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
    /// Indirect command start index
    pub indirect_start: u32,
    pub object_buffer_addr: u64,
    /// Debug mode (0=None, 1=LOD, 2=ClusterID)
    pub debug_mode: u32,
    pub _padding: u32,
    /// Address of Global Cluster Buffer
    pub cluster_buffer_addr: u64,
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
            object_buffer_addr: 0,
            debug_mode: 0,
            _padding: 0,
            cluster_buffer_addr: 0,
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
    pub debug_enabled: bool,
    /// Cache for visible indices set to avoid per-frame allocation in debug_boxes
    visible_set_cache: std::collections::HashSet<u32>,
    objects: Vec<CullObjectData>,
    stats: CullStats,
}

impl OcclusionCulling {
    pub fn new() -> Self {
        Self {
            enabled: true,
            frustum_only: false,
            debug_enabled: false,
            visible_set_cache: std::collections::HashSet::with_capacity(1024),
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
        first_index: u32,
        index_count: u32,
        material_index: u32,
        vertex_offset: i32,
        clusters: &[crate::renderer::resources::mesh::MeshCluster],
    ) {
        if clusters.is_empty() {
            // Assume caller handles capacity for hot path performance
            let mut data = CullObjectData::new(bounds, model, draw_index);
            data.first_index = first_index;
            data.index_count = index_count;
            data.material_index = material_index;
            data.vertex_offset = vertex_offset;
            self.objects.push(data);
        } else {
            let cluster_start_offset = self.objects.len() as u32;
            for cluster in clusters {
                let global_parent = if cluster.parent_index == u32::MAX {
                    u32::MAX
                } else {
                    cluster_start_offset + cluster.parent_index
                };

                self.objects.push(CullObjectData::for_cluster(
                    cluster.bounds_center,
                    cluster.bounds_radius,
                    model,
                    draw_index,
                    cluster.first_index,
                    cluster.index_count,
                    global_parent,
                    cluster.error_metric,
                    material_index,
                    vertex_offset,
                ));
            }
        }
    }

    pub fn add_object(
        &mut self,
        bounds: CullBoundingBox,
        model: Mat4,
        draw_index: u32,
        first_index: u32,
        index_count: u32,
        material_index: u32,
        vertex_offset: i32,
    ) {
        self.push_clusters(
            bounds,
            model,
            draw_index,
            first_index,
            index_count,
            material_index,
            vertex_offset,
            &[],
        );
    }

    /// Get object data for GPU upload
    pub fn object_data(&self) -> &[CullObjectData] {
        &self.objects
    }

    /// Get mutable object data
    pub fn object_data_mut(&mut self) -> &mut Vec<CullObjectData> {
        &mut self.objects
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
            object_buffer_addr: 0,
            debug_mode: 0,
            _padding: 0,
            cluster_buffer_addr: 0,
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

    /// Generate debug visualization data for culled objects
    ///
    /// Returns a vector of (transform, color) pairs for rendering bounding boxes.
    /// Color encoding: Green = Visible, Red = Culled
    pub fn debug_boxes(&mut self, visible_indices: &[u32]) -> Vec<(Mat4, [f32; 4])> {
        if !self.debug_enabled || self.objects.is_empty() {
            return Vec::new();
        }

        let mut boxes = Vec::with_capacity(self.objects.len());

        // Use the persistent set to avoid re-allocation
        self.visible_set_cache.clear();
        self.visible_set_cache
            .extend(visible_indices.iter().copied());

        for (i, obj) in self.objects.iter().enumerate() {
            let is_visible = self.visible_set_cache.contains(&(i as u32));

            // Reconstruct model matrix from rows
            let model = Mat4::from_cols_array_2d(&[
                obj.model_row0,
                obj.model_row1,
                obj.model_row2,
                obj.model_row3,
            ]);

            // Scale unit cube to match bounding box
            let center = Vec3::new(
                obj.bounds.center[0],
                obj.bounds.center[1],
                obj.bounds.center[2],
            );
            let extents = Vec3::new(
                obj.bounds.extents[0],
                obj.bounds.extents[1],
                obj.bounds.extents[2],
            );

            let scale = Mat4::from_scale(extents);
            let translate = Mat4::from_translation(center);
            let transform = model * translate * scale;

            // Color: Green if visible, Red if culled
            let color = if is_visible {
                [0.0, 1.0, 0.0, 0.5] // Green, semi-transparent
            } else {
                [1.0, 0.0, 0.0, 0.5] // Red, semi-transparent
            };

            boxes.push((transform, color));
        }

        boxes
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
