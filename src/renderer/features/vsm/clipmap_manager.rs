//! Clipmap Manager - Handles toroidal scrolling and level management for directional lights

use glam::{Vec2, Vec3};

use super::page_manager::PageManager;
use super::resources::{PageRequest, VsmConfig};

/// Clipmap level state
#[derive(Debug, Clone)]
pub struct ClipmapLevel {
    /// World-space center (snapped to page boundaries)
    pub center: Vec2,
    /// World-space extent (width/height in meters)
    pub extent: f32,
    /// Layer index in page table array
    pub layer: u32,
    /// Previous center (for delta calculation)
    prev_center: Vec2,
}

impl ClipmapLevel {
    /// Create a new clipmap level
    pub fn new(layer: u32, base_extent: f32) -> Self {
        // Each level is 2x the size of the previous
        let extent = base_extent * (1 << layer) as f32;

        Self {
            center: Vec2::ZERO,
            extent,
            layer,
            prev_center: Vec2::ZERO,
        }
    }

    /// Calculate snap size (world space per page)
    fn snap_size(&self, page_table_resolution: u32) -> f32 {
        self.extent / page_table_resolution as f32
    }

    /// Snap center to page boundaries to prevent shimmering
    pub fn snap_center(&mut self, camera_pos: Vec3, page_table_resolution: u32) {
        let snap = self.snap_size(page_table_resolution);

        self.prev_center = self.center;

        // Snap to nearest page boundary
        self.center.x = (camera_pos.x / snap).round() * snap;
        self.center.y = (camera_pos.z / snap).round() * snap;
    }

    /// Calculate delta in pages since last update
    pub fn delta_pages(&self, page_table_resolution: u32) -> (i32, i32) {
        let snap = self.snap_size(page_table_resolution);

        let delta_x = ((self.center.x - self.prev_center.x) / snap).round() as i32;
        let delta_y = ((self.center.y - self.prev_center.y) / snap).round() as i32;

        (delta_x, delta_y)
    }

    /// Check if a world position is within this level's bounds
    pub fn contains(&self, world_pos: Vec2) -> bool {
        let half_extent = self.extent * 0.5;
        let min = self.center - Vec2::splat(half_extent);
        let max = self.center + Vec2::splat(half_extent);

        world_pos.x >= min.x && world_pos.x <= max.x && world_pos.y >= min.y && world_pos.y <= max.y
    }

    /// Convert world position to virtual page coordinates
    pub fn world_to_page(&self, world_pos: Vec2, page_table_resolution: u32) -> Option<(u32, u32)> {
        if !self.contains(world_pos) {
            return None;
        }

        let half_extent = self.extent * 0.5;
        let min = self.center - Vec2::splat(half_extent);

        // Normalize to [0, 1]
        let uv = (world_pos - min) / self.extent;

        // Convert to page coordinates
        let page_x = (uv.x * page_table_resolution as f32).floor() as u32;
        let page_y = (uv.y * page_table_resolution as f32).floor() as u32;

        Some((
            page_x.min(page_table_resolution - 1),
            page_y.min(page_table_resolution - 1),
        ))
    }

    /// Calculate the view-projection matrix for this clipmap level
    ///
    /// This matrix transforms world-space coordinates into the shadow map's clip space.
    /// The center is already snapped to prevent shimmering.
    pub fn view_projection_matrix(
        &self,
        light_dir: Vec3,
        page_table_resolution: u32,
    ) -> glam::Mat4 {
        // Texel size in world space
        let texel_size = self.snap_size(page_table_resolution);

        // Snap center to texel boundaries (already done in snap_center, but ensure it's precise)
        let snapped_center = Vec2::new(
            (self.center.x / texel_size).round() * texel_size,
            (self.center.y / texel_size).round() * texel_size,
        );

        // Orthographic projection covering the level's extent
        let half_extent = self.extent * 0.5;
        let projection = glam::Mat4::orthographic_rh(
            -half_extent, // left
            half_extent,  // right
            -half_extent, // bottom
            half_extent,  // top
            -1000.0,      // near (negative for RH, covers objects behind light)
            1000.0,       // far
        );

        // View matrix: Look from light direction toward the center
        // For directional lights, position doesn't matter (parallel rays)
        // We place the "eye" far along the light direction
        let light_pos =
            Vec3::new(snapped_center.x, 500.0, snapped_center.y) - light_dir.normalize() * 500.0;
        let target = Vec3::new(snapped_center.x, 0.0, snapped_center.y);
        let up = Vec3::Y;

        let view = glam::Mat4::look_at_rh(light_pos, target, up);

        // Combine: projection * view
        projection * view
    }

    /// Get the AABB for this clipmap level in world space
    pub fn aabb(&self) -> (Vec3, Vec3) {
        let half_extent = self.extent * 0.5;
        let min = Vec3::new(
            self.center.x - half_extent,
            -1000.0, // Vertical extent (arbitrary, covers most scenes)
            self.center.y - half_extent,
        );
        let max = Vec3::new(
            self.center.x + half_extent,
            1000.0,
            self.center.y + half_extent,
        );
        (min, max)
    }
}

/// Manages clipmap levels for directional lights
pub struct ClipmapManager {
    levels: Vec<ClipmapLevel>,
    config: VsmConfig,
    /// Pending region invalidations (min, max)
    pending_invalidations: Vec<(Vec3, Vec3)>,
}

impl ClipmapManager {
    /// Create a new clipmap manager
    pub fn new(config: VsmConfig) -> Self {
        let mut levels = Vec::new();

        for layer in 0..config.clipmap_levels {
            levels.push(ClipmapLevel::new(layer, config.clipmap_base_extent));
        }

        Self {
            levels,
            config,
            pending_invalidations: Vec::new(),
        }
    }

    /// Update clipmap centers based on camera position
    pub fn update(&mut self, camera_pos: Vec3) {
        let page_table_res = self.config.page_table_resolution();

        for level in &mut self.levels {
            level.snap_center(camera_pos, page_table_res);
        }
    }

    /// Invalidate a region of the world (e.g. for moving objects)
    pub fn invalidate_region(&mut self, min: Vec3, max: Vec3) {
        self.pending_invalidations.push((min, max));
    }

    /// Generate page requests for invalidated regions
    ///
    /// This is called after update() to determine which pages need to be re-rendered
    pub fn generate_invalidation_requests(
        &mut self,
        page_manager: &mut PageManager,
    ) -> Vec<PageRequest> {
        let mut requests = Vec::new();
        let page_table_res = self.config.page_table_resolution();

        // Process scrolling
        for level in &self.levels {
            let (delta_x, delta_y) = level.delta_pages(page_table_res);

            // Toroidal update: only request new pages that have scrolled into view
            let abs_dx = delta_x.abs();
            let abs_dy = delta_y.abs();

            if abs_dx >= page_table_res as i32 || abs_dy >= page_table_res as i32 {
                // If movement is larger than the entire table, invalidate everything
                for y in 0..page_table_res {
                    for x in 0..page_table_res {
                        requests.push(PageRequest {
                            virtual_x: x,
                            virtual_y: y,
                            priority: 1.0 / (level.layer + 1) as f32,
                            layer: level.layer,
                        });
                    }
                }
            } else {
                // Determine the new range in "World Page Coordinates" (unwrapped)
                // We wrap it to texture space using modulo at the end.

                let radius = (page_table_res as i32) / 2;

                // Note: delta_x = current - old.
                // We assume we need to update the edge in the direction of movement.
                //
                // Example: Moving Right (dx > 0).
                // New viewport covers [current - radius, current + radius].
                // Old viewport covered [current - dx - radius, current - dx + radius].
                // The "New" strip is [current + radius - dx + 1, current + radius] (approx).
                //
                // We will use the explicit logic from the plan:
                // Moving Right: Update Right Edge.

                // Helper to queue a page
                let mut queue_page = |wx: i32, wy: i32| {
                    let tx = wx.rem_euclid(page_table_res as i32) as u32;
                    let ty = wy.rem_euclid(page_table_res as i32) as u32;
                    requests.push(PageRequest {
                        virtual_x: tx,
                        virtual_y: ty,
                        priority: 1.0 / (level.layer + 1) as f32,
                        layer: level.layer,
                    });
                };

                // Current absolute page center (approximate for loop bounds)
                // We don't need exact absolute coordinates if we just iterate the *relative* range of the table
                // and apply the delta logic.
                //
                // Actually, since we don't have absolute coordinates, we CANNOT perfectly solve "which index corresponds to world X".
                // BUT, if we assume the loop 0..res represents the "Window", and we just update the specific indices...
                //
                // User's plan said: "Modular Arithmetic".
                // `page_x = (world_page_x) % table_size`.
                //
                // We WILL calculate approximate world page coords.
                let page_size = level.extent / page_table_res as f32;
                let center_world_x = (level.center.x / page_size).floor() as i32;
                let center_world_y = (level.center.y / page_size).floor() as i32;

                let min_x = center_world_x - radius;
                let max_x = center_world_x + radius - 1;
                let min_y = center_world_y - radius;
                let max_y = center_world_y + radius - 1;

                let old_center_x = center_world_x - delta_x;
                let old_center_y = center_world_y - delta_y;
                let old_min_x = old_center_x - radius;
                let old_max_x = old_center_x + radius - 1;
                let old_min_y = old_center_y - radius;
                let old_max_y = old_center_y + radius - 1;

                // X-Axis Updates
                if delta_x != 0 {
                    let (start, end) = if delta_x > 0 {
                        // Moving Right: New cols are (old_max_x, max_x]
                        (old_max_x + 1, max_x)
                    } else {
                        // Moving Left: New cols are [min_x, old_min_x)
                        (min_x, old_min_x - 1)
                    };

                    for wx in start..=end {
                        for wy in min_y..=max_y {
                            queue_page(wx, wy);
                        }
                    }
                }

                // Y-Axis Updates
                if delta_y != 0 {
                    let (start, end) = if delta_y > 0 {
                        (old_max_y + 1, max_y)
                    } else {
                        (min_y, old_min_y - 1)
                    };

                    for wy in start..=end {
                        for wx in min_x..=max_x {
                            queue_page(wx, wy);
                        }
                    }
                }
            }
        }

        // Process pending invalidations
        if !self.pending_invalidations.is_empty() {
            for (min, max) in self.pending_invalidations.drain(..) {
                for level in &self.levels {
                    // Check if AABB overlaps level
                    let level_half_size = level.extent * 0.5;
                    let level_min = Vec3::new(
                        level.center.x - level_half_size,
                        -1000.0,
                        level.center.y - level_half_size,
                    );
                    let level_max = Vec3::new(
                        level.center.x + level_half_size,
                        1000.0,
                        level.center.y + level_half_size,
                    );

                    if max.x < level_min.x
                        || min.x > level_max.x
                        || max.z < level_min.z
                        || min.z > level_max.z
                    {
                        continue;
                    }

                    // Convert world AABB to page indices
                    // We need to construct 2D bounds from the AABB (ignoring Y/Height as clipmaps are top-down)
                    // But standard VSM logic uses orthographic projection, so XZ plane matters.
                    // level.contains checks X and Y (where Y is Z in world space? No, level uses Vec2 for center)
                    // Wait, ClipmapLevel center is Vec2: (x, z) usually.
                    // Let's check `ClipmapLevel` struct. Assuming center is (x, z).

                    // We approximate by invalidating pages covering (min.x, min.z) to (max.x, max.z)
                    let min_pos = Vec2::new(min.x, min.z);
                    let max_pos = Vec2::new(max.x, max.z);

                    if let (Some((min_px, min_py)), Some((max_px, max_py))) = (
                        level.world_to_page(min_pos, page_table_res),
                        level.world_to_page(max_pos, page_table_res),
                    ) {
                        // Handle strict loop range if min/max on same page or spread
                        // Note: world_to_page returns clamped to resolution.
                        // We iterate range.

                        let start_x = min_px.min(max_px);
                        let end_x = min_px.max(max_px);
                        let start_y = min_py.min(max_py);
                        let end_y = min_py.max(max_py);

                        for y in start_x..=end_x {
                            for x in start_y..=end_y {
                                // Mark dirty
                                page_manager.invalidate_page(x, y, level.layer);

                                // Ensure allocated
                                requests.push(PageRequest {
                                    virtual_x: x,
                                    virtual_y: y,
                                    priority: 1.0, // High priority for updates
                                    layer: level.layer,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Process pending invalidations
        if !self.pending_invalidations.is_empty() {
            for (min, max) in self.pending_invalidations.drain(..) {
                for level in &self.levels {
                    // Check if AABB overlaps level
                    let level_half_size = level.extent * 0.5;
                    let level_min = Vec3::new(
                        level.center.x - level_half_size,
                        -1000.0,
                        level.center.y - level_half_size,
                    );
                    let level_max = Vec3::new(
                        level.center.x + level_half_size,
                        1000.0,
                        level.center.y + level_half_size,
                    );

                    if max.x < level_min.x
                        || min.x > level_max.x
                        || max.z < level_min.z
                        || min.z > level_max.z
                    {
                        continue;
                    }

                    // Convert world AABB to page indices
                    // We need to construct 2D bounds from the AABB (ignoring Y/Height as clipmaps are top-down)
                    // But standard VSM logic uses orthographic projection, so XZ plane matters.
                    // level.contains checks X and Y (where Y is Z in world space? No, level uses Vec2 for center)
                    // Wait, ClipmapLevel center is Vec2: (x, z) usually.
                    // Let's check `ClipmapLevel` struct. Assuming center is (x, z).

                    // We approximate by invalidating pages covering (min.x, min.z) to (max.x, max.z)
                    let min_pos = Vec2::new(min.x, min.z);
                    let max_pos = Vec2::new(max.x, max.z);

                    if let (Some((min_px, min_py)), Some((max_px, max_py))) = (
                        level.world_to_page(min_pos, page_table_res),
                        level.world_to_page(max_pos, page_table_res),
                    ) {
                        // Handle strict loop range if min/max on same page or spread
                        // Note: world_to_page returns clamped to resolution.
                        // We iterate range.

                        let start_x = min_px.min(max_px);
                        let end_x = min_px.max(max_px);
                        let start_y = min_py.min(max_py);
                        let end_y = min_py.max(max_py);

                        for y in start_x..=end_x {
                            for x in start_y..=end_y {
                                // Mark dirty
                                page_manager.invalidate_page(x, y, level.layer);

                                // Ensure allocated
                                requests.push(PageRequest {
                                    virtual_x: x,
                                    virtual_y: y,
                                    priority: 1.0, // High priority for updates
                                    layer: level.layer,
                                });
                            }
                        }
                    }
                }
            }
        }

        requests
    }

    /// Get clipmap level by index
    pub fn level(&self, index: usize) -> Option<&ClipmapLevel> {
        self.levels.get(index)
    }

    /// Get number of levels
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Select appropriate clipmap level for a world position
    pub fn select_level(&self, world_pos: Vec2) -> Option<usize> {
        // Start from finest level (0) and find first level that contains the position
        for (i, level) in self.levels.iter().enumerate() {
            if level.contains(world_pos) {
                return Some(i);
            }
        }
        None
    }

    /// Get iterator over all levels
    pub fn levels(&self) -> impl Iterator<Item = &ClipmapLevel> {
        self.levels.iter()
    }

    /// Get page table resolution
    pub fn page_table_resolution(&self) -> u32 {
        self.config.page_table_resolution()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipmap_level_snapping() {
        let mut level = ClipmapLevel::new(0, 100.0);
        let page_table_res = 128;

        // Snap to camera position
        level.snap_center(Vec3::new(5.5, 0.0, 3.2), page_table_res);

        // Center should be snapped to page boundaries
        let snap_size = level.snap_size(page_table_res);
        assert!((level.center.x % snap_size).abs() < 0.001);
        assert!((level.center.y % snap_size).abs() < 0.001);
    }

    #[test]
    fn test_clipmap_level_contains() {
        let mut level = ClipmapLevel::new(0, 100.0);
        level.center = Vec2::new(0.0, 0.0);

        assert!(level.contains(Vec2::new(0.0, 0.0)));
        assert!(level.contains(Vec2::new(25.0, 25.0)));
        assert!(!level.contains(Vec2::new(60.0, 60.0)));
    }

    #[test]
    fn test_clipmap_manager_level_selection() {
        let config = VsmConfig {
            clipmap_levels: 4,
            clipmap_base_extent: 100.0,
            ..Default::default()
        };

        let manager = ClipmapManager::new(config);

        // Close position should select level 0
        assert_eq!(manager.select_level(Vec2::new(10.0, 10.0)), Some(0));

        // Far position should select higher level
        // (depends on camera position, but this tests the logic)
    }
}
