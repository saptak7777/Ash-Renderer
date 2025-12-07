//! Level of Detail (LOD) System
//!
//! Provides automatic mesh quality selection based on screen-space size.
//! Reduces triangle count for distant objects while maintaining visual quality.
//!
//! # Features
//! - Screen-space size calculation
//! - Smooth LOD transitions (dithered)
//! - Per-object LOD bias
//! - Statistics tracking

use glam::{Mat4, Vec3};

/// Maximum LOD levels per mesh
pub const MAX_LOD_LEVELS: usize = 8;

/// LOD selection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodSelectionMode {
    /// Use screen-space projected size
    ScreenSize,
    /// Use direct distance from camera
    Distance,
    /// Force specific LOD level
    Force(usize),
}

impl Default for LodSelectionMode {
    fn default() -> Self {
        Self::ScreenSize
    }
}

/// LOD configuration
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Selection mode
    pub mode: LodSelectionMode,
    /// Enable smooth transitions (dithered)
    pub smooth_transitions: bool,
    /// Transition region size (0.0 - 1.0)
    pub transition_width: f32,
    /// Global LOD bias (-N = higher quality, +N = lower quality)
    pub global_bias: f32,
    /// Minimum screen coverage to render (0.0 - 1.0)
    pub cull_threshold: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            mode: LodSelectionMode::ScreenSize,
            smooth_transitions: true,
            transition_width: 0.1,
            global_bias: 0.0,
            cull_threshold: 0.001, // Cull if < 0.1% of screen
        }
    }
}

/// Per-LOD level data
#[derive(Debug, Clone, Default)]
pub struct LodLevel {
    /// Screen-space size threshold (0.0 - 1.0)
    pub screen_threshold: f32,
    /// Distance threshold (for distance mode)
    pub distance_threshold: f32,
    /// Triangle count at this LOD
    pub triangle_count: u32,
    /// Vertex count at this LOD
    pub vertex_count: u32,
    /// Index buffer offset
    pub index_offset: u32,
    /// Index count
    pub index_count: u32,
}

/// LOD mesh descriptor
#[derive(Debug, Clone)]
pub struct LodMesh {
    /// LOD levels (from highest to lowest quality)
    pub levels: Vec<LodLevel>,
    /// Bounding sphere radius
    pub bounding_radius: f32,
    /// Object-specific LOD bias
    pub lod_bias: f32,
}

impl LodMesh {
    /// Create with single LOD (no LOD switching)
    pub fn single(triangle_count: u32, bounding_radius: f32) -> Self {
        Self {
            levels: vec![LodLevel {
                screen_threshold: 0.0,
                distance_threshold: f32::MAX,
                triangle_count,
                vertex_count: triangle_count, // Approximate
                index_offset: 0,
                index_count: triangle_count * 3,
            }],
            bounding_radius,
            lod_bias: 0.0,
        }
    }

    /// Create with multiple LODs
    pub fn new(levels: Vec<LodLevel>, bounding_radius: f32) -> Self {
        Self {
            levels,
            bounding_radius,
            lod_bias: 0.0,
        }
    }

    /// Add LOD level
    pub fn with_level(mut self, level: LodLevel) -> Self {
        self.levels.push(level);
        self
    }

    /// Number of LOD levels
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

/// LOD selection result
#[derive(Debug, Clone, Copy)]
pub struct LodSelection {
    /// Selected LOD level index
    pub level: usize,
    /// Blend factor to next LOD (0.0 - 1.0, for smooth transitions)
    pub blend: f32,
    /// Whether object should be culled (too small)
    pub culled: bool,
    /// Screen-space coverage (0.0 - 1.0)
    pub screen_coverage: f32,
}

/// LOD manager
pub struct LodManager {
    config: LodConfig,
    /// Statistics
    stats: LodStats,
}

/// LOD system statistics
#[derive(Debug, Clone, Default)]
pub struct LodStats {
    /// Objects at each LOD level
    pub objects_per_lod: [u32; MAX_LOD_LEVELS],
    /// Objects culled (too small)
    pub objects_culled: u32,
    /// Total triangles before LOD
    pub triangles_before: u64,
    /// Total triangles after LOD
    pub triangles_after: u64,
}

impl LodStats {
    /// Calculate triangle reduction rate
    pub fn reduction_rate(&self) -> f64 {
        if self.triangles_before == 0 {
            0.0
        } else {
            1.0 - (self.triangles_after as f64 / self.triangles_before as f64)
        }
    }

    /// Format as summary string
    pub fn format(&self) -> String {
        format!(
            "LOD: {:.1}% reduction ({:.1}M → {:.1}M tris), {} culled",
            self.reduction_rate() * 100.0,
            self.triangles_before as f64 / 1_000_000.0,
            self.triangles_after as f64 / 1_000_000.0,
            self.objects_culled
        )
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl LodManager {
    /// Create a new LOD manager
    pub fn new() -> Self {
        Self::with_config(LodConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: LodConfig) -> Self {
        Self {
            config,
            stats: LodStats::default(),
        }
    }

    /// Begin new frame
    pub fn begin_frame(&mut self) {
        self.stats.reset();
    }

    /// Calculate screen-space coverage of a sphere
    pub fn calculate_screen_coverage(
        position: Vec3,
        radius: f32,
        view_proj: &Mat4,
        screen_width: f32,
        screen_height: f32,
    ) -> f32 {
        // Project sphere center
        let clip = *view_proj * position.extend(1.0);

        if clip.w <= 0.0 {
            return 0.0; // Behind camera
        }

        let _ndc = clip.truncate() / clip.w;

        // Approximate projected radius
        let dist = clip.w;
        let proj_radius = radius / dist;

        // Convert to screen pixels
        let pixel_radius = proj_radius * screen_width.max(screen_height) * 0.5;

        // Calculate coverage as ratio of screen area
        let area = std::f32::consts::PI * pixel_radius * pixel_radius;
        let screen_area = screen_width * screen_height;

        (area / screen_area).min(1.0)
    }

    /// Select LOD for an object
    pub fn select_lod(
        &mut self,
        mesh: &LodMesh,
        world_position: Vec3,
        view_proj: &Mat4,
        camera_position: Vec3,
        screen_width: f32,
        screen_height: f32,
    ) -> LodSelection {
        // Record base triangle count
        if !mesh.levels.is_empty() {
            self.stats.triangles_before += mesh.levels[0].triangle_count as u64;
        }

        // Force mode
        if let LodSelectionMode::Force(level) = self.config.mode {
            let level = level.min(mesh.levels.len().saturating_sub(1));
            if !mesh.levels.is_empty() {
                self.stats.triangles_after += mesh.levels[level].triangle_count as u64;
                self.stats.objects_per_lod[level] += 1;
            }
            return LodSelection {
                level,
                blend: 0.0,
                culled: false,
                screen_coverage: 1.0,
            };
        }

        // Calculate screen coverage
        let screen_coverage = Self::calculate_screen_coverage(
            world_position,
            mesh.bounding_radius,
            view_proj,
            screen_width,
            screen_height,
        );

        // Check cull threshold
        if screen_coverage < self.config.cull_threshold {
            self.stats.objects_culled += 1;
            return LodSelection {
                level: mesh.levels.len().saturating_sub(1),
                blend: 0.0,
                culled: true,
                screen_coverage,
            };
        }

        // Select LOD based on mode
        let (level, blend) = match self.config.mode {
            LodSelectionMode::ScreenSize => self.select_by_screen_size(mesh, screen_coverage),
            LodSelectionMode::Distance => {
                let distance = (world_position - camera_position).length();
                self.select_by_distance(mesh, distance)
            }
            LodSelectionMode::Force(_) => unreachable!(),
        };

        // Apply global and per-object bias
        let biased_level = (level as f32 + self.config.global_bias + mesh.lod_bias)
            .clamp(0.0, (mesh.levels.len() - 1) as f32) as usize;

        // Update stats
        if biased_level < mesh.levels.len() {
            self.stats.triangles_after += mesh.levels[biased_level].triangle_count as u64;
            if biased_level < MAX_LOD_LEVELS {
                self.stats.objects_per_lod[biased_level] += 1;
            }
        }

        LodSelection {
            level: biased_level,
            blend,
            culled: false,
            screen_coverage,
        }
    }

    fn select_by_screen_size(&self, mesh: &LodMesh, screen_coverage: f32) -> (usize, f32) {
        for (i, level) in mesh.levels.iter().enumerate() {
            if screen_coverage >= level.screen_threshold {
                // Calculate blend for smooth transitions
                let blend = if self.config.smooth_transitions && i + 1 < mesh.levels.len() {
                    let next_threshold = mesh.levels[i + 1].screen_threshold;
                    let range = level.screen_threshold - next_threshold;
                    if range > 0.0 {
                        let pos_in_range = (level.screen_threshold - screen_coverage) / range;
                        (pos_in_range / self.config.transition_width).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                return (i, blend);
            }
        }
        (mesh.levels.len().saturating_sub(1), 0.0)
    }

    fn select_by_distance(&self, mesh: &LodMesh, distance: f32) -> (usize, f32) {
        for (i, level) in mesh.levels.iter().enumerate() {
            if distance <= level.distance_threshold {
                return (i, 0.0);
            }
        }
        (mesh.levels.len().saturating_sub(1), 0.0)
    }

    /// Get current statistics
    pub fn stats(&self) -> &LodStats {
        &self.stats
    }

    /// Get mutable config
    pub fn config_mut(&mut self) -> &mut LodConfig {
        &mut self.config
    }
}

impl Default for LodManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_lod() {
        let mesh = LodMesh::single(1000, 1.0);
        assert_eq!(mesh.level_count(), 1);
    }

    #[test]
    fn test_screen_coverage() {
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let vp = proj * view;

        let coverage = LodManager::calculate_screen_coverage(Vec3::ZERO, 1.0, &vp, 1920.0, 1080.0);
        assert!(coverage > 0.0 && coverage < 1.0);
    }

    #[test]
    fn test_reduction_rate() {
        let stats = LodStats {
            triangles_before: 1000,
            triangles_after: 250,
            ..Default::default()
        };
        assert!((stats.reduction_rate() - 0.75).abs() < 0.001);
    }
}
