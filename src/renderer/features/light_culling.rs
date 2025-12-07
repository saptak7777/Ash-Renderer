//! GPU Light Culling System
//!
//! Implements Forward+ style tile-based light culling using compute shaders.
//! This allows efficient rendering of scenes with hundreds of lights.
//!
//! # Architecture
//! 1. Compute shader divides screen into 16x16 pixel tiles
//! 2. For each tile, culls lights against the tile's frustum
//! 3. Outputs a per-tile light index list
//! 4. Fragment shader reads only relevant lights per tile

use glam::Mat4;

use super::lighting::{DirectionalLight, PointLight};

/// Maximum number of lights that can be culled
pub const MAX_LIGHTS: usize = 1024;

/// Maximum lights per tile
pub const MAX_LIGHTS_PER_TILE: usize = 256;

/// Tile size in pixels
pub const TILE_SIZE: u32 = 16;

/// GPU light structure (matches shader)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    /// xyz = position, w = radius
    pub position: [f32; 4],
    /// rgb = color, a = intensity
    pub color: [f32; 4],
    /// xyz = direction (for spot), w = type (0=point, 1=spot, 2=directional)
    pub direction: [f32; 4],
    /// x = innerConeAngle, y = outerConeAngle, z = falloff, w = enabled
    pub params: [f32; 4],
}

impl GpuLight {
    /// Create from a point light
    pub fn from_point_light(light: &PointLight) -> Self {
        Self {
            position: [
                light.position.x,
                light.position.y,
                light.position.z,
                light.radius,
            ],
            color: [light.color.x, light.color.y, light.color.z, light.intensity],
            direction: [0.0, 0.0, 0.0, 0.0], // Point light
            params: [0.0, 0.0, 1.0, 1.0],    // Enabled
        }
    }

    /// Create from a directional light (infinite radius)
    pub fn from_directional_light(light: &DirectionalLight) -> Self {
        Self {
            position: [0.0, 0.0, 0.0, f32::MAX], // Infinite radius
            color: [light.color.x, light.color.y, light.color.z, light.intensity],
            direction: [light.direction.x, light.direction.y, light.direction.z, 2.0],
            params: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

/// Push constants for light culling shader
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCullingPushConstants {
    /// Screen size in pixels
    pub screen_size: [u32; 2],
    /// Number of lights
    pub light_count: u32,
    /// Padding
    pub _padding: u32,
}

/// Camera data UBO for culling shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullingCameraData {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub inv_projection: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl Default for CullingCameraData {
    fn default() -> Self {
        Self {
            view: Mat4::IDENTITY.to_cols_array_2d(),
            projection: Mat4::IDENTITY.to_cols_array_2d(),
            inv_projection: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
        }
    }
}

/// Light culling configuration
#[derive(Clone, Debug)]
pub struct LightCullingConfig {
    /// Whether GPU culling is enabled
    pub enabled: bool,
    /// Debug visualization mode
    pub debug_tiles: bool,
}

impl Default for LightCullingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debug_tiles: false,
        }
    }
}

/// GPU Light Culling Pass
///
/// Manages the compute pipeline and buffers for tile-based light culling.
pub struct LightCullingPass {
    /// Configuration
    config: LightCullingConfig,
    /// Cached GPU lights
    lights: Vec<GpuLight>,
    /// Number of tiles in X
    tiles_x: u32,
    /// Number of tiles in Y
    tiles_y: u32,
    /// Last screen size
    last_screen_size: (u32, u32),
}

impl LightCullingPass {
    /// Create a new light culling pass
    pub fn new() -> Self {
        Self {
            config: LightCullingConfig::default(),
            lights: Vec::with_capacity(MAX_LIGHTS),
            tiles_x: 0,
            tiles_y: 0,
            last_screen_size: (0, 0),
        }
    }

    /// Create with config
    pub fn with_config(config: LightCullingConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Get config
    pub fn config(&self) -> &LightCullingConfig {
        &self.config
    }

    /// Get mutable config
    pub fn config_mut(&mut self) -> &mut LightCullingConfig {
        &mut self.config
    }

    /// Update lights from lighting config
    pub fn update_lights(
        &mut self,
        point_lights: &[PointLight],
        directional_lights: &[DirectionalLight],
    ) {
        self.lights.clear();

        // Add point lights
        for light in point_lights {
            if self.lights.len() >= MAX_LIGHTS {
                log::warn!("Light culling: exceeded max lights ({MAX_LIGHTS})");
                break;
            }
            self.lights.push(GpuLight::from_point_light(light));
        }

        // Add directional lights (these always affect all tiles)
        for light in directional_lights {
            if self.lights.len() >= MAX_LIGHTS {
                break;
            }
            self.lights.push(GpuLight::from_directional_light(light));
        }
    }

    /// Calculate number of tiles for screen size
    pub fn calculate_tiles(&mut self, width: u32, height: u32) {
        if (width, height) != self.last_screen_size {
            self.tiles_x = width.div_ceil(TILE_SIZE);
            self.tiles_y = height.div_ceil(TILE_SIZE);
            self.last_screen_size = (width, height);
            log::info!(
                "Light culling: {}x{} tiles for {}x{} screen",
                self.tiles_x,
                self.tiles_y,
                width,
                height
            );
        }
    }

    /// Get dispatch dimensions for compute shader
    pub fn get_dispatch_dimensions(&self) -> (u32, u32, u32) {
        (self.tiles_x, self.tiles_y, 1)
    }

    /// Get push constants for current state
    pub fn get_push_constants(
        &self,
        screen_width: u32,
        screen_height: u32,
    ) -> LightCullingPushConstants {
        LightCullingPushConstants {
            screen_size: [screen_width, screen_height],
            light_count: self.lights.len() as u32,
            _padding: 0,
        }
    }

    /// Get light buffer data
    pub fn get_light_buffer_data(&self) -> &[GpuLight] {
        &self.lights
    }

    /// Get number of lights
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }

    /// Get tile buffer size in bytes
    pub fn get_tile_buffer_size(&self) -> usize {
        let total_tiles = (self.tiles_x * self.tiles_y) as usize;
        // Each tile stores: [count, light_indices...]
        total_tiles * (MAX_LIGHTS_PER_TILE + 1) * std::mem::size_of::<u32>()
    }

    /// Is culling enabled?
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && !self.lights.is_empty()
    }
}

impl Default for LightCullingPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_gpu_light_from_point() {
        let point = PointLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::new(1.0, 0.5, 0.0),
            intensity: 2.0,
            radius: 10.0,
        };
        let gpu = GpuLight::from_point_light(&point);
        assert_eq!(gpu.position[3], 10.0); // radius
        assert_eq!(gpu.color[3], 2.0); // intensity
    }

    #[test]
    fn test_tile_calculation() {
        let mut pass = LightCullingPass::new();
        pass.calculate_tiles(1920, 1080);
        assert_eq!(pass.tiles_x, 120); // 1920/16 = 120
        assert_eq!(pass.tiles_y, 68); // ceil(1080/16) = 68
    }
}
