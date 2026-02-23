use glam::Vec3;

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};

#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: [f32; 4]) -> Self {
        Self {
            direction,
            color: Vec3::new(color[0], color[1], color[2]),
            intensity: color[3],
        }
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::splat(1.0),
            intensity: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

impl PointLight {
    pub fn new(position: Vec3, color: [f32; 4], radius: f32) -> Self {
        Self {
            position,
            color: Vec3::new(color[0], color[1], color[2]),
            intensity: color[3],
            radius,
        }
    }
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::splat(1.0),
            intensity: 1.0,
            radius: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpotLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub direction: Vec3,
    pub inner_angle: f32,
    pub outer_angle: f32,
}

impl SpotLight {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: [f32; 4],
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            color: Vec3::new(color[0], color[1], color[2]),
            intensity: color[3],
            range,
            inner_angle,
            outer_angle,
        }
    }
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::splat(1.0),
            intensity: 1.0,
            range: 10.0,
            inner_angle: 0.5,
            outer_angle: 0.785,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LightingConfig {
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            ambient_color: Vec3::splat(0.1),
            ambient_intensity: 1.0,
            directional_lights: vec![DirectionalLight::default()],
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
        }
    }
}

pub struct LightingFeature {
    config: LightingConfig,
    dirty: bool,
}

impl LightingFeature {
    pub fn new() -> Self {
        Self {
            config: LightingConfig::default(),
            dirty: true,
        }
    }

    pub fn with_config(config: LightingConfig) -> Self {
        Self {
            config,
            dirty: true,
        }
    }

    pub fn config(&self) -> &LightingConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut LightingConfig {
        self.dirty = true;
        &mut self.config
    }
}

impl Default for LightingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for LightingFeature {
    fn name(&self) -> &'static str {
        "LightingFeature"
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Upload lighting configuration to GPU when dirty
        if self.dirty {
            // For now, we'll mark the feature as handled. The actual lighting upload
            // happens in the main render loop through renderer.set_lighting() and
            // forward_plus.upload_to_gpu(). This method serves as a coordination point
            // for future lighting system improvements.
            self.dirty = false;
        }
    }

    unsafe fn render(&self, _ctx: &FeatureRenderContext<'_>) {
        // Lighting is applied in the main render pass. Nothing to do for now.
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
