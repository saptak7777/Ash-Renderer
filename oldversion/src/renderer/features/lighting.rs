use glam::Vec3;

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};

#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
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

#[derive(Debug, Clone)]
pub struct LightingConfig {
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            ambient_color: Vec3::splat(0.1),
            ambient_intensity: 1.0,
            directional_lights: vec![DirectionalLight::default()],
            point_lights: Vec::new(),
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
        Self { config, dirty: true }
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
        // A future implementation would upload the lighting config to GPU buffers here.
        self.dirty = false;
    }

    unsafe fn render(&self, _ctx: &FeatureRenderContext<'_>) {
        // Lighting is applied in the main render pass. Nothing to do for now.
    }
}
