//! Shadow mapping feature for directional lights
//!
//! Provides depth-based shadow mapping with PCF soft shadows.

use ash::Device;

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};
use crate::renderer::shadow_map::{ShadowConfig, ShadowMap};

/// Shadow mapping feature
pub struct ShadowFeature {
    shadow_map: Option<ShadowMap>,
    pub config: ShadowConfig,
    /// Light direction for directional shadows
    pub light_direction: glam::Vec3,
    /// Scene bounds for shadow frustum
    pub scene_center: glam::Vec3,
    pub scene_radius: f32,
}

impl ShadowFeature {
    /// Create a new shadow mapping feature
    pub fn new() -> Self {
        Self {
            shadow_map: None,
            config: ShadowConfig::default(),
            light_direction: glam::Vec3::new(-0.5, -1.0, -0.3).normalize(),
            scene_center: glam::Vec3::ZERO,
            scene_radius: 20.0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ShadowConfig) -> Self {
        Self {
            shadow_map: None,
            config,
            light_direction: glam::Vec3::new(-0.5, -1.0, -0.3).normalize(),
            scene_center: glam::Vec3::ZERO,
            scene_radius: 20.0,
        }
    }

    /// Set light direction
    pub fn set_light_direction(&mut self, dir: glam::Vec3) {
        self.light_direction = dir.normalize();
    }

    /// Set scene bounds for shadow frustum calculation
    pub fn set_scene_bounds(&mut self, center: glam::Vec3, radius: f32) {
        self.scene_center = center;
        self.scene_radius = radius;
    }

    /// Set the shadow map
    pub fn set_shadow_map(&mut self, shadow_map: ShadowMap) {
        self.shadow_map = Some(shadow_map);
    }

    /// Get shadow map reference if initialized
    pub fn shadow_map(&self) -> Option<&ShadowMap> {
        self.shadow_map.as_ref()
    }

    /// Get mutable shadow map reference
    pub fn shadow_map_mut(&mut self) -> Option<&mut ShadowMap> {
        self.shadow_map.as_mut()
    }

    /// Check if shadows are enabled and initialized
    pub fn is_active(&self) -> bool {
        self.config.enabled && self.shadow_map.is_some()
    }

    /// Get the light-space matrix
    pub fn light_space_matrix(&self) -> glam::Mat4 {
        self.shadow_map
            .as_ref()
            .map(|sm| sm.light_space_matrix)
            .unwrap_or(glam::Mat4::IDENTITY)
    }
}

impl Default for ShadowFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for ShadowFeature {
    fn name(&self) -> &'static str {
        "Shadows"
    }

    fn on_added(&mut self, device: &Device) {
        log::info!("[ShadowFeature] Shadow feature added");
        // Store device for later initialization
        // Note: We can't safely initialize here without memory properties
        // Shadow map will be created on first frame if needed
        let _ = device; // Acknowledge device
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Update light-space matrix based on current light direction
        if let Some(shadow_map) = &mut self.shadow_map {
            shadow_map.update_light_matrix(
                self.light_direction,
                self.scene_center,
                self.scene_radius,
            );
        }
    }

    unsafe fn render(&self, _ctx: &FeatureRenderContext<'_>) {
        // Shadow pass is handled separately in the main renderer
        // This is called after the main pass for any post-processing
    }

    fn on_removed(&mut self, _device: &Device) {
        log::info!("[ShadowFeature] Cleaning up shadow mapping");
        self.shadow_map = None;
    }
}
