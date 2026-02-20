//! Lighting System
//!
//! Manages all scene lighting state and drives the Forward+ light culling pipeline.
//! This is the single authoritative owner for per-frame light updates.
//!
//! # Responsibilities
//! - Accepts typed light slices from the application (point, directional, spot).
//! - Forwards them to the underlying `ForwardPlusIntegration` (light GPU buffers).
//! - Exposes simple query helpers so the renderer does not need to touch internals.

use std::sync::{Arc, RwLock};

use crate::renderer::{
    features::{DirectionalLight, PointLight, SpotLight},
    forward_plus_integration::ForwardPlusIntegration,
};
use crate::{AshError, Result};

/// Persistent lighting orchestrator owned by `Systems`.
///
/// Wraps the `ForwardPlusIntegration` and provides ergonomic light-update
/// methods so that `Renderer` is no longer responsible for routing lighting
/// calls through the pipeline graph.
pub struct LightingSystem {
    /// Forward+ integration: owns light buffers, cull compute pipeline, etc.
    /// Kept behind `Arc<RwLock<_>>` to match the existing ownership pattern in
    /// `RenderPipeline` (which also holds a reference for descriptor binding).
    pub forward_plus: Arc<RwLock<ForwardPlusIntegration>>,
}

impl LightingSystem {
    /// Wraps an already-initialised `ForwardPlusIntegration`.
    ///
    /// The same `Arc` is shared with `RenderPipeline` so both sides see the
    /// same GPU state without copying.
    pub fn new(forward_plus: Arc<RwLock<ForwardPlusIntegration>>) -> Self {
        Self { forward_plus }
    }

    // ─── Light Update API (previously on Renderer) ───────────────────────────

    /// Replace the current point light list.
    ///
    /// Call once per frame before submitting render commands.
    pub fn update_point_lights(&self, lights: &[PointLight]) -> Result<()> {
        let mut fp = self
            .forward_plus
            .write()
            .map_err(|_| AshError::LockPoisoned("LightingSystem ForwardPlus".to_string()))?;
        fp.update_lights(lights, &[], &[]);
        Ok(())
    }

    /// Replace the current directional light list.
    pub fn update_directional_lights(&self, lights: &[DirectionalLight]) -> Result<()> {
        let mut fp = self
            .forward_plus
            .write()
            .map_err(|_| AshError::LockPoisoned("LightingSystem ForwardPlus".to_string()))?;
        fp.update_lights(&[], lights, &[]);
        Ok(())
    }

    /// Replace the current spot light list.
    pub fn update_spot_lights(&self, lights: &[SpotLight]) -> Result<()> {
        let mut fp = self
            .forward_plus
            .write()
            .map_err(|_| AshError::LockPoisoned("LightingSystem ForwardPlus".to_string()))?;
        fp.update_lights(&[], &[], lights);
        Ok(())
    }

    /// Convenience: update point *and* directional lights in a single call.
    pub fn update_lights(
        &self,
        point_lights: &[PointLight],
        directional_lights: &[DirectionalLight],
    ) -> Result<()> {
        let mut fp = self
            .forward_plus
            .write()
            .map_err(|_| AshError::LockPoisoned("LightingSystem ForwardPlus".to_string()))?;
        fp.update_lights(point_lights, directional_lights, &[]);
        Ok(())
    }

    /// Update all three light types in a single call.
    pub fn update_all_lights(
        &self,
        point_lights: &[PointLight],
        directional_lights: &[DirectionalLight],
        spot_lights: &[SpotLight],
    ) -> Result<()> {
        let mut fp = self
            .forward_plus
            .write()
            .map_err(|_| AshError::LockPoisoned("LightingSystem ForwardPlus".to_string()))?;
        fp.update_lights(point_lights, directional_lights, spot_lights);
        Ok(())
    }

    // ─── Query Helpers ────────────────────────────────────────────────────────

    /// Returns `true` if the Forward+ integration has at least one active light.
    pub fn is_lighting_enabled(&self) -> bool {
        self.forward_plus
            .read()
            .map(|fp| fp.is_enabled())
            .unwrap_or(false)
    }

    /// Returns the total number of GPU-visible lights.
    pub fn light_count(&self) -> usize {
        self.forward_plus
            .read()
            .map(|fp| fp.light_count())
            .unwrap_or(0)
    }
}
