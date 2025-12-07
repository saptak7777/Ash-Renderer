use std::f32::consts::TAU;

use glam::Vec3;

use super::{FeatureFrameContext, RenderFeature};

pub struct AutoRotateFeature;

impl AutoRotateFeature {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AutoRotateFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for AutoRotateFeature {
    fn name(&self) -> &'static str {
        "AutoRotateFeature"
    }

    fn before_frame(&mut self, ctx: &mut FeatureFrameContext<'_>) {
        if !ctx.auto_rotate {
            return;
        }

        let angle = ctx.elapsed_seconds * TAU / 6.0;
        ctx.transform
            .set_rotation(Vec3::new(angle * 0.5, angle, angle * 0.3));
    }
}
