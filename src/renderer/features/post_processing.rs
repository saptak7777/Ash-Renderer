use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};

#[derive(Debug, Clone, Copy)]
pub struct PostProcessingConfig {
    pub enable_bloom: bool,
    pub enable_fxaa: bool,
    pub enable_tonemapping: bool,
    pub exposure: f32,
    pub gamma: f32,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            enable_bloom: true,
            enable_fxaa: true,
            enable_tonemapping: true,
            exposure: 1.0,
            gamma: 2.2,
        }
    }
}

pub struct PostProcessingFeature {
    config: PostProcessingConfig,
}

impl PostProcessingFeature {
    pub fn new() -> Self {
        Self {
            config: PostProcessingConfig::default(),
        }
    }

    pub fn with_config(config: PostProcessingConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PostProcessingConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut PostProcessingConfig {
        &mut self.config
    }
}

impl Default for PostProcessingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for PostProcessingFeature {
    fn name(&self) -> &'static str {
        "PostProcessingFeature"
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Hook for future parameter animation.
    }

    unsafe fn render(&self, _ctx: &FeatureRenderContext<'_>) {
        // Apply post-processing when we have a full-screen pass in place.
    }
}
