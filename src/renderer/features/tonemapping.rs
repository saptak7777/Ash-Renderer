//! Tonemapping Feature
//!
//! Applies HDR to LDR tonemapping with configurable operators.

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};
use ash::{vk, Device};

/// Tonemapping operator selection
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TonemapOperator {
    /// ACES filmic curve - industry standard, cinematic look
    #[default]
    Aces,
    /// Reinhard - classic, softer highlights
    Reinhard,
    /// Uncharted 2 filmic - game-optimized, good contrast
    Uncharted2,
    /// No tonemapping - direct HDR output (will clamp)
    None,
}

/// Configuration for tonemapping
#[derive(Debug, Clone, Copy)]
pub struct TonemappingConfig {
    /// Which tonemapping operator to use
    pub operator: TonemapOperator,
    /// Exposure multiplier (1.0 = neutral)
    pub exposure: f32,
    /// Gamma correction value (2.2 = standard sRGB)
    pub gamma: f32,
    /// Whether tonemapping is enabled
    pub enabled: bool,
}

impl Default for TonemappingConfig {
    fn default() -> Self {
        Self {
            operator: TonemapOperator::Aces,
            exposure: 1.0,
            gamma: 2.2,
            enabled: true,
        }
    }
}

/// Tonemapping render feature
pub struct TonemappingFeature {
    config: TonemappingConfig,
    pipeline: Option<vk::Pipeline>,
    device: Option<ash::Device>,
}

impl TonemappingFeature {
    /// Creates a new tonemapping feature with default config
    pub fn new() -> Self {
        Self {
            config: TonemappingConfig::default(),
            pipeline: None,
            device: None,
        }
    }

    /// Creates a new tonemapping feature with the given config
    pub fn with_config(config: TonemappingConfig) -> Self {
        Self {
            config,
            pipeline: None,
            device: None,
        }
    }

    /// Returns the current config
    pub fn config(&self) -> &TonemappingConfig {
        &self.config
    }

    /// Returns a mutable reference to the config
    pub fn config_mut(&mut self) -> &mut TonemappingConfig {
        &mut self.config
    }

    /// Sets the exposure value
    pub fn set_exposure(&mut self, exposure: f32) {
        self.config.exposure = exposure;
    }

    /// Sets the gamma value
    pub fn set_gamma(&mut self, gamma: f32) {
        self.config.gamma = gamma;
    }

    /// Sets the tonemapping operator
    pub fn set_operator(&mut self, operator: TonemapOperator) {
        self.config.operator = operator;
    }

    /// Toggles tonemapping on/off
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

impl Default for TonemappingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for TonemappingFeature {
    fn name(&self) -> &'static str {
        "TonemappingFeature"
    }

    fn on_added(&mut self, device: &Device) {
        log::info!(
            "Tonemapping feature added (operator: {:?})",
            self.config.operator
        );
        // Store device reference for pipeline creation
        // Pipeline will be created when we have access to shader modules
        self.device = Some(device.clone());
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Could animate exposure here for auto-exposure
    }

    unsafe fn render(&self, ctx: &FeatureRenderContext<'_>) {
        if !self.config.enabled {
            return;
        }

        // Tonemapping will be applied as a post-process pass
        // For now, this is a placeholder - the actual rendering happens
        // through the fullscreen pass infrastructure
        let _ = ctx;
    }

    fn on_removed(&mut self, device: &Device) {
        // Clean up pipeline if created
        if let Some(pipeline) = self.pipeline.take() {
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
        log::info!("Tonemapping feature removed");
    }
}
