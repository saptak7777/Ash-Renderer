//! Bloom Feature
//!
//! Multi-pass bloom effect with threshold, downsample, and upsample stages.

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};
use ash::Device;

/// Configuration for bloom effect
#[derive(Debug, Clone, Copy)]
pub struct BloomConfig {
    /// Brightness threshold for bloom extraction (0.0 - 1.0)
    pub threshold: f32,
    /// Bloom intensity multiplier
    pub intensity: f32,
    /// Number of mip levels for blur (5-6 typical)
    pub mip_count: u32,
    /// Soft knee for threshold (smoother transition)
    pub soft_knee: f32,
    /// Whether bloom is enabled
    pub enabled: bool,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            intensity: 0.5,
            mip_count: 5,
            soft_knee: 0.5,
            enabled: true,
        }
    }
}

/// Bloom render feature
///
/// Implements a multi-pass bloom effect:
/// 1. Threshold pass: Extract bright pixels
/// 2. Downsample chain: Progressive gaussian blur
/// 3. Upsample chain: Additive blend back to full resolution
pub struct BloomFeature {
    config: BloomConfig,
    device: Option<ash::Device>,
    // Intermediate textures would be stored here
    // downsample_targets: Vec<vk::Image>,
    // upsample_targets: Vec<vk::Image>,
}

impl BloomFeature {
    /// Creates a new bloom feature with default config
    pub fn new() -> Self {
        Self {
            config: BloomConfig::default(),
            device: None,
        }
    }

    /// Creates a new bloom feature with the given config
    pub fn with_config(config: BloomConfig) -> Self {
        Self {
            config,
            device: None,
        }
    }

    /// Returns the current config
    pub fn config(&self) -> &BloomConfig {
        &self.config
    }

    /// Returns a mutable reference to the config
    pub fn config_mut(&mut self) -> &mut BloomConfig {
        &mut self.config
    }

    /// Sets the threshold value
    pub fn set_threshold(&mut self, threshold: f32) {
        self.config.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Sets the intensity value
    pub fn set_intensity(&mut self, intensity: f32) {
        self.config.intensity = intensity.max(0.0);
    }

    /// Sets the mip count
    pub fn set_mip_count(&mut self, mip_count: u32) {
        self.config.mip_count = mip_count.clamp(1, 10);
    }

    /// Toggles bloom on/off
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

impl Default for BloomFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for BloomFeature {
    fn name(&self) -> &'static str {
        "BloomFeature"
    }

    fn on_added(&mut self, device: &Device) {
        log::info!(
            "Bloom feature added (threshold: {}, intensity: {}, mips: {})",
            self.config.threshold,
            self.config.intensity,
            self.config.mip_count
        );
        self.device = Some(device.clone());
        // TODO: Create downsample/upsample render targets
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Could animate bloom parameters here
    }

    unsafe fn render(&self, ctx: &FeatureRenderContext<'_>) {
        if !self.config.enabled {
            return;
        }

        // Bloom rendering will happen through dedicated passes:
        // 1. Threshold extraction
        // 2. Downsample chain (mip_count passes)
        // 3. Upsample chain (mip_count passes)
        // 4. Final composite with HDR buffer
        let _ = ctx;
    }

    fn on_removed(&mut self, _device: &Device) {
        // TODO: Clean up downsample/upsample targets
        log::info!("Bloom feature removed");
    }
}
