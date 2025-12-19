//! Bloom Feature
//!
//! Multi-pass bloom effect with threshold, downsample, and upsample stages.
//! Implements industry-standard dual-filtering bloom with firefly suppression.

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};
use ash::Device;

/// Push constants for bloom shaders, corresponding to the GLSL layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BloomPushConstants {
    /// Texel size (1.0 / texture_width, 1.0 / texture_height)
    pub texel_size: [f32; 2],
    /// Brightness threshold for prefilter
    pub threshold: f32,
    /// Soft knee for smooth threshold transition
    pub soft_knee: f32,
}

/// Configuration for bloom effect
#[derive(Debug, Clone, Copy)]
pub struct BloomConfig {
    /// Brightness threshold for bloom extraction (0.0 - 2.0, default 1.0)
    pub threshold: f32,
    /// Bloom intensity multiplier (0.0 - 1.0, default 0.5)
    pub intensity: f32,
    /// Number of mip levels for blur (3-8, default 5)
    pub mip_count: u32,
    /// Soft knee for threshold (0.0 - 1.0, default 0.5)
    pub soft_knee: f32,
    /// Whether bloom is enabled
    pub enabled: bool,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            intensity: 0.5,
            mip_count: 5,
            soft_knee: 0.5,
            enabled: true,
        }
    }
}

/// Mip level info for bloom chain
#[derive(Debug, Clone, Copy)]
pub struct MipInfo {
    pub width: u32,
    pub height: u32,
}

/// Bloom pass data manager (CPU-side logic, GPU-agnostic)
///
/// Handles mip chain calculations and push constant generation.
/// GPU resources are managed by BloomFeature.
pub struct BloomPass {
    config: BloomConfig,
    mip_chain: Vec<MipInfo>,
    base_width: u32,
    base_height: u32,
}

impl BloomPass {
    /// Create a new bloom pass with default config
    pub fn new() -> Self {
        Self {
            config: BloomConfig::default(),
            mip_chain: Vec::new(),
            base_width: 0,
            base_height: 0,
        }
    }

    /// Create with custom config
    pub fn with_config(config: BloomConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Get config reference
    pub fn config(&self) -> &BloomConfig {
        &self.config
    }

    /// Get mutable config reference
    pub fn config_mut(&mut self) -> &mut BloomConfig {
        &mut self.config
    }

    /// Calculate mip chain for given resolution
    ///
    /// Should be called on resize or first frame.
    pub fn calculate_mip_chain(&mut self, width: u32, height: u32) {
        if width == self.base_width && height == self.base_height {
            return; // Already calculated
        }

        self.base_width = width;
        self.base_height = height;
        self.mip_chain.clear();

        let mut mip_width = width;
        let mut mip_height = height;

        for _ in 0..self.config.mip_count {
            // Each subsequent mip level is half the dimensions of the previous, with a minimum size of 1x1.
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);

            self.mip_chain.push(MipInfo {
                width: mip_width,
                height: mip_height,
            });

            // Termination condition reached if dimensions are 1x1.
            if mip_width == 1 && mip_height == 1 {
                break;
            }
        }

        log::debug!(
            "Bloom: calculated {} mip levels for {}x{} (smallest: {}x{})",
            self.mip_chain.len(),
            width,
            height,
            self.mip_chain.last().map(|m| m.width).unwrap_or(0),
            self.mip_chain.last().map(|m| m.height).unwrap_or(0),
        );
    }

    /// Get push constants for prefilter pass
    pub fn get_prefilter_push_constants(&self) -> BloomPushConstants {
        BloomPushConstants {
            texel_size: [1.0 / self.base_width as f32, 1.0 / self.base_height as f32],
            threshold: self.config.threshold,
            soft_knee: self.config.soft_knee,
        }
    }

    /// Get push constants for a specific downsample mip level
    pub fn get_downsample_push_constants(&self, mip_index: usize) -> Option<BloomPushConstants> {
        self.mip_chain.get(mip_index).map(|mip| BloomPushConstants {
            texel_size: [1.0 / mip.width as f32, 1.0 / mip.height as f32],
            threshold: 0.0, // Not used in downsample
            soft_knee: 0.0,
        })
    }

    /// Get push constants for a specific upsample mip level
    pub fn get_upsample_push_constants(&self, mip_index: usize) -> Option<BloomPushConstants> {
        // Upsample goes from smallest to largest
        let target_index = self.mip_chain.len().saturating_sub(1 + mip_index);
        self.mip_chain
            .get(target_index)
            .map(|mip| BloomPushConstants {
                texel_size: [1.0 / mip.width as f32, 1.0 / mip.height as f32],
                threshold: self.config.intensity, // Repurpose for blend factor
                soft_knee: 0.0,
            })
    }

    /// Get number of mip levels in the chain
    pub fn mip_count(&self) -> usize {
        self.mip_chain.len()
    }

    /// Get mip info for a specific level
    pub fn get_mip_info(&self, index: usize) -> Option<&MipInfo> {
        self.mip_chain.get(index)
    }

    /// Is bloom enabled and valid?
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && !self.mip_chain.is_empty()
    }
}

impl Default for BloomPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Bloom render feature
///
/// Integrates BloomPass with the rendering pipeline.
/// Manages GPU resources (images, framebuffers, pipelines).
pub struct BloomFeature {
    pass: BloomPass,
    device: Option<ash::Device>,
    // GPU resources will be lazily initialized
    // prefilter_pipeline: Option<vk::Pipeline>,
    // downsample_pipeline: Option<vk::Pipeline>,
    // upsample_pipeline: Option<vk::Pipeline>,
    // mip_images: Vec<vk::Image>,
    // mip_views: Vec<vk::ImageView>,
}

impl BloomFeature {
    /// Creates a new bloom feature with default config
    pub fn new() -> Self {
        Self {
            pass: BloomPass::new(),
            device: None,
        }
    }

    /// Creates a new bloom feature with the given config
    pub fn with_config(config: BloomConfig) -> Self {
        Self {
            pass: BloomPass::with_config(config),
            device: None,
        }
    }

    /// Returns the current config
    pub fn config(&self) -> &BloomConfig {
        self.pass.config()
    }

    /// Returns a mutable reference to the config
    pub fn config_mut(&mut self) -> &mut BloomConfig {
        self.pass.config_mut()
    }

    /// Access the underlying BloomPass
    pub fn pass(&self) -> &BloomPass {
        &self.pass
    }

    /// Access the underlying BloomPass mutably
    pub fn pass_mut(&mut self) -> &mut BloomPass {
        &mut self.pass
    }

    /// Sets the threshold value
    pub fn set_threshold(&mut self, threshold: f32) {
        self.pass.config_mut().threshold = threshold.clamp(0.0, 2.0);
    }

    /// Sets the intensity value
    pub fn set_intensity(&mut self, intensity: f32) {
        self.pass.config_mut().intensity = intensity.clamp(0.0, 1.0);
    }

    /// Sets the mip count (requires re-calculation of chain)
    pub fn set_mip_count(&mut self, mip_count: u32) {
        self.pass.config_mut().mip_count = mip_count.clamp(3, 8);
        // Force recalculation on next frame
        self.pass.base_width = 0;
        self.pass.base_height = 0;
    }

    /// Toggles bloom on/off
    pub fn set_enabled(&mut self, enabled: bool) {
        self.pass.config_mut().enabled = enabled;
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
        let cfg = self.pass.config();
        log::info!(
            "Bloom feature added (threshold: {:.2}, intensity: {:.2}, mips: {})",
            cfg.threshold,
            cfg.intensity,
            cfg.mip_count
        );
        self.device = Some(device.clone());
        // Pipelines for prefilter, downsample, and upsample stages remain to be implemented.
        // Allocation of mip chain images remains to be implemented.
    }

    fn before_frame(&mut self, ctx: &mut FeatureFrameContext<'_>) {
        if !self.pass.config().enabled {
            return;
        }

        // Mip chain recalculation based on screen dimensions is pending.
        // Placeholder implementation logic follows.
        // self.pass.calculate_mip_chain(screen_width, screen_height);
        let _ = ctx;
    }

    unsafe fn render(&self, ctx: &FeatureRenderContext<'_>) {
        if !self.pass.is_enabled() {
            return;
        }

        // Bloom rendering pipeline:
        // 1. Prefilter: Extract bright pixels using bloom_prefilter.frag
        // 2. Downsample chain: Progressive blur using bloom_downsample.frag
        // 3. Upsample chain: Additive blend back using bloom_upsample.frag
        // 4. Composite: Blend with original HDR buffer

        // Push constants example for prefilter:
        // let prefilter_pc = self.pass.get_prefilter_push_constants();
        // device.cmd_push_constants(
        //     ctx.command_buffer,
        //     pipeline_layout,
        //     vk::ShaderStageFlags::FRAGMENT,
        //     0,
        //     bytemuck::bytes_of(&prefilter_pc),
        // );

        let _ = ctx;
    }

    fn on_removed(&mut self, device: &Device) {
        // GPU resources (pipelines, images, and views) cleanup is pending.
        let _ = device;
        log::info!("Bloom feature removed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mip_chain_calculation() {
        let mut pass = BloomPass::new();
        pass.calculate_mip_chain(1920, 1080);

        // Default is 5 mips
        assert_eq!(pass.mip_count(), 5);

        // First mip should be half of 1920x1080
        let mip0 = pass.get_mip_info(0).unwrap();
        assert_eq!(mip0.width, 960);
        assert_eq!(mip0.height, 540);

        // Second mip
        let mip1 = pass.get_mip_info(1).unwrap();
        assert_eq!(mip1.width, 480);
        assert_eq!(mip1.height, 270);
    }

    #[test]
    fn test_push_constants() {
        let mut pass = BloomPass::with_config(BloomConfig {
            threshold: 1.2,
            soft_knee: 0.3,
            ..Default::default()
        });
        pass.calculate_mip_chain(1920, 1080);

        let pc = pass.get_prefilter_push_constants();
        assert!((pc.threshold - 1.2).abs() < 0.001);
        assert!((pc.soft_knee - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_small_resolution() {
        let mut pass = BloomPass::with_config(BloomConfig {
            mip_count: 10, // More than possible
            ..Default::default()
        });
        pass.calculate_mip_chain(32, 32);

        // Stop before reaching 10 mips if resolution reaches 1x1.
        assert!(pass.mip_count() < 10);

        // Last mip should be 1x1 or close
        let last = pass.get_mip_info(pass.mip_count() - 1).unwrap();
        assert!(last.width <= 2 && last.height <= 2);
    }
}
