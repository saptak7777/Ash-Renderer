//! Temporal Anti-Aliasing (TAA) System
//!
//! Provides high-quality anti-aliasing by blending the current frame with
//! previous frames using motion vectors and color clamping.
//!
//! # Features
//! - Halton jitter sequence
//! - Velocity buffer support
//! - Neighborhood color clamping
//! - Configurable blend factor

use glam::{Mat4, Vec2};

/// TAA quality preset (AAA-grade, Unreal TSR style)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaaQuality {
    /// Responsive (low latency, less stable)
    Responsive,
    /// Balanced (default)
    Balanced,
    /// Quality (most stable)
    Quality,
}

impl TaaQuality {
    /// Unreal's presets (tuned over years)
    pub const RESPONSIVE: Self = Self::Responsive;
    pub const BALANCED: Self = Self::Balanced;
    pub const QUALITY: Self = Self::Quality;

    /// Get history weight
    pub const fn history_weight(self) -> f32 {
        match self {
            Self::Responsive => 0.7,
            Self::Balanced => 0.85,
            Self::Quality => 0.95,
        }
    }

    /// Get clamping gamma (for AABB clipping)
    pub const fn clamping_gamma(self) -> f32 {
        match self {
            Self::Responsive => 1.5,
            Self::Balanced => 1.2,
            Self::Quality => 1.0,
        }
    }
}

impl Default for TaaQuality {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Sharpening mode (Unreal pattern)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SharpeningMode {
    /// No sharpening
    None,
    /// Subtle (0.2 strength)
    Subtle,
    /// Moderate (0.4 strength) - Unreal default
    Moderate,
    /// Strong (0.6 strength)
    Strong,
}

impl SharpeningMode {
    pub fn strength(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Subtle => 0.2,
            Self::Moderate => 0.4,
            Self::Strong => 0.6,
        }
    }
}

impl Default for SharpeningMode {
    fn default() -> Self {
        Self::Moderate
    }
}

/// TAA configuration
#[derive(Debug, Clone)]
pub struct TaaConfig {
    /// Enable/disable TAA
    pub enabled: bool,
    /// Quality preset (AAA-grade)
    pub quality: TaaQuality,
    /// Sharpening mode
    pub sharpening: SharpeningMode,
    /// Blend factor (0.0 = current only, 1.0 = history only)
    /// Note: Overridden by quality preset if not manually set
    pub blend_factor: f32,
    /// Enable color clamping to reduce ghosting
    pub color_clamp: bool,
    /// Enable velocity rejection
    pub velocity_rejection: bool,
    /// Velocity rejection threshold (higher = more responsive)
    pub velocity_threshold: f32,
    /// Depth rejection threshold
    pub depth_threshold: f32,
    /// Anti-flicker (Unreal TSR feature)
    pub anti_flicker: bool,
    /// Jitter scale (typically 1.0)
    pub jitter_scale: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        let quality = TaaQuality::default();
        Self {
            enabled: true,
            quality,
            sharpening: SharpeningMode::default(),
            blend_factor: quality.history_weight(),
            color_clamp: true,
            velocity_rejection: true,
            velocity_threshold: 0.02, // Unreal default
            depth_threshold: 0.1,     // Unreal default
            anti_flicker: true,
            jitter_scale: 1.0,
        }
    }
}

/// Halton sequence for jitter positions
pub struct HaltonSequence {
    index: u32,
}

impl HaltonSequence {
    pub fn new() -> Self {
        Self { index: 0 }
    }

    /// Generate next jitter offset in range [-0.5, 0.5]
    pub fn next_jitter(&mut self) -> Vec2 {
        let jitter = Vec2::new(
            Self::halton(self.index + 1, 2) - 0.5,
            Self::halton(self.index + 1, 3) - 0.5,
        );
        self.index = (self.index + 1) % 16;
        jitter
    }

    /// Halton sequence value
    fn halton(mut index: u32, base: u32) -> f32 {
        let mut f = 1.0f32;
        let mut r = 0.0f32;
        while index > 0 {
            f /= base as f32;
            r += f * (index % base) as f32;
            index /= base;
        }
        r
    }

    /// Reset sequence
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

impl Default for HaltonSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// TAA push constants for shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TaaPushConstants {
    /// Screen size (width, height, 1/width, 1/height)
    pub screen_params: [f32; 4],
    /// Blend factor, color clamp toggle, velocity rejection, clamping gamma
    pub params: [f32; 4],
    /// Velocity threshold, depth threshold, anti-flicker, padding
    pub quality_params: [f32; 4],
    /// Current jitter offset
    pub jitter: [f32; 2],
    /// Previous jitter offset
    pub prev_jitter: [f32; 2],
}

impl Default for TaaPushConstants {
    fn default() -> Self {
        Self {
            screen_params: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            params: [0.85, 1.0, 1.0, 1.2],
            quality_params: [0.02, 0.1, 1.0, 0.0],
            jitter: [0.0, 0.0],
            prev_jitter: [0.0, 0.0],
        }
    }
}

/// Temporal Anti-Aliasing manager
pub struct TemporalAA {
    config: TaaConfig,
    halton: HaltonSequence,
    current_jitter: Vec2,
    previous_jitter: Vec2,
    frame_index: u64,
}

impl TemporalAA {
    /// Create a new TAA manager
    pub fn new() -> Self {
        Self::with_config(TaaConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: TaaConfig) -> Self {
        Self {
            config,
            halton: HaltonSequence::new(),
            current_jitter: Vec2::ZERO,
            previous_jitter: Vec2::ZERO,
            frame_index: 0,
        }
    }

    /// Begin new frame - update jitter
    pub fn begin_frame(&mut self) {
        self.previous_jitter = self.current_jitter;
        self.current_jitter = self.halton.next_jitter() * self.config.jitter_scale;
        self.frame_index += 1;
    }

    /// Get jittered projection matrix
    pub fn jitter_projection(&self, projection: Mat4, width: u32, height: u32) -> Mat4 {
        if !self.config.enabled {
            return projection;
        }

        let jitter_x = self.current_jitter.x * 2.0 / width as f32;
        let jitter_y = self.current_jitter.y * 2.0 / height as f32;

        let mut jittered = projection;
        jittered.w_axis.x += jitter_x;
        jittered.w_axis.y += jitter_y;
        jittered
    }

    /// Get push constants for TAA resolve shader
    pub fn push_constants(&self, width: u32, height: u32) -> TaaPushConstants {
        TaaPushConstants {
            screen_params: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
            params: [
                self.config.blend_factor,
                if self.config.color_clamp { 1.0 } else { 0.0 },
                if self.config.velocity_rejection {
                    1.0
                } else {
                    0.0
                },
                self.config.quality.clamping_gamma(),
            ],
            quality_params: [
                self.config.velocity_threshold,
                self.config.depth_threshold,
                if self.config.anti_flicker { 1.0 } else { 0.0 },
                0.0,
            ],
            jitter: [self.current_jitter.x, self.current_jitter.y],
            prev_jitter: [self.previous_jitter.x, self.previous_jitter.y],
        }
    }

    /// Get current jitter
    pub fn current_jitter(&self) -> Vec2 {
        self.current_jitter
    }

    /// Is TAA enabled?
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get mutable config reference
    pub fn config_mut(&mut self) -> &mut TaaConfig {
        &mut self.config
    }

    /// Get config reference
    pub fn config(&self) -> &TaaConfig {
        &self.config
    }

    /// Reset history (call on camera cut or teleport)
    pub fn reset_history(&mut self) {
        self.halton.reset();
        self.current_jitter = Vec2::ZERO;
        self.previous_jitter = Vec2::ZERO;
    }
}

impl Default for TemporalAA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_sequence() {
        let mut halton = HaltonSequence::new();
        let j1 = halton.next_jitter();
        let j2 = halton.next_jitter();
        // Should be different
        assert_ne!(j1, j2);
        // Should be in range
        assert!(j1.x >= -0.5 && j1.x <= 0.5);
    }

    #[test]
    fn test_jittered_projection() {
        let taa = TemporalAA::new();
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let jittered = taa.jitter_projection(proj, 1920, 1080);
        // Initial jitter is zero, so should be same
        assert_eq!(proj, jittered);
    }

    #[test]
    fn test_taa_quality_presets() {
        // Responsive: low history weight, high clamping gamma
        assert_eq!(TaaQuality::Responsive.history_weight(), 0.7);
        assert_eq!(TaaQuality::Responsive.clamping_gamma(), 1.5);

        // Balanced: middle ground
        assert_eq!(TaaQuality::Balanced.history_weight(), 0.85);
        assert_eq!(TaaQuality::Balanced.clamping_gamma(), 1.2);

        // Quality: high history weight, low clamping gamma
        assert_eq!(TaaQuality::Quality.history_weight(), 0.95);
        assert_eq!(TaaQuality::Quality.clamping_gamma(), 1.0);
    }

    #[test]
    fn test_sharpening_modes() {
        assert_eq!(SharpeningMode::None.strength(), 0.0);
        assert_eq!(SharpeningMode::Subtle.strength(), 0.2);
        assert_eq!(SharpeningMode::Moderate.strength(), 0.4);
        assert_eq!(SharpeningMode::Strong.strength(), 0.6);
    }

    #[test]
    fn test_taa_config_defaults() {
        let config = TaaConfig::default();
        assert_eq!(config.quality, TaaQuality::Balanced);
        assert_eq!(config.sharpening, SharpeningMode::Moderate);
        assert_eq!(config.blend_factor, 0.85); // Balanced preset
        assert!(config.anti_flicker);
        assert_eq!(config.velocity_threshold, 0.02);
        assert_eq!(config.depth_threshold, 0.1);
    }
}
