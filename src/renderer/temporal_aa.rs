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

/// TAA configuration
#[derive(Debug, Clone)]
pub struct TaaConfig {
    /// Enable/disable TAA
    pub enabled: bool,
    /// Blend factor (0.0 = current only, 1.0 = history only)
    pub blend_factor: f32,
    /// Enable color clamping to reduce ghosting
    pub color_clamp: bool,
    /// Enable velocity rejection
    pub velocity_rejection: bool,
    /// Jitter scale (typically 1.0)
    pub jitter_scale: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            blend_factor: 0.9,
            color_clamp: true,
            velocity_rejection: true,
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
    /// Blend factor, color clamp toggle, velocity rejection, padding
    pub params: [f32; 4],
    /// Current jitter offset
    pub jitter: [f32; 2],
    /// Previous jitter offset
    pub prev_jitter: [f32; 2],
}

impl Default for TaaPushConstants {
    fn default() -> Self {
        Self {
            screen_params: [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0],
            params: [0.9, 1.0, 1.0, 0.0],
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
}
