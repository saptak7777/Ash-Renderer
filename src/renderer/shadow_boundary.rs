//! Shadow boundary handling with AAA engine techniques
//!
//! Combines best practices from Unity, Unreal, RAGE, and id Tech 7.
//! Enhanced with Rust's compile-time guarantees for robust PCF sampling.

use bytemuck::{Pod, Zeroable};

/// Shadow boundary configuration with compile-time validation.
/// Ensures that PCF kernels never sample outside the valid shadow map data.
#[derive(Debug, Clone, Copy)]
pub struct ShadowBoundaryConfig {
    /// Shadow map resolution (must be power of 2)
    pub resolution: u32,

    /// PCF kernel radius in texels (e.g., 2 = 5x5 kernel if diameter is 5)
    /// Note: The project uses textureGather-based 4x4 PCF in some shaders,
    /// so radius=2 is a safe upper bound for 4x4 kernels.
    pub pcf_radius: u32,

    /// Safety margin in texels (extra buffer beyond PCF radius)
    pub safety_margin: u32,
}

impl ShadowBoundaryConfig {
    /// Create a new configuration with validation.
    ///
    /// # Panics
    /// Panics if resolution isn't a power of two or if the kernel is too large.
    pub const fn new(resolution: u32, pcf_radius: u32) -> Self {
        // Human Pattern: Initialization-time assertions represent non-negotiable invariants
        assert!(
            resolution.is_power_of_two(),
            "Shadow resolution must be power of 2"
        );
        assert!(resolution >= 512, "Shadow resolution must be at least 512");
        assert!(pcf_radius <= 4, "PCF radius too large (max 4)");

        Self {
            resolution,
            pcf_radius,
            safety_margin: 1, // Always keep 1 texel safety margin for float precision issues
        }
    }

    /// Calculate guard band in UV space [0,1].
    ///
    /// Formula: (PCF_RADIUS + SAFETY_MARGIN) / RESOLUTION
    pub const fn guard_band_uv(&self) -> f32 {
        let total_guard_texels = self.pcf_radius + self.safety_margin;
        (total_guard_texels as f32) / (self.resolution as f32)
    }

    /// Calculate minimum safe UV coordinate.
    pub const fn uv_min(&self) -> f32 {
        self.guard_band_uv()
    }

    /// Calculate maximum safe UV coordinate.
    pub const fn uv_max(&self) -> f32 {
        1.0 - self.guard_band_uv()
    }

    /// Calculate texel size in UV space.
    pub const fn texel_size_uv(&self) -> f32 {
        1.0 / (self.resolution as f32)
    }

    /// Validate that PCF samples will never exceed hardware bounds.
    pub const fn validate_pcf_bounds(&self) -> bool {
        let max_offset_uv = (self.pcf_radius as f32) * self.texel_size_uv();
        let min_safe = self.uv_min();
        let max_safe = self.uv_max();

        // Worst case: sample at uv_min with max_negative offset
        let min_sample = min_safe - max_offset_uv;
        // Worst case: sample at uv_max with max_positive offset
        let max_sample = max_safe + max_offset_uv;

        min_sample >= 0.0 && max_sample <= 1.0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShadowBoundaryParams {
    /// Minimum safe UV coordinate (x, y)
    pub uv_min: f32,
    /// Maximum safe UV coordinate (x, y)
    pub uv_max: f32,
    /// Texel size in UV space
    pub texel_size: f32,
}

impl Default for ShadowBoundaryParams {
    fn default() -> Self {
        Self {
            uv_min: 0.0,
            uv_max: 1.0,
            texel_size: 1.0 / 4096.0,
        }
    }
}

impl From<ShadowBoundaryConfig> for ShadowBoundaryParams {
    fn from(config: ShadowBoundaryConfig) -> Self {
        Self {
            uv_min: config.uv_min(),
            uv_max: config.uv_max(),
            texel_size: config.texel_size_uv(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_band_calculation() {
        let config = ShadowBoundaryConfig::new(4096, 2);
        // (2 + 1) / 4096 = 3 / 4096 ≈ 0.0007324
        let expected = 3.0 / 4096.0;
        assert!((config.guard_band_uv() - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pcf_bounds_validation() {
        let config = ShadowBoundaryConfig::new(4096, 2);
        assert!(config.validate_pcf_bounds());
    }

    #[test]
    #[should_panic(expected = "Shadow resolution must be power of 2")]
    fn test_invalid_resolution_panics() {
        let _ = ShadowBoundaryConfig::new(1000, 2);
    }

    #[test]
    fn test_params_conversion() {
        let config = ShadowBoundaryConfig::new(2048, 1);
        let params = ShadowBoundaryParams::from(config);

        assert!(params.uv_min > 0.0);
        assert!(params.uv_max < 1.0);
        assert!(params.texel_size > 0.0);
    }
}
