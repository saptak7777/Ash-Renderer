//! RAGE-style Hemisphere Ambient Lighting
//!
//! Provides a type-safe ambient lighting system using the hemisphere ambient
//! method popularized by the RAGE engine (GTA V, RDR2).

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::marker::PhantomData;

/// Ambient lighting preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbientPreset {
    IndoorDark,
    IndoorLit,
    OutdoorDay,
    OutdoorNight,
    Custom,
}

impl AmbientPreset {
    pub const fn sky_color(self) -> Vec3 {
        match self {
            Self::IndoorDark => Vec3::new(0.02, 0.03, 0.04),
            Self::IndoorLit => Vec3::new(0.04, 0.045, 0.05),
            Self::OutdoorDay => Vec3::new(0.5, 0.6, 0.8),
            Self::OutdoorNight => Vec3::new(0.01, 0.015, 0.03),
            Self::Custom => Vec3::ZERO,
        }
    }

    pub const fn ground_color(self) -> Vec3 {
        match self {
            Self::IndoorDark => Vec3::new(0.01, 0.01, 0.01),
            Self::IndoorLit => Vec3::new(0.02, 0.02, 0.02),
            Self::OutdoorDay => Vec3::new(0.2, 0.25, 0.15),
            Self::OutdoorNight => Vec3::new(0.005, 0.005, 0.01),
            Self::Custom => Vec3::ZERO,
        }
    }

    pub const fn intensity(self) -> f32 {
        match self {
            Self::IndoorDark => 0.3,
            Self::IndoorLit => 0.5,
            Self::OutdoorDay => 1.0,
            Self::OutdoorNight => 0.2,
            Self::Custom => 1.0,
        }
    }
}

/// Hemisphere ambient lighting (GPU-aligned)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HemisphereAmbient {
    pub sky_color: [f32; 4],
    pub ground_color: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<HemisphereAmbient>() == 32);
const _: () = assert!(std::mem::align_of::<HemisphereAmbient>() == 16);

impl HemisphereAmbient {
    pub const fn from_preset(preset: AmbientPreset) -> Self {
        let sky = preset.sky_color();
        let ground = preset.ground_color();
        let intensity = preset.intensity();

        Self {
            sky_color: [sky.x, sky.y, sky.z, intensity],
            ground_color: [ground.x, ground.y, ground.z, 0.0],
        }
    }

    pub fn custom(sky_color: Vec3, ground_color: Vec3, intensity: f32) -> Self {
        Self {
            sky_color: [sky_color.x, sky_color.y, sky_color.z, intensity],
            ground_color: [ground_color.x, ground_color.y, ground_color.z, 0.0],
        }
    }
}

impl Default for HemisphereAmbient {
    fn default() -> Self {
        Self::from_preset(AmbientPreset::IndoorLit)
    }
}

/// Directional light (sun/moon)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DirectionalLight {
    pub direction: [f32; 4],
    pub color_intensity: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<DirectionalLight>() == 32);
const _: () = assert!(std::mem::align_of::<DirectionalLight>() == 16);

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        let dir = direction.normalize();
        Self {
            direction: [dir.x, dir.y, dir.z, 1.0],
            color_intensity: [color.x, color.y, color.z, intensity],
        }
    }

    pub const fn sun() -> Self {
        Self {
            direction: [0.3, -0.9, 0.2, 1.0],
            color_intensity: [1.0, 0.98, 0.95, 1.5],
        }
    }

    pub const fn moon() -> Self {
        Self {
            direction: [0.2, -0.8, 0.3, 1.0],
            color_intensity: [0.6, 0.7, 0.9, 0.3],
        }
    }
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self::sun()
    }
}

impl SceneLighting {
    // IBL indices are -1 for None
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SceneLighting {
    pub ambient: HemisphereAmbient,
    pub directional: DirectionalLight,
    pub point_light_count: u32,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub tile_size: u32,
    pub ibl_irradiance_index: i32, // -1 = None
    pub ibl_prefilter_index: i32,  // -1 = None
    pub ibl_brdf_lut_index: i32,   // -1 = None
    pub ibl_intensity: f32,
}

const _: () = assert!(std::mem::size_of::<SceneLighting>() == 96);
const _: () = assert!(std::mem::align_of::<SceneLighting>() == 16);

/// Type-state builder for SceneLighting
pub struct NoAmbient;
pub struct AmbientSet;
pub struct DirectionalSet;

pub struct LightingBuilder<State = NoAmbient> {
    ambient: HemisphereAmbient,
    directional: DirectionalLight,
    _state: PhantomData<State>,
}

impl LightingBuilder<NoAmbient> {
    pub fn new() -> Self {
        Self {
            ambient: HemisphereAmbient::default(),
            directional: DirectionalLight::default(),
            _state: PhantomData,
        }
    }

    pub fn with_ambient_preset(self, preset: AmbientPreset) -> LightingBuilder<AmbientSet> {
        LightingBuilder {
            ambient: HemisphereAmbient::from_preset(preset),
            directional: self.directional,
            _state: PhantomData,
        }
    }

    pub fn with_ambient_custom(
        self,
        sky: Vec3,
        ground: Vec3,
        intensity: f32,
    ) -> LightingBuilder<AmbientSet> {
        LightingBuilder {
            ambient: HemisphereAmbient::custom(sky, ground, intensity),
            directional: self.directional,
            _state: PhantomData,
        }
    }
}

impl Default for LightingBuilder<NoAmbient> {
    fn default() -> Self {
        Self::new()
    }
}

impl LightingBuilder<AmbientSet> {
    pub fn with_sun(self) -> LightingBuilder<DirectionalSet> {
        LightingBuilder {
            ambient: self.ambient,
            directional: DirectionalLight::sun(),
            _state: PhantomData,
        }
    }

    pub fn with_moon(self) -> LightingBuilder<DirectionalSet> {
        LightingBuilder {
            ambient: self.ambient,
            directional: DirectionalLight::moon(),
            _state: PhantomData,
        }
    }

    pub fn with_directional(
        self,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
    ) -> LightingBuilder<DirectionalSet> {
        LightingBuilder {
            ambient: self.ambient,
            directional: DirectionalLight::new(direction, color, intensity),
            _state: PhantomData,
        }
    }
}

impl LightingBuilder<DirectionalSet> {
    pub fn build(self) -> SceneLighting {
        SceneLighting {
            ambient: self.ambient,
            directional: self.directional,
            point_light_count: 0,
            num_tiles_x: 0,
            num_tiles_y: 0,
            tile_size: 16,
            ibl_irradiance_index: -1,
            ibl_prefilter_index: -1,
            ibl_brdf_lut_index: -1,
            ibl_intensity: 0.0,
        }
    }
}

/// Compile-time lighting presets
pub struct LightingPresets;

impl LightingPresets {
    pub const INDOOR_DARK: SceneLighting = SceneLighting {
        ambient: HemisphereAmbient::from_preset(AmbientPreset::IndoorDark),
        directional: DirectionalLight::sun(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 0.0,
    };

    pub const INDOOR_LIT: SceneLighting = SceneLighting {
        ambient: HemisphereAmbient::from_preset(AmbientPreset::IndoorLit),
        directional: DirectionalLight::sun(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 0.0,
    };

    pub const OUTDOOR_DAY: SceneLighting = SceneLighting {
        ambient: HemisphereAmbient::from_preset(AmbientPreset::OutdoorDay),
        directional: DirectionalLight::sun(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 0.0,
    };

    pub const OUTDOOR_NIGHT: SceneLighting = SceneLighting {
        ambient: HemisphereAmbient::from_preset(AmbientPreset::OutdoorNight),
        directional: DirectionalLight::moon(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 0.0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_preset_sun() {
        let lighting = LightingBuilder::new()
            .with_ambient_preset(AmbientPreset::OutdoorDay)
            .with_sun()
            .build();

        assert_eq!(lighting.ambient.sky_color[2], 0.8); // Blue
        assert_eq!(lighting.directional.direction[1], -0.9); // Sun angle
    }

    #[test]
    fn test_builder_custom() {
        let lighting = LightingBuilder::new()
            .with_ambient_custom(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.5)
            .with_directional(Vec3::new(0.0, -1.0, 0.0), Vec3::ONE, 1.0)
            .build();

        assert_eq!(lighting.ambient.sky_color[0], 1.0);
        assert_eq!(lighting.ambient.ground_color[1], 1.0);
        assert_eq!(lighting.ambient.sky_color[3], 0.5); // Intensity
        assert_eq!(lighting.directional.direction[1], -1.0);
        assert_eq!(lighting.directional.direction[3], 1.0); // Shadow enabled by default
    }

    #[test]
    fn test_compile_time_consts() {
        // Just verify they exist and match expected values
        let preset = LightingPresets::OUTDOOR_NIGHT;
        assert_eq!(preset.ambient.sky_color[3], 0.2); // Intensity
    }
}
