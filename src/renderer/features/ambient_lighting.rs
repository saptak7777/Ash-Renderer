//! Modern Lighting System (IBL & Directional)
//!
//! Provides a type-safe lighting system focused on modern Image-Based Lighting
//! and Directional lights. Legacy Hemisphere Ambient has been removed.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::marker::PhantomData;

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

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SceneLighting {
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

const _: () = assert!(std::mem::size_of::<SceneLighting>() == 64);
const _: () = assert!(std::mem::align_of::<SceneLighting>() == 16);

/// Type-state builder for SceneLighting
pub struct NoDirectional;
pub struct DirectionalSet;

pub struct LightingBuilder<State = NoDirectional> {
    directional: DirectionalLight,
    _state: PhantomData<State>,
}

impl LightingBuilder<NoDirectional> {
    pub fn new() -> Self {
        Self {
            directional: DirectionalLight::default(),
            _state: PhantomData,
        }
    }

    pub fn with_sun(self) -> LightingBuilder<DirectionalSet> {
        LightingBuilder {
            directional: DirectionalLight::sun(),
            _state: PhantomData,
        }
    }

    pub fn with_moon(self) -> LightingBuilder<DirectionalSet> {
        LightingBuilder {
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
            directional: DirectionalLight::new(direction, color, intensity),
            _state: PhantomData,
        }
    }
}

impl Default for LightingBuilder<NoDirectional> {
    fn default() -> Self {
        Self::new()
    }
}

impl LightingBuilder<DirectionalSet> {
    pub fn build(self) -> SceneLighting {
        SceneLighting {
            directional: self.directional,
            point_light_count: 0,
            num_tiles_x: 0,
            num_tiles_y: 0,
            tile_size: 16,
            ibl_irradiance_index: -1,
            ibl_prefilter_index: -1,
            ibl_brdf_lut_index: -1,
            ibl_intensity: 1.0,
        }
    }
}

/// Compile-time lighting presets
pub struct LightingPresets;

impl LightingPresets {
    pub const INDOOR_DARK: SceneLighting = SceneLighting {
        directional: DirectionalLight::sun(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 1.0,
    };

    pub const INDOOR_LIT: SceneLighting = SceneLighting {
        directional: DirectionalLight::sun(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 1.0,
    };

    pub const OUTDOOR_DAY: SceneLighting = SceneLighting {
        directional: DirectionalLight::sun(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 1.0,
    };

    pub const OUTDOOR_NIGHT: SceneLighting = SceneLighting {
        directional: DirectionalLight::moon(),
        point_light_count: 0,
        num_tiles_x: 0,
        num_tiles_y: 0,
        tile_size: 16,
        ibl_irradiance_index: -1,
        ibl_prefilter_index: -1,
        ibl_brdf_lut_index: -1,
        ibl_intensity: 1.0,
    };
}
