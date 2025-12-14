mod auto_rotate;
pub mod bloom;
mod feature_trait;
pub mod light_culling;
pub mod lighting;
pub mod post_processing;
pub mod shadows;
pub mod tonemapping;

pub use auto_rotate::AutoRotateFeature;
pub use bloom::{BloomConfig, BloomFeature};
pub use feature_trait::{FeatureFrameContext, FeatureManager, FeatureRenderContext, RenderFeature};
pub use light_culling::{
    GpuLight, LightCullingConfig, LightCullingPass, MAX_LIGHTS, MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub use lighting::{DirectionalLight, LightingConfig, LightingFeature, PointLight};
pub use post_processing::{PostProcessingConfig, PostProcessingFeature};
pub use shadows::ShadowFeature;
pub use tonemapping::{TonemapOperator, TonemappingConfig, TonemappingFeature};
