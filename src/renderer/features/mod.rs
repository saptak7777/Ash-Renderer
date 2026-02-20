pub mod ambient_lighting;
mod auto_rotate;
pub mod bloom;
mod feature_trait;
pub mod light_culling;
pub mod light_manager;
pub mod lighting;
pub mod post_processing;
pub mod tonemapping;
pub mod vsm;

pub use ambient_lighting::{
    AmbientPreset, HemisphereAmbient, LightingBuilder, LightingPresets, SceneLighting,
};
pub use auto_rotate::AutoRotateFeature;
pub use bloom::{BloomConfig, BloomFeature, BloomPass, BloomPushConstants};
pub use feature_trait::{FeatureFrameContext, FeatureManager, FeatureRenderContext, RenderFeature};
pub use light_culling::{
    GpuLight, LightCullingConfig, LightCullingPass, MAX_LIGHTS, MAX_LIGHTS_PER_TILE, TILE_SIZE,
};
pub use light_manager::{ForwardPlusInfo, LightManager};
pub use lighting::{DirectionalLight, LightingConfig, LightingFeature, PointLight, SpotLight};
pub use post_processing::{PostProcessingConfig, PostProcessingFeature};
pub use tonemapping::{TonemapOperator, TonemappingConfig, TonemappingFeature};
pub use vsm::{
    default_vsm_config, high_quality_vsm_config, performance_vsm_config, VsmConfig, VsmManager,
};
