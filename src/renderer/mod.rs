//! High-level renderer module.
//!
//! This module provides the main [`Renderer`] struct and all supporting types
//! for PBR rendering, materials, meshes, and textures.

pub mod assets;
pub mod cleanup_traits;
pub mod command_list;
pub mod context;
pub mod diagnostics;
pub mod features;
pub mod forward_plus_integration;
pub mod frame;
pub mod frame_manager;
pub mod init_types;
pub mod initialization;
pub mod instancing;
pub mod light_culling_integration;
pub mod model_renderer;
pub mod passes;
pub mod pipeline_cache;
pub mod queue;
pub mod render_pipeline;
pub mod render_stats;
pub mod renderer;
pub mod resource_pool;
pub mod resource_registry;
pub mod resources;
pub mod scene;
pub(crate) mod swapchain_manager;
pub mod systems;
pub mod types;
pub mod util;
pub mod vcgs;
pub mod vram_budget;

// Re-exports for public API
pub use assets::AssetManager;
pub use cleanup_traits::{BufferCleanup, VulkanResourceCleanup};
pub use features::{AutoRotateFeature, FeatureManager, RenderFeature};
pub use forward_plus_integration::ForwardPlusIntegration;
pub use instancing::{InstanceData, InstancingManager};
pub use model_renderer::{MaterialPushConstants, ModelRenderer};
pub use pipeline_cache::PipelineCache;
pub use render_pipeline::{GeometryRenderContext, RenderPipeline};
pub use render_stats::{RenderStats, StatsCollector};
pub use renderer::Renderer;
pub use resource_registry::{ResourceId, ResourceRegistry};
pub use scene::Scene;

// Pass-specific re-exports (if not covered by passes::*)
pub use passes::fullscreen::FullscreenPass;
pub use passes::hiz::HiZPass;
pub use passes::motion::MotionVectorPass;
pub use passes::temporal_aa::{
    detect_config_change, ConfigChangeType, ConfigMetrics, ConfigMetricsReport,
    ConfigValidationError, SharpeningMode, TaaConfig, Validate,
};
pub use passes::vsr::{SharpenConfig, VsrConfig, VsrInputs, VsrPass, VsrQuality, VsrUpscaleConfig};

// Resource and type re-exports
pub use resources::*;
pub use types::*;
pub use vcgs::*;
