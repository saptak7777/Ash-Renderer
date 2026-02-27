//! High-level renderer module.
//!
//! This module provides the main [`Renderer`] struct and all supporting types
//! for PBR rendering, materials, meshes, and textures.

pub mod assets;
pub mod builder;
pub mod cleanup_traits;
pub mod context;
pub mod diagnostics;
pub mod features;
pub mod frame;
pub mod frame_manager;
pub mod init_types;
pub mod initialization;
pub mod instancing;
pub mod lighting;
pub mod model_renderer;
pub mod passes;
pub mod pipeline_cache;
pub mod queue;
pub mod render_pipeline;
pub mod render_stats;
pub mod renderer;
pub mod resource_registry;
pub mod resources;
pub mod scene;
pub mod shading;
pub(crate) mod swapchain_manager;
pub mod sync;
pub use sync::FramePreparationInfo;
pub mod systems;
pub mod types;
pub mod util;
pub mod vcgs;
pub mod vram_budget;

// Re-exports for public API
pub use assets::AssetManager;
pub use cleanup_traits::{BufferCleanup, VulkanResourceCleanup};
pub use features::{AutoRotateFeature, FeatureManager, RenderFeature};
pub use instancing::{InstanceData, InstancingManager};
pub use model_renderer::{MaterialPushConstants, ModelRenderer};
pub use pipeline_cache::PipelineCache;
pub use render_pipeline::{GeometryRenderContext, RenderPipeline};
pub use render_stats::{RenderStats, StatsCollector};
pub use renderer::Renderer;
pub use resource_registry::{ResourceId, ResourceRegistry};
pub use scene::{MeshUploadInfo, Scene};
pub use shading::forward_plus::ForwardPlusIntegration;

// Pass-specific re-exports (if not covered by passes::*)
pub use passes::fullscreen::FullscreenPass;
pub use passes::hiz::HiZPass;
pub use passes::motion::MotionVectorPass;
pub use passes::temporal_aa::{
    ConfigChangeType, ConfigMetrics, ConfigMetricsReport, ConfigValidationError, SharpeningMode,
    TaaConfig, Validate, detect_config_change,
};

// Resource and type re-exports
pub use resources::*;
pub use types::*;
pub use vcgs::*;
