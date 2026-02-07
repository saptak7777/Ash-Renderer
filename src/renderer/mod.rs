//! High-level renderer module.
//!
//! This module provides the main [`Renderer`] struct and all supporting types
//! for PBR rendering, materials, meshes, and textures.

pub mod cleanup_traits;
pub mod command_list;
pub mod diagnostics;
pub mod features;
pub mod forward_plus_integration;
pub mod frame_graph;
pub mod fullscreen_pass;
pub mod gbuffer;
pub mod hdr_framebuffer;
pub mod hiz_pass;
pub mod init_types;
pub mod initialization;
pub mod instancing;
pub mod light_culling_integration;
pub mod model_renderer;
pub mod motion_pass;
pub mod passes;

pub mod pipeline_cache;
pub mod render_graph;
pub mod render_stats;
pub mod renderer;
pub mod resource_pool;
pub mod resource_registry;
pub mod resources;
pub mod temporal_aa;
pub mod types;
pub mod util;
pub mod vcgs;
pub mod vram_budget;
pub mod vsr_pass;

// Re-exports for public API
pub use cleanup_traits::{BufferCleanup, VulkanResourceCleanup};
pub use features::{AutoRotateFeature, FeatureManager, RenderFeature};
pub use forward_plus_integration::ForwardPlusIntegration;
pub use gbuffer::GBuffer;
pub use hiz_pass::HiZPass;
pub use instancing::{InstanceData, InstancingManager};
pub use model_renderer::{MaterialPushConstants, ModelRenderer};
pub use motion_pass::MotionVectorPass;
pub use pipeline_cache::PipelineCache;
pub use render_stats::{RenderStats, StatsCollector};
pub use renderer::Renderer;
pub use resource_registry::{ResourceId, ResourceRegistry};
pub use resources::*;
pub use temporal_aa::{
    detect_config_change, ConfigChangeType, ConfigMetrics, ConfigMetricsReport,
    ConfigValidationError, SharpeningMode, TaaConfig, Validate,
};
pub use types::*;
pub use vcgs::*;
pub use vsr_pass::{SharpenConfig, VsrConfig, VsrInputs, VsrPass, VsrQuality, VsrUpscaleConfig};
