//! High-level renderer module.
//!
//! This module provides the main [`Renderer`] struct and all supporting types
//! for PBR rendering, materials, meshes, and textures.

pub mod cleanup_traits;
pub mod cluster_builder;
pub mod command_list;
pub mod diagnostics;
pub mod features;
// pub mod forward_plus_descriptor; // DELETED: Using BDA
pub mod forward_plus_integration;
pub mod frame_graph;
pub mod fullscreen_pass;
pub mod gbuffer;
pub mod hdr_framebuffer;
pub mod hiz_pass;
pub mod indirect_draw;
pub mod instancing;
pub mod light_culling_integration;
// pub mod lod_system; // Deleted for VCGS transition
pub mod model_renderer;
pub mod motion_pass;
pub mod passes;

pub mod occlusion_culling;
pub mod pipeline_cache;
pub mod render_graph;
pub mod render_stats;
pub mod renderer;
pub mod resource_pool;
pub mod resource_registry;
pub mod resources;
pub mod temporal_aa;
pub mod util;
pub mod vram_budget;
pub mod vsr_pass;

// Re-exports for public API
pub use cleanup_traits::{BufferCleanup, VulkanResourceCleanup};
pub use features::{AutoRotateFeature, FeatureManager, RenderFeature};
// pub use forward_plus_descriptor::ForwardPlusDescriptor; // DELETED
pub use forward_plus_integration::ForwardPlusIntegration;
pub use gbuffer::GBuffer;
pub use hiz_pass::HiZPass;
pub use indirect_draw::IndirectDrawPass;
pub use instancing::{InstanceData, InstancingManager};
pub use model_renderer::{MaterialPushConstants, ModelRenderer};
pub use motion_pass::MotionVectorPass;
pub use occlusion_culling::{CullBoundingBox, CullObjectData, OcclusionCulling};
pub use pipeline_cache::PipelineCache;
pub use render_stats::{RenderStats, StatsCollector};
pub use renderer::{DebugMode, RenderCommand, Renderer};
pub use resource_registry::{ResourceId, ResourceRegistry};
pub use vsr_pass::{VsrConfig, VsrPass, VsrQuality};

// Re-export from resources submodule
pub use resources::{
    BufferAllocation, BufferHandle, BufferPool, Camera, DepthBuffer, DescriptorSetHandle,
    ImageHandle, InstanceBuffer, Material, MaterialHandle, MaterialManager, Mesh, MvpMatrices,
    ObjectMotionData, PipelineHandle, TemporalCamera, Texture, TextureData, Transform,
    UniformBuffer, Vertex, MVP,
};
