//! High-level renderer module.
//!
//! This module provides the main [`Renderer`] struct and all supporting types
//! for PBR rendering, materials, meshes, and textures.

pub mod cleanup_traits;
pub mod diagnostics;
pub mod features;
pub mod frame_graph;
pub mod fullscreen_pass;
pub mod hdr_framebuffer;
pub mod instancing;
pub mod light_culling_integration;
pub mod lod_system;
pub mod model_renderer;
pub mod msaa_targets;
pub mod occlusion_culling;
pub mod pipeline_cache;
pub mod render_stats;
#[allow(clippy::module_inception)]
pub mod renderer;
pub mod resource_registry;
pub mod resources;
pub mod shadow_map;

// Re-exports for public API
pub use cleanup_traits::{BufferCleanup, VulkanResourceCleanup};
pub use features::{AutoRotateFeature, FeatureManager, RenderFeature};
pub use instancing::{InstanceData, InstancingManager};
pub use lod_system::{LodManager, LodMesh, LodSelection};
pub use model_renderer::{MaterialPushConstants, ModelRenderer};
pub use msaa_targets::{MsaaColorTarget, MsaaDepthTarget};
pub use occlusion_culling::{CullBoundingBox, OcclusionCulling};
pub use pipeline_cache::PipelineCache;
pub use render_stats::{RenderStats, StatsCollector};
pub use renderer::{RenderCommand, Renderer};
pub use resource_registry::{ResourceId, ResourceRegistry};

// Re-export from resources submodule
pub use resources::{
    BufferAllocation, BufferHandle, BufferPool, Camera, CascadedShadowMap, DepthBuffer,
    DescriptorSetHandle, ImageHandle, Material, Mesh, MvpMatrices, PipelineHandle, Texture,
    TextureData, Transform, UniformBuffer, Vertex, VertexBuffer, MVP,
};
