//! High-level renderer module.
//!
//! This module provides the main [`Renderer`] struct and all supporting types
//! for PBR rendering, materials, meshes, and textures.

pub mod async_readback;
pub mod atrous_denoiser;
pub mod cleanup_traits;
pub mod diagnostics;
pub mod features;
pub mod forward_plus_descriptor;
pub mod forward_plus_integration;
pub mod frame_graph;
pub mod fullscreen_pass;
pub mod gbuffer;
pub mod hdr_framebuffer;
pub mod hiz_pass;
pub mod indirect_draw;
pub mod instancing;
pub mod light_culling_integration;
pub mod lod_system;
pub mod model_renderer;
pub mod motion_pass;
pub mod msaa_targets;
pub mod occlusion_culling;
pub mod pass_manager;
pub mod pipeline_cache;
pub mod render_stats;
pub mod renderer;
pub mod resource_registry;
pub mod resources;
pub mod sdfgi_pass;
pub mod shadow_map;
pub mod skinned_vertex;
pub mod ssgi_pass;
pub mod temporal_aa;
pub mod temporal_upscaling;
pub mod vram_budget;

// Re-exports for public API
pub use async_readback::AsyncReadbackManager;
pub use cleanup_traits::{BufferCleanup, VulkanResourceCleanup};
pub use features::{AutoRotateFeature, FeatureManager, RenderFeature};
pub use forward_plus_descriptor::ForwardPlusDescriptor;
pub use forward_plus_integration::ForwardPlusIntegration;
pub use gbuffer::GBuffer;
pub use hiz_pass::HiZPass;
pub use indirect_draw::IndirectDrawPass;
pub use instancing::{InstanceData, InstancingManager};
pub use lod_system::{LodManager, LodMesh, LodSelection};
pub use model_renderer::{MaterialPushConstants, ModelRenderer};
pub use motion_pass::MotionVectorPass;
pub use msaa_targets::{MsaaColorTarget, MsaaDepthTarget};
pub use occlusion_culling::{CullBoundingBox, CullObjectData, OcclusionCulling};
pub use pass_manager::{RenderPassManager, RenderingMode};
pub use pipeline_cache::PipelineCache;
pub use render_stats::{RenderStats, StatsCollector};
pub use renderer::{RenderCommand, Renderer};
pub use resource_registry::{ResourceId, ResourceRegistry};
pub use sdfgi_pass::{SdfgiPass, SdfgiQuality};
pub use ssgi_pass::{SsgiPass, SsgiQuality};
pub use temporal_upscaling::{VsrPass, VsrQuality};

// Re-export from resources submodule
pub use resources::{
    BufferAllocation, BufferHandle, BufferPool, Camera, CascadedShadowMap, DepthBuffer,
    DescriptorSetHandle, ImageHandle, InstanceBuffer, Material, MaterialHandle, MaterialManager,
    Mesh, MvpMatrices, ObjectMotionData, PipelineHandle, TemporalCamera, Texture, TextureData,
    Transform, UniformBuffer, Vertex, VertexBuffer, MVP,
};
pub use skinned_vertex::SkinnedVertex;
