//! VCGS (Vulkan Cluster Graphics System)
//!
//! Modern GPU-driven rendering system featuring cluster culling,
//! indirect draw generation, and virtual geometry management.

pub mod builder;
pub mod culling;
pub mod indirect;

// Re-exports for internal use
pub use builder::build_mesh_dag;
pub use culling::{
    CullBoundingBox, CullObjectData, CullObjectDesc, CullingPushConstants, IndirectDrawCommand,
    OcclusionCulling, CULL_FLAG_CAST_SHADOWS, CULL_FLAG_ENABLED, CULL_FLAG_HIDDEN,
    CULL_FLAG_RECEIVE_SHADOWS, CULL_FLAG_TRANSPARENT, MAX_CULLABLE_OBJECTS,
};
pub use indirect::{IndirectDrawPass, MAX_INDIRECT_OBJECTS};
