//! # ASH Renderer
//!
//! A production-quality Vulkan renderer built with ASH (Vulkan bindings) and VMA (GPU memory allocator).
//!
//! ## Features
//!
//! - **PBR Materials**: Physically-based rendering with metallic/roughness workflow
//! - **Shadow Mapping**: PCF-filtered shadow maps with slope-scale bias
//! - **Post-Processing**: Bloom, tonemapping, and temporal anti-aliasing
//! - **GPU Profiling**: Built-in timing queries and performance diagnostics
//! - **Feature System**: Extensible plugin architecture for rendering features
//!
//! ## Quick Start
//!
//! ```ignore
//! use ash_renderer::{Renderer, Result};
//! use winit::window::Window;
//!
//! fn main() -> Result<()> {
//!     let window = create_window();
//!     let mut renderer = Renderer::new(&window)?;
//!
//!     // Main loop
//!     renderer.render_frame()?;
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into two main tiers:
//!
//! - **`vulkan`**: Low-level Vulkan abstractions (internal)
//! - **`renderer`**: High-level rendering API (public)

// TODO: Re-enable missing_docs once documentation is complete
// #![warn(missing_docs)]
#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

mod error;
pub mod renderer;
pub mod vulkan;

// Re-export public API
pub use error::{AshError, Result};

// Backwards compatibility alias
#[doc(hidden)]
pub use renderer::{
    Camera, DepthBuffer, Material, Mesh, PipelineCache, RenderStats, Renderer, ResourceId,
    ResourceRegistry, StatsCollector, Texture, TextureData, Transform, Vertex, MVP,
};

pub use renderer::features::{AutoRotateFeature, FeatureManager, RenderFeature};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        AshError, Camera, Material, Mesh, Renderer, Result, Texture, Transform, Vertex,
    };
}
