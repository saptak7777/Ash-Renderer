//! # ASH Renderer
//!
//! A Vulkan rendering library built with ASH and VMA. This is an experimental renderer focusing on
//! modern techniques (GPU-driven culling, SSGI, VSR) in a standalone, ECS-free architecture.
//!
//! ## Status
//!
//! This crate is in early development. APIs are subject to frequent change.
//! It is primarily a testbed for advanced rendering features.
//!
//! **Stable Versions:** 0.1.2, 0.3.8, 0.3.9, 0.4.0, 0.4.1.
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

// Documentation coverage is a work-in-progress.
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

mod error;
pub mod renderer;
pub mod vulkan;

// Re-export dependencies for convenience/examples
pub extern crate vk_mem;

// Re-export public API
pub use error::{AshError, Result};

// Backwards compatibility alias
#[doc(hidden)]
pub use renderer::{
    Camera, DepthBuffer, MVP, Material, Mesh, PipelineCache, RenderStats, Renderer, ResourceId,
    ResourceRegistry, StatsCollector, Texture, TextureData, Transform, Vertex,
};

pub use renderer::features::{
    AutoRotateFeature, DirectionalLight, FeatureManager, PointLight, RenderFeature,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        AshError, Camera, DirectionalLight, Material, Mesh, PointLight, Renderer, Result, Texture,
        Transform, Vertex,
    };
}
