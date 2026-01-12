# Ash Renderer

[![Crates.io](https://img.shields.io/crates/v/ash_renderer.svg)](https://crates.io/crates/ash_renderer)
[![Documentation](https://docs.rs/ash_renderer/badge.svg)](https://docs.rs/ash_renderer)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Vulkan rendering library built with [ash](https://github.com/ash-rs/ash). This project explores modern graphics techniques (GPU culling, SSGI, Bindless resources) in a standalone, ECS-free architecture.

> [!NOTE]
> This is still very much a "work in progress." Expect breaking changes and occasional Vulkan validation errors if you feed it weird data.
> **Stable Versions:** 0.1.2, 0.3.8, 0.3.9, 0.4.0-0.4.9, 0.4.45, 0.4.84.

## Features

- **Core Renderer**: Basic PBR metallic/roughness workflow with automatic GLB material registration.
- **Refactored Asset Pipeline**: Decoupled loading using the [`archetype_asset`](https://github.com/saptak7777/Archetype-Asset) crate.
- **GLB Support**: Robust material registration from GLB files with PBR properties (metallic, roughness, emissive) via the `gltf_loader` utility.
- **Occlusion Culling**: Hi-Z based visibility testing (GPU driven).
- **GPU Culling**: Frustum culling and indirect draw call generation.
- **Lighting**: Cascaded Shadow Mapping (CSM), Screen-Space Global Illumination (SSGI), and IBL (Image-Based Lighting).
- **Bindless Architecture**: Full bindless texture support with 16,384 slots.
- **Post-Processing**: Tonemapping, Bloom, and internal VSR (Temporal upscaling) support.
- **GPU Skinning**: Linear blend skinning (LBS) with compute-based joint updates.
- **Buffer Safety**: Type-safe `BufferBuilder` API with runtime validation and allocation tracking.
- **Headless**: Decoupled from windowing via `SurfaceProvider`.
- **Advanced Diagnostics**: Real-time VRAM budgeting and nanosecond-precision GPU timestamp profiling.

## Quick Start (Winit 0.30)

The renderer is designed to be used with `winit`'s `ApplicationHandler`. Here is a minimal setup:

```rust
use ash_renderer::prelude::*;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(Default::default()).unwrap();
        
        // Wrap window for Vulkan surface
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);
        
        // Init renderer (handles device/swapchain internally)
        self.renderer = Some(Renderer::new(&surface_provider).expect("Vulkan init failed"));
        self.window = Some(window);
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                if let (Some(r), Some(w)) = (&mut self.renderer, &self.window) {
                    let size = w.inner_size();
                    let aspect = size.width as f32 / size.height.max(1) as f32;
                    
                    let camera_pos = glam::Vec3::new(0.0, 2.0, 5.0);
                    let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                    let mut proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip
                    
                    // Render the frame (no global transform override)
                    r.render_frame(view, proj, camera_pos, None).unwrap();
                    w.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.request_swapchain_resize(ash::vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    });
                }
            }
            WindowEvent::CloseRequested => _el.exit(),
            _ => {}
        }
    }
}
```

## Status

| Feature | Status |
| :--- | :--- |
| **Material System** | Functional (Full GLB PBR Support) |
| **Shadows** | Working (Cascaded Shadow Maps) |
| **SSGI** | Experimental (Expect noise) |
| **IBL** | Implemented (Supports irradiance and prefiltered maps) |
| **Temporal Upscaling** | Implemented (VSR/TAA patterns) |
| **GPU Skinning** | Stable (Double-buffered) |
| **Asset Loading** | Decoupled via `archetype_asset` and `gltf_loader` |

## Examples

```bash
# Basic cube with PBR
cargo run --example 02_cube

# GLTF loading (Default features include gltf_loading)
cargo run --example 03_model_loading
```

## API Usage: Asset Loading (Dumb Pipe)

Following the "Dumb Pipe" philosophy, loading is handled outside the core renderer.

```rust
// 1. Load meshes from GLB using the utility bridge
let meshes = gltf_loader::load_model("model.glb")?;

// 2. Upload mesh to GPU
let mesh = meshes.into_iter().next().unwrap();
let mesh_handle = renderer.upload_mesh(mesh)?;

// 3. Register a render command
renderer.submit_render_commands(&[RenderCommand {
    mesh_handle,
    material_handle: MaterialHandle::null(), // Use auto-detected material from registration
    transform: glam::Mat4::IDENTITY,
    ..Default::default()
}])?;
```

## API Usage: Buffer Creation

The `BufferBuilder` API makes creating GPU buffers explicit and safe:

```rust
// Using the fluent builder
let (buffer, allocation) = BufferBuilder::new(size)
    .storage_buffer()
    .cpu_writable()
    .named("My Custom Buffer")
    .build(renderer.allocator())?;
```

## Requirements

- **Rust**: 1.75+
- **Vulkan**: 1.2+ (Requires support for dynamic indexing and descriptor indexing)

---
Licensed under Apache 2.0.
