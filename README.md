# Ash Renderer

[![Crates.io](https://img.shields.io/crates/v/ash_renderer.svg)](https://crates.io/crates/ash_renderer)
[![Documentation](https://docs.rs/ash_renderer/badge.svg)](https://docs.rs/ash_renderer)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Vulkan rendering library built with [ash](https://github.com/ash-rs/ash). This project explores modern graphics techniques (GPU culling, SSGI, Bindless resources) in a standalone, ECS-free architecture.

> [!NOTE]
> This is still very much a "work in progress." Expect breaking changes and occasional Vulkan validation errors if you feed it weird data.
> **Stable Versions:** 0.1.2, 0.3.8, 0.3.9, 0.4.0, 0.4.1.

## Features

- **Core Renderer**: Basic PBR metallic/roughness workflow.
- **Occlusion Culling**: Hi-Z based visibility testing (GPU driven).
- **GPU Culling**: Frustum culling and indirect draw call generation.
- **Lighting**: Cascaded Shadow Mapping (CSM) and Screen-Space Global Illumination (SSGI).
- **Bindless Architecture**: Full bindless texture support (`SampledImage` arrays).
- **Post-Processing**: Tonemapping, Bloom, and internal VSR (Temporal upscaling) support.
- **Headless**: Decoupled from windowing via `SurfaceProvider`.

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
                    let aspect = size.width as f32 / size.height as f32;
                    
                    // Simple camera setup
                    let view = glam::Mat4::look_at_rh(
                        glam::Vec3::new(0.0, 2.0, 5.0),
                        glam::Vec3::ZERO,
                        glam::Vec3::Y
                    );
                    let mut proj = glam::Mat4::perspective_rh(
                        45.0_f32.to_radians(),
                        aspect,
                        0.1,
                        100.0
                    );
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip
                    
                    r.render_frame(view, proj, glam::Vec3::new(0.0, 2.0, 5.0)).unwrap();
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
            _ => {}
        }
    }
}
```

## Status

| Feature | Status |
| :--- | :--- |
| **Material System** | Functional (Basic PBR) |
| **Shadows** | Working, but cascades need tuning |
| **SSGI** | Experimental (Expect noise) |
| **VSR (Temporal Upscaling)** | Implemented (basic jitter patterns, needs refinement) |
| **GLTF Loading** | Basic support via `gltf` crate |
| **Stability** | Dev-grade (Validation layers recommended during dev) |

## Examples

```bash
# Basic cube with PBR
cargo run --example 02_cube

# GLTF loading (experimental)
cargo run --example 03_model_loading --features gltf_loading
```

## Requirements

- **Rust**: 1.70+
- **Vulkan**: 1.2+ (Requires support for dynamic indexing and descriptor indexing)

---
Licensed under Apache 2.0.
