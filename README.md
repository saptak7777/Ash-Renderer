# Ash Renderer

[![Crates.io](https://img.shields.io/crates/v/ash_renderer.svg)](https://crates.io/crates/ash_renderer)
[![Documentation](https://docs.rs/ash_renderer/badge.svg)](https://docs.rs/ash_renderer)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A Vulkan rendering library built with [ash](https://github.com/ash-rs/ash). This project explores modern graphics techniques (GPU culling, SSGI, Bindless resources) in a standalone, ECS-free architecture.

> [!NOTE]
> This is still very much a "work in progress." Expect breaking changes and occasional Vulkan validation errors if you feed it weird data.
> **Stable Versions:** 0.1.2, 0.3.8, 0.3.9, 0.4.0, 0.4.1, 0.4.2, 0.4.3, 0.4.4, 0.4.9.

## Features

- **Core Renderer**: Basic PBR metallic/roughness workflow.
- **Occlusion Culling**: Hi-Z based visibility testing (GPU driven).
- **GPU Culling**: Frustum culling and indirect draw call generation.
- **Lighting**: Cascaded Shadow Mapping (CSM) and Screen-Space Global Illumination (SSGI).
- **Bindless Architecture**: Full bindless texture support with 16,384 slots.
- **Texture Compression**: CPU-side BC7 (albedo) and BC5 (normals) compression for 4x+ VRAM savings.
- **Post-Processing**: Tonemapping, Bloom, and internal VSR (Temporal upscaling) support.
- **GPU Skinning**: Linear blend skinning (LBS) with compute-based joint updates and double-buffering.
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
| **GPU Skinning** | Stable (Double-buffered, 1024 bone limit) |
| **GLTF Loading** | Basic support via `gltf` crate |
| **Stability** | Dev-grade (Validation layers recommended during dev) |

## Examples

```bash
# Basic cube with PBR
cargo run --example 02_cube

# GLTF loading (experimental)
cargo run --example 03_model_loading --features gltf_loading
```

## API Usage: Skeletal Animation

The skeletal animation API uses explicit updates for safety and performance. Note the `unsafe` requirement for buffer updates.

```rust
// 1. Update Joint Matrices (Unsafe because it writes directly to mapped GPU memory)
let joints: &[glam::Mat4] = ...; // Your calculated joint matrices
unsafe {
    renderer.update_joint_ssbo(joints).expect("Failed to update joints");
}

// 2. Draw Skinned Mesh
renderer.draw_skinned_mesh(
    mesh_handle,
    material_handle,
    transform_matrix,
    joint_offset, // Offset into the SSBO where this instance's joints begin
);
```

## Requirements

- **Rust**: 1.70+
- **Vulkan**: 1.2+ (Requires support for dynamic indexing and descriptor indexing)

---
Licensed under Apache 2.0.
