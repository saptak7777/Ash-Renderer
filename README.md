# ASH Renderer

[![Crates.io](https://img.shields.io/crates/v/ash_renderer.svg)](https://crates.io/crates/ash_renderer)
[![Documentation](https://docs.rs/ash_renderer/badge.svg)](https://docs.rs/ash_renderer)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> [!IMPORTANT]
> **Version Stability Notice**
> The following versions are strictly verified as **STABLE and USABLE**:
> - **v0.4.0** (Latest, Featured with UE5-inspired tech)
> - **v0.3.9** (Previous stable)
> - **v0.1.2**
>
> All other versions/variants may contain critical bugs or instability. Stability is guaranteed only for recommended versions. Please stay on the recommended versions for a production-ready experience.

A **production-quality Vulkan renderer** built with [ASH](https://github.com/ash-rs/ash) (Vulkan bindings) and [VMA](https://github.com/gwihlern-gp/vk-mem-rs) (GPU memory allocator).

**ECS-free, pure rendering engine** - decoupled camera and input handling, ready for any game engine.

## Features

- 🎨 **PBR Materials** - Physically-based rendering with metallic/roughness workflow
- 🌑 **Shadow Mapping** - Cascaded shadow maps with PCF filtering
- ✨ **VSR (Vulkan Super Resolution)** - Next-gen temporal upscaling for high-fidelity performance
- 💡 **SSGI (Screen-Space Global Illumination)** - Dynamic, high-performance indirect lighting
- 👁️ **Hi-Z Occlusion Culling** - Hierarchy-based visibility testing for massive scenes
- ⚡ **Indirect Draw & GPU Culling** - Compute-based draw call optimization and frustum culling
- 📊 **GPU Profiling** - Built-in timing queries and performance diagnostics
- 🔌 **Feature System** - Extensible plugin architecture for rendering features
- 🚀 **High Performance** - 60+ FPS @ 4K with VSR enabled
- 🔧 **LOD System** - Automatic level-of-detail management
- ⚡ **GPU Instancing** - Efficient batch rendering
- 🔄 **Hot Reloading** - Automatic shader recompilation and pipeline recreation on file change
- 🛡️ **Robust Validation** - GPU-assisted validation with automatic fallback
- 📦 **Bindless Textures** - Efficient bindless texture management (1024+ textures)
- 🖥️ **Headless Support** - Decoupled rendering via `SurfaceProvider` trait
- 🎞️ **Motion Vectors** - Dedicated G-Buffer pass for high-quality temporal effects

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ash-renderer = "0.4.0"
glam = "0.30" # Required for math types
```

### Basic Usage

```rust
use ash_renderer::prelude::*;
use glam::{Mat4, Vec3};
// use winit::window::Window; // Assumed available from context

// 1. Initialization (inside your winit event loop)
// Wraps the window to provide a Vulkan surface
let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

// Renderer::new handles Vulkan instance, device, and swapchain creation.
let mut renderer = Renderer::new(&surface_provider)?;

// 2. Resource Setup
// Create a built-in primitive mesh
let cube = Mesh::create_cube();

// Define PBR material properties
let material = Material {
    color: [1.0, 0.5, 0.2, 1.0], // RGBA
    metallic: 0.5,
    roughness: 0.3,
    ..Default::default()
};

// Assign resources to the renderer
renderer.set_mesh(cube);
*renderer.material_mut() = material;

// 3. Render Loop (e.g., inside RedrawRequested)
let aspect_ratio = width as f32 / height as f32;

// Camera Setup
let camera_pos = Vec3::new(0.0, 2.0, 5.0);
let target = Vec3::ZERO;
let view = Mat4::look_at_rh(camera_pos, target, Vec3::Y);

// Projection Setup (Note: Vulkan requires Y-flip)
let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
proj.y_axis.y *= -1.0; 

// Render the frame with the current camera state
renderer.render_frame(view, proj, camera_pos)?;

// 4. Resize Handling
// Call this when the window is resized to recreate the swapchain
renderer.request_swapchain_resize(ash::vk::Extent2D {
    width: new_size.width,
    height: new_size.height,
});
```

### Mesh Creation

```rust
// Built-in primitives
let cube = Mesh::create_cube();
let sphere = Mesh::create_sphere(32, 16);
let plane = Mesh::create_plane();

// Custom mesh
let mesh = Mesh::new(vertices, indices);
```

### Materials

```rust
let material = Material {
    color: [1.0, 1.0, 1.0, 1.0],      // Base color (RGBA)
    metallic: 0.0,                     // 0.0 = dielectric, 1.0 = metal
    roughness: 0.5,                    // 0.0 = smooth, 1.0 = rough
    emissive: [0.0, 0.0, 0.0],        // Emission color
    ..Default::default()
};
```

## Examples

Run the provided examples to see the renderer in action:

```bash
# Simple triangle
cargo run --example 01_triangle

# Textured cube with materials (Basic Usage)
cargo run --example 02_cube

# GLTF model loading
cargo run --example 03_model_loading --features gltf_loading
```

## Architecture

```
ash_renderer/
├── src/
│   ├── vulkan/          # Low-level Vulkan abstractions
│   │   ├── device.rs    # Logical device management
│   │   ├── pipeline.rs  # Graphics/compute pipelines
│   │   ├── shader.rs    # Shader loading & reflection
│   │   └── ...
│   ├── renderer/        # High-level rendering API
│   │   ├── renderer.rs  # Main Renderer struct
│   │   ├── resources/   # GPU resources (mesh, texture, material)
│   │   ├── features/    # Extensible feature system
│   │   └── diagnostics/ # Profiling & debugging
│   └── shaders/         # GLSL shader sources
└── examples/            # Usage examples
```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| FPS @ 1080p | 60+ | ✅ |
| Objects | 1000+ | ✅ |
| Memory (idle) | < 200MB | ✅ |
| Frame time | < 16.6ms | ✅ |

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `validation` | Vulkan validation layers | ✅ |
| `gltf_loading` | GLTF model loading | ✅ |
| `shader_compilation` | Runtime shader compilation | ❌ |
| `profiling` | GPU profiling queries | ❌ |
| `parallel` | Parallel command recording | ❌ |

## Requirements

- **Rust**: 1.70+
- **Vulkan**: 1.2+ capable GPU
- **Vulkan SDK**: For validation layers (optional)

## Author

**Saptak Santra**

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

Made with ❤️ and Vulkan
