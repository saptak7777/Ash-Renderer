# ASH Renderer

[![Crates.io](https://img.shields.io/crates/v/ash_renderer.svg)](https://crates.io/crates/ash_renderer)
[![Documentation](https://docs.rs/ash_renderer/badge.svg)](https://docs.rs/ash_renderer)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A **production-quality Vulkan renderer** built with [ASH](https://github.com/ash-rs/ash) (Vulkan bindings) and [VMA](https://github.com/gwihlern-gp/vk-mem-rs) (GPU memory allocator).

**ECS-free, pure rendering engine** - decoupled camera and input handling, ready for any game engine.

## Features

- 🎨 **PBR Materials** - Physically-based rendering with metallic/roughness workflow
- 🌑 **Shadow Mapping** - Cascaded shadow maps with PCF filtering
- ✨ **Post-Processing** - Bloom, tonemapping, and temporal anti-aliasing
- 📊 **GPU Profiling** - Built-in timing queries and performance diagnostics
- 🔌 **Feature System** - Extensible plugin architecture for rendering features
- 🚀 **High Performance** - 60+ FPS @ 1080p with 1000+ objects
- 🔧 **LOD System** - Automatic level-of-detail management
- ⚡ **GPU Instancing** - Efficient batch rendering
- 👁️ **Occlusion Culling** - GPU-accelerated visibility testing
- 💡 **Light Culling** - Tiled/clustered forward rendering

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ash_renderer = "0.1.1"
glam = "0.29" # Required for math types
```

### Basic Usage

```

## Examples

```bash
# Simple triangle
cargo run --example 01_triangle

# Textured cube with materials
cargo run --example 02_cube

# GLTF model loading
cargo run --example 03_model_loading --features gltf_loading
```

## API Overview

### Renderer

```rust
// Create renderer
let mut renderer = Renderer::new(&window)?;

// Set mesh and material
renderer.set_mesh(Mesh::create_cube());
*renderer.material_mut() = Material {
    color: [1.0, 0.5, 0.2, 1.0],
    metallic: 0.5,
    roughness: 0.3,
    ..Default::default()
};

// Update camera (per frame)
// You can use the helper Camera struct or your own math library
let camera = Camera::default(aspect);
renderer.update_camera(
    camera.view_matrix(),
    camera.projection_matrix(),
    camera.position
);

// Render
renderer.render_frame()?;

// Handle resize
renderer.request_swapchain_resize(ash::vk::Extent2D { width, height });
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
