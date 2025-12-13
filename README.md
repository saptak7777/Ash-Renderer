# ASH Renderer

[![Crates.io](https://img.shields.io/crates/v/ash_renderer.svg)](https://crates.io/crates/ash_renderer)
[![Documentation](https://docs.rs/ash_renderer/badge.svg)](https://docs.rs/ash_renderer)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/saptak7777/Ash-Renderer/actions/workflows/ci.yml/badge.svg)](https://github.com/saptak7777/Ash-Renderer/actions)

A **production-read Vulkan renderer** built with [ASH](https://github.com/ash-rs/ash) and [VMA](https://github.com/gwihlern-gp/vk-mem-rs).
Designed for high-performance games and graphics applications, featuring ECS-independence and deep GPU optimization.

## ✨ Features (v0.2.6)

> [!IMPORTANT]
> **Release 0.2.6** fixes a critical regression in **v0.2.5** where resizing the window caused a panic due to resource dependency tracking. It is highly recommended to update.

### Check out what's new!
- **🐛 Fixed Resize Crash**: Resolved a "Render Pass Dependency" panic by correctly managing pipeline dependencies.
- **🔒 Safer Shader Loading**: (From v0.2.5) Fixed memory alignment issues for embedded shaders.


- **🎨 Bindless Texturing**: Fully dynamic texture access using `descriptor_indexing`. Supports thousands of textures with zero binding overhead.
- **🖥️ Headless Support**: Run heavy rendering workloads or benchmarks on CI without a window (virtual swapchain).
- **🌑 Advanced Shadows**: Cascaded Shadow Maps (CSM) with PCF filtering and light culling.

### Core Capabilities
- **PBR Materials**: Metallic/Roughness workflow with texture mapping.
- **Post-Processing**: Integrated Bloom, Tonemapping (ACES), and TAA.
- **GPU Profiling**: Built-in Diagnostics Overlay and timestamp queries.
- **Hot-Reloading**: Detects shader changes at runtime (experimental).
- **Cross-Platform**: Runs on **Windows** (Win32), **Linux** (X11/Wayland), and **macOS** (Metal) via a unified `SurfaceProvider`.

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ash_renderer = "0.2.6"
winit = "0.30"
glam = "0.30"
```

### Initialization

```rust
use ash_renderer::{Renderer, WindowSurfaceProvider};
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let window = winit::window::Window::builder().build(&event_loop)?;
    
    // Create Renderer with cross-platform SurfaceProvider
    let surface_provider = WindowSurfaceProvider::new(&window, 800, 600);
    let mut renderer = Renderer::new(surface_provider)?;

    // Load assets...
    let mesh = Mesh::create_cube();
    renderer.set_mesh(mesh);

    // Render loop
    event_loop.run(move |event, _, control_flow| {
        // ... handling logic ...
        if let Event::MainEventsCleared = event {
            renderer.render_frame(view, projection, camera_pos).unwrap();
        }
    })?;
    Ok(())
}
```

## 🔌 Headless Benchmarking

Run graphics benchmarks in purely headless mode (no window required):

```rust
use ash_renderer::{Renderer, HeadlessSurfaceProvider};

// Initialize without winit/window
let provider = HeadlessSurfaceProvider::new(1920, 1080);
let mut renderer = Renderer::new(provider)?;

// Loop as fast as GPU allows (no VSync)
for _ in 0..1000 {
    renderer.render_frame(view, proj, cam_pos)?;
}
```

## 🛠️ Performance

| Metric | Target | Achieved (v0.2.6) |
|--------|--------|-------------------|
| Draw Calls (Bindless) | 10k+ | ✅ |
| Headless FPS | Unlocked | ✅ |
| Texture Switches | Zero Cost | ✅ |

## 📦 Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `validation` | Enables Vulkan Validation Layers | ✅ |
| `gltf_loading` | Support for loading .gltf/.glb models | ✅ |
| `profiling` | Enables GPU timestamp queries | ❌ |

## 📜 License

Licensed under Apache-2.0.

### Troubleshooting (Windows)
> [!WARNING]
> If you encounter `exit code: 0xc000041d` on Windows, this is a **Fatal User Callback Exception**.
> 
> Possible causes and fixes:
> 1. **Overlay interference**: Disable Discord/Steam/NVIDIA overlays.
> 2. **Driver Hooks**: Update GPU drivers or verify no other software hooks `vulkan-1.dll`.
> 3. **Validation Layers**: The application enables validation layers by default. If the crash persists, use `--no-default-features` (requires `gltf_loading`).
