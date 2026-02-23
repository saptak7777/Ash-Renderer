# Ash Renderer

### A Vulkan renderer that might actually draw something on your screen

[Crates.io](https://crates.io/crates/ash_renderer) | [Documentation](https://docs.rs/ash_renderer)

## What This Is (Probably)

Ash Renderer is a low-level Vulkan rendering library for Rust projects that want more control than a high-level engine but don't want to write 1,000 lines of boilerplate just to see a triangle. It uses a Forward+ approach, meaning it can technically handle more lights than your average basement, and relies on a "Dumb Pipe" philosophy—it mostly just renders what you give it. 

Oh, and did I mention it's **FULL BDA** (Buffer Device Address) and **FULL BINDLESS**? Yeah, we went there. Your CPU will thank us for not constantly bothering it with buffer bindings and descriptor updates. It's basically on vacation while your GPU does all the heavy lifting.

![We Are Full BDA Bro](we%20are%20full%20bda%20bro.jpg)

> [!IMPORTANT]
> **Fair Warning**: This is a rendering component, not a game engine. You still have to handle your own physics, ECS, and logic. It's like buying a high-performance engine for a car you haven't built yet; it runs great on a test stand, but you can't drive it to the grocery store.

> [!NOTE]
> **Quick Apology**: The skinning system is temporarily disabled because it was using legacy LOD architecture. I'll implement modern skinning in the next update. Yes, I know this is annoying if you wanted animated characters. Sorry about that.

---

## 🎯 Architecture & Goals
Before you dive in, read our [Philosophy & Goals](GOALS.md) document. It explains why we do things the way we do (and why we delete legacy features without mercy).

---

## Core Concepts

### What It Does (Mostly)

- ✅ **Forward+ Lighting**: Tile-based culling for hundreds of point and spot lights.
- ✅ **RAGE Hemisphere Ambient**: AAA-standard ambient model for physically plausible fill lighting.
- ✅ **Bindless or Bust**: We don't bind descriptors per object. We bind the whole world once and index it like a boss. (Up to 16,384 slots).
- ✅ **FULL BDA (Buffer Device Address)**: Direct GPU memory access. No more descriptor binding gymnastics.
- ✅ **GPU-Driven Culling**: Hi-Z occlusion and frustum culling so your GPU doesn't melt.
- ✅ **PBR Workflow**: Metallic/Roughness standard, with automatic GLB material ingestion.
- ✅ **Targeted Readbacks**: We read back what matters (VSR metrics, headless screenshots) without stalling. The rest stays on the GPU.
- ✅ **VSM (Virtual Shadow Maps)**: Full implementation with 16k+ resolution shadows, virtual memory paging, clipmap cascades, and smart cache invalidation. No artifacts, just crispy shadows.
- ✅ **VCGS (Virtual Clustered Geometry System)**: Professional mesh clustering using `meshopt` for leaf generation and simplification. Spatial sorting (Morton Codes) ensures extremely high cache locality. Continuous, invisible LOD transitions.
- ✅ **Post-Processing**: Bloom, Tonemapping, and a VSR (Temporal) implementation that's surprisingly okay.
- ✅ **Stability First**: Fixed the infamous crashes. Descriptor leaks patched. Resize is rock solid.
- ✅ **Debug Visualization**: See exactly what's being culled with colored wireframes (Red=Gone, Green=Seen).
- ✅ **Shader Hot-Reload**: Iterate on compute shaders instantly (F5). (Disabled by default, enable via `config.watch_shaders = true`).
- ✅ **Safe RHI**: No more raw pointers bro, I swear. We use `GpuBuffer<T>` now, very safe, very typed.
- ✅ **Render Graph**: Handles barriers automatically so I don't cry at night. Barriers merged also, performance stonks.
- ✅ **Parallel Everything**: Command recording on all cores? Yes. Culling on all cores? Yes. CPU fan go brrr? Also yes.
- ✅ **Async Transfers**: Dedicated transfer queue. Data loading happens in background, no lag spike guarantee (mostly).
- ✅ **Frame Profiler**: Built-in GPU timing queries. Know exactly which pass is killing your framerate.
- ✅ **Pipeline Cache**: Shader compilation caching. The first load is slow; the second is instant.
- ✅ **Motion Vectors**: Full Motion Vector generation for TAA/VSR.
- ✅ **VRAM Budget**: We track usage so you don't OOM yourself.
- ✅ **Headless Mode**: Run without a window. Great for CI/CD or feeling like a hacker.

### What It Doesn't Do (Yet)

- ❌ **Physics/Collision**: That's your job.
- ❌ **Scene Graph**: We strictly draw what's submitted each frame.
- ❌ **Asset Management**: We provide loaders, but you decide where they live.
- ❌ **Audio**: Total silence.
- ❌ **Skinning/Animation**: Currently disabled due to legacy LOD architecture issues (see apology above).

---

## Installation

Add this to your `Cargo.toml` and pray to the Vulkan gods:

```toml
[dependencies]
ash_renderer = "0.5.25"
glam = "0.31" # Or whatever version we're using this week
```

---

## Quick Start

### 1. The Minimum Viable Boilerplate (Winit 0.30)

We use a `SurfaceProvider` trait to keep things flexible. If you're using `winit`, it looks something like this:

```rust
use ash_renderer::prelude::*;
use winit::application::ApplicationHandler;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};
use winit::event::WindowEvent;

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(Default::default()).unwrap();
        
        // Wrap window for Vulkan surface
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);
        
        // Init renderer (Warning: May take a second to cook shaders)
        self.renderer = Some(Renderer::builder()
            .with_vsync(true) // Explicit VSync control
            .build(&surface_provider)
            .expect("Vulkan forgot how to GPU"));
        self.window = Some(window);
        
        // Setup basic mesh and material
        if let Some(ref mut renderer) = self.renderer {
            let mut cube = Mesh::create_cube();
            let mesh_handle = renderer.upload_mesh(cube).unwrap();
            
            let material = Material {
                color: [0.8, 0.8, 0.8, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            };
            let material_handle = renderer.register_and_upload_material(material).unwrap();
            
            self.render_commands.push(ash_renderer::renderer::RenderCommand {
                mesh_handle,
                material_handle,
                transform: glam::Mat4::IDENTITY,
                ..Default::default()
            });
        }
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let WindowEvent::RedrawRequested = event {
            if let (Some(r), Some(w)) = (&mut self.renderer, &self.window) {
                let camera_pos = glam::Vec3::new(0.0, 2.0, 5.0);
                let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                let aspect = w.inner_size().width as f32 / w.inner_size().height as f32;
                let mut proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
                proj.y_axis.y *= -1.0;
                
                // Submit commands then render
                let _ = r.submit_render_commands(&self.render_commands);
                let _ = r.render_frame(view, proj, camera_pos, None);
            }
        }
    }
}
```

### 2. Modern Lighting System

Ash Renderer uses a standardized Image-Based Lighting (IBL) and Directional light system. Legacy hemisphere ambient models have been replaced with a high-performance, single-path architecture.

```rust
use ash_renderer::renderer::features::ambient_lighting::{LightingPresets, LightingBuilder};

// Use a pre-defined calibrated preset
let lighting = LightingPresets::OUTDOOR_DAY;

// Or build a custom configuration
let custom_lighting = LightingBuilder::new()
    .with_directional(
        Vec3::new(-0.5, -1.0, -0.5).normalize(),
        Vec3::new(1.0, 0.9, 0.8), // Warm sun color
        3.0,                      // Intensity
    )
    .build();

renderer.set_lighting_config(lighting);
```

### 3. GLB Model Loading (The Easy Way)

Loading GLB models is now ridiculously simple:

```rust
use ash_renderer::renderer::resources::gltf_loader;

// Load model from file
let meshes = gltf_loader::load_model("assets/models/car.glb")?;

// Upload first mesh
let mesh = meshes.into_iter().next().unwrap();
let mesh_handle = renderer.upload_mesh(mesh)?;

// Create render command
let command = ash_renderer::renderer::RenderCommand {
    mesh_handle,
    material_handle: ash_renderer::renderer::MaterialHandle::null(),
    transform: glam::Mat4::IDENTITY,
    ..Default::default()
};

renderer.submit_render_commands(&[command])?;
```

### 4. Buffer Building (The Fluent Way)

Creating buffers doesn't have to be a nightmare:

```rust
let (buffer, allocation) = BufferBuilder::new(1024)
    .storage_buffer()
    .cpu_writable()
    .named("My Not-A-Leak Buffer")
    .build(renderer.allocator())?;
```

### 5. Parallel Command Recording (The "Fastness" Way)

If you have many CPU cores (rich guy), use them all to record commands.

```rust
// Auto-switches to parallel if you have more than 4 passes
// Trust me, it works very fast.
recorder.record_parallel(cmd, pass_count, |idx, cmd| {
    // Record commands here, very thread safe
    Ok(())
})?;
```

---

## API Reference (The Important Bits)

### Renderer
The heavy lifter. You probably only need one.
- `Renderer::builder()`: The entry point. Use `.build(provider)` to create the renderer.
- `render_frame(view, proj, camera_pos, target)`: Call this every frame or nothing happens.
- `upload_mesh(mesh)`: Sends geometry to the GPU, returns handle.
- `register_and_upload_material(material)`: Registers and uploads PBR material.
- `submit_render_commands(commands)`: Submit render commands for the frame.
- `set_lighting(lighting)`: Set up RAGE hemisphere ambient + directional lights.
- `request_swapchain_resize(extent)`: Handle window resizing properly.

### RenderCommand
What you actually submit each frame:
- `mesh_handle`: Handle to uploaded mesh
- `material_handle`: Handle to uploaded material  
- `transform`: World transformation matrix
- `..Default::default()`: Other fields have sensible defaults

### Lighting
AAA-standard lighting setup:
- `LightingBuilder`: Fluent API for building lighting configurations
- `AmbientPreset`: Pre-configured hemisphere ambient setups (OutdoorDay, IndoorLit, etc.)
- `DirectionalLight`: Sun/moon style lighting
- `PointLight`: Spherical light sources
- `SpotLight`: Conical light sources

### VSM (Virtual Shadow Maps)
The shadow system that doesn't suck:
- `VsmFeature`: Complete VSM system with clipmaps and paging
- `default_vsm_config()`: Good starting configuration
- `high_quality_vsm_config()`: For when you really need crispy shadows
- `performance_vsm_config()`: For potatos

### Forward+ Pipeline
The magic that makes lights fast.
- Tile size is 16x16 by default.
- Automatic light culling and clustering.
- Supports hundreds of dynamic lights without melting your GPU.

---

## Known Limitations

- **Only .glb files**: We really like GLTF. If you have OBJs, you'll need to convert them.
- **Vulkan 1.2+**: If your GPU is from the Victorian era, it might not support descriptor indexing.
- **Sparse Assets**: We don't handle streaming yet; everything you submit should be on the GPU.
- **No Skinning**: Currently disabled due to legacy LOD architecture issues (see apology above).
- **No Animation**: Same story as skinning - will be back with modern implementation.

---

## Troubleshooting

### "The screen is black!"
1. Did you submit any render commands using `submit_render_commands()`?
2. Did you upload a mesh and material?
3. Is your camera pointing at the floor?
4. Check the Vulkan validation layers. If they're screaming, listen to them.

### "It crashed during init!"
Usually means the driver doesn't support a required extension (like `descriptor_indexing`). We try to check, but sometimes we just crash. Oops.

### "My GLB model doesn't load!"
1. Make sure the file path is correct
2. Check that the GLB contains actual meshes (not just animations)
3. Verify texture paths are embedded or accessible

### "VSM shadows are flickering!"
1. Try `high_quality_vsm_config()` instead of default
2. Check if your directional light is moving too fast
3. Consider increasing cache size in VSM config

---

## Examples (Actually Working)

- `01_triangle`: The classic "hello world" of graphics
- `02_cube`: Simple colored cube with basic PBR
- `03_model_loading`: Load and display a GLB model
- `08_car_model`: Full PBR car model with materials
- `09_basic_mesh`: Minimal example, great starting point

Run them with:
```bash
cargo run --example 01_triangle
cargo run --example 08_car_model --features "gltf_loading"
```

## Contributing

PRs are welcome! If you find a bug, open an issue. I might fix it, or I might just document it here as a "feature."

**Priority areas for contributions:**
- Modern skinning system (to fix the current limitation)
- More format support (OBJ, FBX, etc.)
- Asset streaming system
- Better documentation and examples

---

## The Real Reason This Exists

**WARNING**: This renderer is built on pure recklessness and a willingness to break everything in the name of progress. I don't care about backward compatibility. I don't care about your carefully crafted code. I care about making this the most badass renderer on the block.

Every update might:
- Rename all your favorite functions
- Change the entire API structure 
- Delete features you loved
- Add new features you didn't know you needed
- Make your existing code explode in spectacular ways

But here's the thing: it's constantly evolving. While other libraries are busy maintaining decade-old APIs for the sake of "stability," we're here pushing the limits of what's possible. Breaking changes aren't bugs—they're features.

Think of this as the opposite of enterprise software. We move fast and break things. If you want stability and predictability, go use something else. If you want cutting-edge rendering tech that evolves faster than your GPU drivers, welcome aboard.

Just remember to check the changelog before updating. Your code's life depends on it.

---

## License

Licensed under the **Apache License, Version 2.0**.
Acknowledgment to: Unity DOTS, archetype_ecs, and the general Rust gamedev community.

---

*Using this in production? You're braver than I am. Let me know if it actually works.*

