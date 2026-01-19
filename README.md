# Ash Renderer

### A Vulkan renderer that might actually draw something on your screen

[Crates.io](https://crates.io/crates/ash_renderer) | [Documentation](https://docs.rs/ash_renderer)

## What This Is (Probably)

Ash Renderer is a low-level Vulkan rendering library for Rust projects that want more control than a high-level engine but don't want to write 1,000 lines of boilerplate just to see a triangle. It uses a Forward+ approach, meaning it can technically handle more lights than your average basement, and relies on a "Dumb Pipe" philosophy—it mostly just renders what you give it.

> [!IMPORTANT]
> **Fair Warning**: This is a rendering component, not a game engine. You still have to handle your own physics, ECS, and logic. It's like buying a high-performance engine for a car you haven't built yet; it runs great on a test stand, but you can't drive it to the grocery store.

---

## Core Concepts

### What It Does (Mostly)

- ✅ **Forward+ Lighting**: Tile-based culling for hundreds of point and spot lights.
- ✅ **RAGE Hemisphere Ambient**: AAA-standard ambient model for physically plausible fill lighting.
- ✅ **Bindless Resources**: Up to 16,384 texture slots because who has time to bind things manually?
- ✅ **GPU-Driven Culling**: Hi-Z occlusion and frustum culling so your GPU doesn't melt.
- ✅ **PBR Workflow**: Metallic/Roughness standard, with automatic GLB material ingestion.
- ✅ **Async Readbacks**: Get data back from the GPU without stalling the whole pipeline (usually).
- ✅ **VSM (Virtual Shadow Maps)**: 16k+ resolution shadows with virtual memory paging. Replaced legacy PCF.
- ✅ **Post-Processing**: Bloom, Tonemapping, and a VSR (Temporal) implementation that's surprisingly okay.
- ✅ **Stability First**: Fixed the infamous `0xc000041d` sporadic crash. Descriptor leaks patched. Resize is rock solid.
- ✅ **Debug Visualization**: See exactly what's being culled with colored wireframes (Red=Gone, Green=Seen).
- ✅ **Shader Hot-Reload**: Iterate on compute shaders instantly (F5) without restarting.
- ✅ **Safe RHI**: No more raw pointers bro, I swear. We use `GpuBuffer<T>` now, very safe, very typed.
- ✅ **Render Graph**: Handles barriers automatically so I don't cry at night. Barriers merged also, performance stonks.
- ✅ **Parallel Everything**: Command recording on all cores? Yes. Culling on all cores? Yes. CPU fan go brrr? Also yes.
- ✅ **Async Transfers**: Data loading happens in background, no lag spike guarantee (mostly).

### What It Doesn't Do (Yet)

- ❌ **Physics/Collision**: That's your job.
- ❌ **Scene Graph**: We strictly draw what's submitted each frame.
- ❌ **Asset Management**: We provide loaders, but you decide where they live.
- ❌ **Audio**: Total silence.

---

## Installation

Add this to your `Cargo.toml` and pray to the Vulkan gods:

```toml
[dependencies]
ash_renderer = "0.4.86"
glam = "0.29" # Or whatever version we're using this week
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop.create_window(Default::default()).unwrap();
        
        // Wrap window for Vulkan surface
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);
        
        // Init renderer (Warning: May take a second to cook shaders)
        self.renderer = Some(Renderer::new(&surface_provider).expect("Vulkan forgot how to GPU"));
        self.window = Some(window);
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let WindowEvent::RedrawRequested = event {
            if let (Some(r), Some(w)) = (&mut self.renderer, &self.window) {
                let camera_pos = glam::Vec3::new(0.0, 2.0, 5.0);
                let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                let proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), 1.6, 0.1, 100.0);
                
                // Draw all the things
                r.render_frame(view, proj, camera_pos, None).unwrap();
            }
        }
    }
}
```

### 2. RAGE Ambient Lighting (The AAA Way)

We've ditched the legacy ambient color for a proper **Hemisphere Ambient** model (as seen in RAGE/GTA V). It uses a sky color and ground color to ensure your metals look good even in the shadows.

```rust
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};

// Build a preset or custom lighting setup (Compile-time verified!)
let lighting = LightingBuilder::new()
    .with_ambient_preset(AmbientPreset::OutdoorDay)
    .with_sun() // Adds a default global directional light
    .build();

renderer.set_lighting(&lighting);
```

### 3. Spotlights & Dynamic Gear

We also haven't ignored spotlights. You can update them like this:

```rust
let spot = SpotLight::new(
    Vec3::new(0.0, 10.0, 0.0), // Position
    Vec3::new(0.0, -1.0, 0.0), // Direction
    [1.0, 0.8, 0.6, 10.0],    // Color + Intensity
    25.0,                      // Range
    0.5,                       // Inner Angle (rads)
    0.8,                       // Outer Angle (rads)
);

// This tells the Forward+ culler about your new flashlight
renderer.update_spot_lights(&[spot]);
```

### 3. Buffer Building (The Fluent Way)

Creating buffers doesn't have to be a nightmare:

```rust
let (buffer, allocation) = BufferBuilder::new(1024)
    .storage_buffer()
    .cpu_writable()
    .named("My Not-A-Leak Buffer")
    .build(renderer.allocator())?;
```

### 4. Parallel Command Recording (The "Fastness" Way)

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
- `Renderer::new(provider)`: The constructor. Expects a `SurfaceProvider`.
- `render_frame(...)`: Call this every frame or nothing happens.
- `upload_mesh(mesh)`: Sends geometry to the GPU using new `GpuBuffer`, very type safe.
- `allocate_transient(...)`: Get temporary image for one frame. Automatic delete, no leak guarantee.
- `update_point_lights(...)`: For the spheres of light.
- `update_spot_lights(...)`: For the cones of light.

### Forward+ Pipeline
The magic that makes lights fast.
- Tile size is 16x16 by default.
- Depth pre-pass is technically integrated but we sometimes skip it if we're feeling lazy.

---

## Known Limitations

- **Only .glb files**: We really like GLTF. If you have OBJs, you'll need to convert them.
- **Vulkan 1.2+**: If your GPU is from the Victorian era, it might not support descriptor indexing.
- **Sparse Assets**: We don't handle streaming yet; everything you submit should be on the GPU.

---

## Troubleshooting

### "The screen is black!"
1. Did you submit any render commands?
2. Is your camera pointing at the floor?
3. Check the Vulkan validation layers. If they're screaming, listen to them.

### "It crashed during init!"
Usually means the driver doesn't support a required extension (like `descriptor_indexing`). We try to check, but sometimes we just crash. Oops.

---

## Contributing

PRs are welcome! If you find a bug, open an issue. I might fix it, or I might just document it here as a "feature."

---

## License

Licensed under the **Apache License, Version 2.0**.
Acknowledgment to: Unity DOTS, archetype_ecs, and the general Rust gamedev community.

---

*Using this in production? You're braver than I am. Let me know if it actually works.*
