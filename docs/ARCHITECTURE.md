# ASH Renderer Architecture

## Overview

ASH Renderer is organized into two main tiers:

```
┌─────────────────────────────────────────────────┐
│                  Public API                      │
│  Renderer, Material, Mesh, Texture, Camera, etc │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              renderer/ (High-Level)              │
│  resources/ │ features/ │ diagnostics/          │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              vulkan/ (Low-Level)                 │
│  Instance, Device, Allocator, Pipeline, etc.    │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│           ash + vk-mem (Vulkan Bindings)        │
└─────────────────────────────────────────────────┘
```

## Module Structure

### `vulkan/` - Low-Level Abstractions

| File | Purpose |
|------|---------|
| `instance.rs` | Vulkan instance, extensions, debug messenger |
| `device.rs` | Physical/logical device selection |
| `allocator.rs` | VMA wrapper for GPU memory |
| `swapchain.rs` | Swapchain management |
| `pipeline.rs` | Graphics pipeline builder |
| `descriptor_manager.rs` | Descriptor set management |
| `command_manager.rs` | Command buffer recording |
| `sync.rs` | Semaphores, fences, frame sync |

### `renderer/` - High-Level API

| File | Purpose |
|------|---------|
| `renderer.rs` | Main `Renderer` struct |
| `mesh.rs` | Vertex data, GLTF loading |
| `texture.rs` | Image loading, mipmaps |
| `material.rs` | PBR material parameters |
| `transform.rs` | Camera, transforms, MVP |

### `renderer/features/` - Extensible Features

```rust
pub trait RenderFeature: Send + Sync {
    fn name(&self) -> &'static str;
    fn initialize(&mut self, ctx: &FeatureContext) -> Result<()>;
    fn prepare_frame(&mut self, ctx: &FrameContext) -> Result<()>;
    fn render(&self, ctx: &RenderContext) -> Result<()>;
}
```

Built-in features:
- `ShadowFeature` - Shadow mapping
- `AutoRotateFeature` - Demo rotation
- Post-processing (bloom, tonemapping)

## Key Patterns

### 1. Resource Registry

All GPU resources are tracked centrally for automatic cleanup:

```rust
let texture_id = registry.register_image(image)?;
// Resource automatically freed when registry drops
```

### 2. Push Constants for Matrices

Fast path for per-object transforms (no descriptor binding):

```rust
#[repr(C)]
pub struct MeshPushConstants {
    model: Mat4,
    view: Mat4,
    projection: Mat4,
}
```

### 3. Frame Pipelining

3 frames in flight to hide GPU latency:

```rust
const FRAMES_IN_FLIGHT: usize = 3;
```

## Shader Organization

```
shaders/
├── vert.glsl          # Main vertex shader
├── frag.glsl          # PBR fragment shader
├── shadow.vert/frag   # Shadow pass
└── postfx/            # Post-processing
    ├── bloom_*.frag
    └── tonemapping.frag
```
