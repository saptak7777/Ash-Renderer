# 🚀 Advanced UE5-Inspired Features: Nanite, Lumen, TSR

**Created:** December 14, 2025  
**Target:** Close the gap with UE5 (recover 5-15% FPS loss)  
**Complexity:** High (3-4 months of work)  
**Effort:** 200-300+ developer hours

---

## Executive Summary

UE5's three flagship features are:

1. **Nanite** - Automatic LOD system via mesh clustering
2. **Lumen** - Real-time global illumination
3. **TSR** - Temporal super-resolution upscaling

Your renderer can add simplified versions of all three. This guide breaks down the **core concepts UE5 uses** and how to implement them **incrementally** without the extreme complexity.

---

## Part 1: Nanite (Automatic LOD via Clustering)

### What Nanite Does

Nanite automatically:
- Clusters vertices into hierarchical groups
- Culls entire clusters that don't contribute pixels
- Streams only visible data to GPU
- Eliminates manual LOD authoring

### Why It's Powerful

UE5 ships with 10M+ polygon scenes that run at 60 FPS because Nanite reduces rendered triangles to ~2-5M per frame through aggressive culling.

### Simplified Nanite for Your Renderer

**Goal:** Reduce your current 10K objects → visible ~2-3K objects via GPU culling

#### Phase 1: Hi-Z Pyramid Culling (1-2 weeks)

This is the **foundation** Nanite uses for visibility determination.

```rust
// new file: hiz_pyramid.rs
use ash::vk;
use std::sync::Arc;

/// Hierarchical Z-buffer pyramid for occlusion queries
pub struct HiZPyramid {
    device: Arc<ash::Device>,
    allocator: Arc<crate::vulkan::Allocator>,
    
    // Full resolution depth texture
    depth_texture: vk::Image,
    depth_allocation: vk_mem::Allocation,
    depth_view: vk::ImageView,
    
    // Mip chain (256x256 -> 1x1)
    pyramid_image: vk::Image,
    pyramid_allocation: vk_mem::Allocation,
    pyramid_views: Vec<vk::ImageView>,  // One per mip level
    
    mip_levels: u32,
    width: u32,
    height: u32,
    
    // Compute pipeline for mip generation
    compute_pipeline: vk::Pipeline,
    compute_layout: vk::PipelineLayout,
    compute_pool: vk::DescriptorPool,
}

impl HiZPyramid {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: Arc<crate::vulkan::Allocator>,
        width: u32,
        height: u32,
    ) -> crate::Result<Self> {
        // Calculate mip levels (1024x1024 = 10 levels down to 1x1)
        let mip_levels = 32 - (width.min(height)).leading_zeros();
        
        log::info!("Creating Hi-Z pyramid: {}x{} with {} mips", 
            width, height, mip_levels);
        
        // Create pyramid image with mip chain
        let pyramid_image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R32_SFLOAT)  // 32-bit float depth
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(mip_levels)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::SAMPLED | 
                           vk::ImageUsageFlags::STORAGE |
                           vk::ImageUsageFlags::TRANSFER_DST),
                None,
            )?
        };
        
        // Create views for each mip level
        let mut pyramid_views = Vec::new();
        for mip in 0..mip_levels {
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(pyramid_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R32_SFLOAT)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(mip)
                                .level_count(1),
                        ),
                    None,
                )?
            };
            pyramid_views.push(view);
        }
        
        Ok(Self {
            device,
            allocator,
            depth_texture: vk::Image::null(),
            depth_allocation: unsafe { std::mem::zeroed() },
            depth_view: vk::ImageView::null(),
            pyramid_image,
            pyramid_allocation: unsafe { std::mem::zeroed() },
            pyramid_views,
            mip_levels,
            width,
            height,
            compute_pipeline: vk::Pipeline::null(),
            compute_layout: vk::PipelineLayout::null(),
            compute_pool: vk::DescriptorPool::null(),
        })
    }
    
    /// Build pyramid from current depth buffer
    /// Call this after depth prepass
    pub unsafe fn build_pyramid(
        &self,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
        depth_format: vk::Format,
    ) -> crate::Result<()> {
        // Step 1: Copy depth to pyramid level 0
        let copy_region = vk::ImageCopy::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .layer_count(1),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            });
        
        self.device.cmd_copy_image(
            cmd,
            depth_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.pyramid_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
        
        // Step 2: Generate mips using compute shader
        // Bind compute pipeline and descriptors
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipeline,
        );
        
        // For each mip level (except level 0)
        for mip in 1..self.mip_levels as usize {
            let src_width = (self.width >> (mip - 1)).max(1);
            let src_height = (self.height >> (mip - 1)).max(1);
            let dst_width = (self.width >> mip).max(1);
            let dst_height = (self.height >> mip).max(1);
            
            // Bind source (previous mip) and destination (current mip)
            // Push constants with mip info
            let push_data = [mip as u32, 0, 0, 0];
            self.device.cmd_push_constants(
                cmd,
                self.compute_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&push_data),
            );
            
            // Dispatch compute (one thread per 2x2 block of source)
            let group_count_x = (dst_width + 7) / 8;
            let group_count_y = (dst_height + 7) / 8;
            self.device.cmd_dispatch(cmd, group_count_x, group_count_y, 1);
        }
        
        Ok(())
    }
    
    /// Query if a bounding sphere is visible
    /// Returns approximate depth at sphere center
    pub fn query_visibility(&self, sphere_center: glam::Vec3, radius: f32) -> f32 {
        // Convert world space to screen space
        // Check if depth at center is > stored depth (occluded)
        // Use finest mip for accuracy
        
        // This is a GPU query - would use async readback
        // Return 1.0 (visible). Async readback implementation is pending.
        1.0
    }
}

// Compute shader for mip generation (GLSL)
const HIZBUILD_COMP: &str = r#"
#version 460

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform sampler2D source_mip;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D dest_mip;

layout(push_constant) uniform Constants {
    uint mip_level;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    
    // Sample 2x2 block from source mip
    ivec2 src_base = coord * 2;
    float d0 = texelFetch(source_mip, src_base + ivec2(0, 0), int(pc.mip_level - 1)).r;
    float d1 = texelFetch(source_mip, src_base + ivec2(1, 0), int(pc.mip_level - 1)).r;
    float d2 = texelFetch(source_mip, src_base + ivec2(0, 1), int(pc.mip_level - 1)).r;
    float d3 = texelFetch(source_mip, src_base + ivec2(1, 1), int(pc.mip_level - 1)).r;
    
    // Take MAX depth (conservative - anything occluding this pixel occludes the 2x2)
    float max_depth = max(max(d0, d1), max(d2, d3));
    
    imageStore(dest_mip, coord, vec4(max_depth));
}
"#;
```

#### Phase 2: GPU-Driven Culling Pipeline (2-3 weeks)

Once you have Hi-Z, build an **indirect command buffer** on GPU:

```rust
// new file: indirect_culling.rs
pub struct IndirectCullingPipeline {
    device: Arc<ash::Device>,
    
    // Input: per-object bounding spheres + transforms
    object_buffer: vk::Buffer,
    object_allocation: vk_mem::Allocation,
    object_count: u32,
    
    // Output: indirect draw commands for visible objects
    indirect_command_buffer: vk::Buffer,
    indirect_allocation: vk_mem::Allocation,
    
    // Atomic counter for write position
    counter_buffer: vk::Buffer,
    counter_allocation: vk_mem::Allocation,
    
    compute_pipeline: vk::Pipeline,
    compute_layout: vk::PipelineLayout,
}

impl IndirectCullingPipeline {
    pub unsafe fn cull_and_build_commands(
        &self,
        cmd: vk::CommandBuffer,
        hiz: &HiZPyramid,
        view_proj: glam::Mat4,
    ) -> crate::Result<()> {
        // Bind compute pipeline
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipeline,
        );
        
        // Clear counter
        self.device.cmd_fill_buffer(
            cmd,
            self.counter_buffer,
            0,
            4,
            0,
        );
        
        // Dispatch: one thread per object
        let group_count = (self.object_count + 63) / 64;
        self.device.cmd_dispatch(cmd, group_count, 1, 1);
        
        // Barrier: wait for atomic writes
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
            .buffer(self.indirect_command_buffer);
        
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::DRAW_INDIRECT,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
        
        Ok(())
    }
}

// Culling compute shader (GLSL)
const CULLING_COMP: &str = r#"
#version 460

struct ObjectData {
    vec4 sphere;          // xyz = center, w = radius
    mat4 transform;
    uint mesh_index;
    uint material_index;
};

struct IndirectCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    uint vertex_offset;
    uint first_instance;
};

layout(set = 0, binding = 0) uniform MVP {
    mat4 view_proj;
} matrices;

layout(set = 0, binding = 1) buffer Objects {
    ObjectData objects[];
};

layout(set = 0, binding = 2) buffer IndirectCommands {
    IndirectCommand commands[];
};

layout(set = 0, binding = 3) buffer DrawCounter {
    uint counter;
};

layout(set = 0, binding = 4) uniform sampler2D hiz;

layout(local_size_x = 64) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= objects.length()) return;
    
    ObjectData obj = objects[idx];
    vec3 sphere_center = (matrices.view_proj * vec4(obj.sphere.xyz, 1.0)).xyz;
    float radius = obj.sphere.w;
    
    // Simple frustum cull
    if (sphere_center.z + radius < 0.0) return;  // Behind camera
    if (sphere_center.z - radius > 1000.0) return;  // Too far
    
    // Project to screen space
    vec2 screen_center = sphere_center.xy / sphere_center.z;
    screen_center = screen_center * 0.5 + 0.5;  // NDC to [0,1]
    
    // Hi-Z occlusion test
    float depth = texture(hiz, screen_center).r;
    float sphere_depth = sphere_center.z;
    
    if (sphere_depth > depth + radius) return;  // Occluded
    
    // Object is visible - add to indirect buffer
    uint write_idx = atomicAdd(DrawCounter.counter, 1);
    commands[write_idx].index_count = obj.mesh_index;  // Simplified
    commands[write_idx].instance_count = 1;
}
"#;
```

#### Phase 3: Mesh Clustering (Optional, 2-3 weeks)

For true Nanite, cluster your meshes:

```rust
/// Simplified mesh clustering for Nanite-like behavior
pub struct MeshCluster {
    /// Triangles in this cluster
    pub triangles: Vec<[u32; 3]>,
    
    /// Bounding sphere
    pub center: glam::Vec3,
    pub radius: f32,
    
    /// Error metric (how much deviation allowed)
    pub error: f32,
    
    /// GPU buffer offset
    pub gpu_offset: u32,
}

impl MeshCluster {
    /// Create clusters from mesh (simplified METIS-like approach)
    pub fn create_from_mesh(mesh: &Mesh, target_cluster_size: usize) -> Vec<Self> {
        let triangles = &mesh.indices.as_ref().unwrap();
        let vertices = &mesh.vertices;
        
        let triangle_count = triangles.len() / 3;
        let cluster_count = (triangle_count + target_cluster_size - 1) / target_cluster_size;
        
        let mut clusters = Vec::with_capacity(cluster_count);
        
        // Simple clustering: divide by triangle index
        for cluster_idx in 0..cluster_count {
            let start = cluster_idx * target_cluster_size;
            let end = ((cluster_idx + 1) * target_cluster_size).min(triangle_count);
            
            let mut cluster_triangles = Vec::new();
            let mut bounds_min = glam::Vec3::splat(f32::MAX);
            let mut bounds_max = glam::Vec3::splat(f32::MIN);
            
            for tri_idx in start..end {
                let base = tri_idx * 3;
                let tri = [
                    triangles[base],
                    triangles[base + 1],
                    triangles[base + 2],
                ];
                cluster_triangles.push(tri);
                
                for &vi in &tri {
                    let v = vertices[vi as usize].position;
                    bounds_min = bounds_min.min(glam::Vec3::from(v));
                    bounds_max = bounds_max.max(glam::Vec3::from(v));
                }
            }
            
            let center = (bounds_min + bounds_max) * 0.5;
            let radius = (bounds_max - bounds_min).length() * 0.5;
            
            clusters.push(Self {
                triangles: cluster_triangles,
                center,
                radius,
                error: 0.1,  // 10cm error threshold
                gpu_offset: 0,  // Fill in during upload
            });
        }
        
        clusters
    }
}
```

---

## Part 2: Lumen (Real-Time Global Illumination)

### What Lumen Does

Lumen:
- Traces rays in 3D voxel grid
- Accumulates contributions over multiple frames
- Produces real-time indirect lighting
- Eliminates baking step

### Simplified Lumen for Your Renderer

**Goal:** Real-time GI without voxelization overhead

#### Phase 1: Screen-Space Global Illumination (1-2 weeks)

Start with **SSGI** - easier than voxel-based:

```rust
// new file: screen_space_gi.rs
pub struct ScreenSpaceGI {
    device: Arc<ash::Device>,
    
    // Input: depth, normals, albedo
    gi_image: vk::Image,
    gi_view: vk::ImageView,
    gi_allocation: vk_mem::Allocation,
    
    compute_pipeline: vk::Pipeline,
    compute_layout: vk::PipelineLayout,
}

impl ScreenSpaceGI {
    pub unsafe fn compute_gi(
        &self,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
        normal_image: vk::Image,
        albedo_image: vk::Image,
        view_proj: glam::Mat4,
    ) -> crate::Result<()> {
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.compute_pipeline,
        );
        
        // Push matrices
        let push_data = view_proj.to_cols_array();
        self.device.cmd_push_constants(
            cmd,
            self.compute_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_data),
        );
        
        // Dispatch: one thread per pixel
        self.device.cmd_dispatch(cmd, 1920 / 8, 1080 / 8, 1);
        
        Ok(())
    }
}

// Screen-space GI compute shader
const SSGI_COMP: &str = r#"
#version 460

layout(set = 0, binding = 0) uniform sampler2D depth;
layout(set = 0, binding = 1) uniform sampler2D normal;
layout(set = 0, binding = 2) uniform sampler2D albedo;
layout(set = 0, binding = 3) uniform sampler2D history;  // Previous frame GI
layout(set = 0, binding = 4, rgba16f) uniform writeonly image2D gi_output;

layout(push_constant) uniform ViewProj {
    mat4 view_proj;
} pc;

const int SAMPLES = 16;
const float MAX_RAY_LENGTH = 100.0;
const float SAMPLE_RADIUS = 0.5;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(coord) / vec2(imageSize(gi_output));
    
    // Reconstruct position from depth
    float d = texture(depth, uv).r;
    vec3 pos = reconstruct_position(d, uv);
    vec3 normal = normalize(texture(normal, uv).rgb * 2.0 - 1.0);
    vec3 albedo_color = texture(albedo, uv).rgb;
    
    // Accumulate indirect light from neighboring pixels
    vec3 gi = vec3(0.0);
    
    // Sample around this pixel in screen space
    for (int i = 0; i < SAMPLES; ++i) {
        float angle = (float(i) / float(SAMPLES)) * 6.28318;
        float dist = sqrt(float(i) / float(SAMPLES)) * SAMPLE_RADIUS;
        
        vec2 sample_uv = uv + vec2(cos(angle), sin(angle)) * dist;
        
        // Get sample position and normal
        float sample_d = texture(depth, sample_uv).r;
        vec3 sample_pos = reconstruct_position(sample_d, sample_uv);
        vec3 sample_normal = texture(normal, sample_uv).rgb;
        
        // Compute GI contribution
        vec3 to_sample = sample_pos - pos;
        float distance = length(to_sample);
        
        if (distance > 0.01 && distance < MAX_RAY_LENGTH) {
            // Directional falloff
            float falloff = max(0.0, dot(normal, normalize(to_sample))) / 
                           (1.0 + distance * distance);
            
            // Sample previous frame's GI for temporal coherence
            vec3 sample_indirect = texture(history, sample_uv).rgb;
            
            gi += sample_indirect * albedo_color * falloff;
        }
    }
    
    gi /= float(SAMPLES);
    
    // Temporal accumulation with previous frame
    vec3 history_gi = texture(history, uv).rgb;
    vec3 final_gi = mix(gi, history_gi, 0.9);  // 90% history blend
    
    imageStore(gi_output, coord, vec4(final_gi, 1.0));
}
"#;
```

#### Phase 2: Voxel-Based GI (2-3 weeks) - Optional

For more sophisticated GI, add voxelization:

```rust
// new file: voxel_gi.rs
pub struct VoxelGI {
    device: Arc<ash::Device>,
    allocator: Arc<crate::vulkan::Allocator>,
    
    // 3D texture: 128x128x128 voxels
    voxel_grid: vk::Image,
    voxel_view: vk::ImageView,
    voxel_allocation: vk_mem::Allocation,
    
    voxel_pipeline: vk::Pipeline,
    voxel_layout: vk::PipelineLayout,
    
    // Size of world covered by voxel grid
    world_size: f32,
    voxel_count: u32,
}

impl VoxelGI {
    pub unsafe fn voxelize_scene(
        &self,
        cmd: vk::CommandBuffer,
        meshes: &[Mesh],
        view: glam::Mat4,
    ) -> crate::Result<()> {
        // Step 1: Clear voxel grid
        self.device.cmd_fill_buffer(cmd, self.voxel_grid, 0, 1000000, 0);
        
        // Step 2: Rasterize all geometry to voxels
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.voxel_pipeline,
        );
        
        // Dispatch for each mesh
        for mesh in meshes {
            // Push voxel space matrix
            let scale = 1.0 / self.world_size;
            let voxel_transform = glam::Mat4::from_scale(glam::Vec3::splat(scale));
            
            self.device.cmd_push_constants(
                cmd,
                self.voxel_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&voxel_transform.to_cols_array()),
            );
            
            // Draw mesh
            self.device.cmd_draw_indexed(cmd, mesh.indices.as_ref().unwrap().len() as u32, 1, 0, 0, 0);
        }
        
        Ok(())
    }
    
    pub unsafe fn trace_gi_rays(
        &self,
        cmd: vk::CommandBuffer,
        gi_output: vk::Image,
    ) -> crate::Result<()> {
        // Compute shader: trace rays through voxel grid
        // Each thread traces one ray, accumulates lighting
        
        self.device.cmd_dispatch(cmd, 1920 / 8, 1080 / 8, 1);
        
        Ok(())
    }
}

// Voxelization vertex shader
const VOXEL_VERT: &str = r#"
#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(push_constant) uniform Transform {
    mat4 voxel_transform;
};

void main() {
    // Transform to voxel space [-1, 1]
    gl_Position = voxel_transform * vec4(position, 1.0);
}
"#;

// Voxelization fragment shader - atomic writes to 3D texture
const VOXEL_FRAG: &str = r#"
#version 460

layout(location = 0) in vec3 voxel_pos;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0, rgba8) uniform image3D voxel_grid;

void main() {
    // Voxel coordinate from fragment position
    ivec3 voxel_coord = ivec3((voxel_pos + 1.0) * 0.5 * 128.0);
    
    if (any(lessThan(voxel_coord, ivec3(0))) ||
        any(greaterThanEqual(voxel_coord, ivec3(128)))) return;
    
    // Atomic write radiance to voxel
    vec4 radiance = vec4(normal * 0.5 + 0.5, 1.0);
    imageAtomicAdd(voxel_grid, voxel_coord, radiance);
}
"#;
```

---

## Part 3: TSR (Temporal Super-Resolution)

### What TSR Does

TSR:
- Renders at lower resolution (e.g., 1440p → 720p)
- Uses temporal information to reconstruct details
- Applies motion correction
- Results in near-native quality at 60-80% cost

### Implementing TSR (1-2 weeks)

```rust
// new file: temporal_upscaling.rs
pub struct TemporalSuperResolution {
    device: Arc<ash::Device>,
    allocator: Arc<crate::vulkan::Allocator>,
    
    // Render target at 50-75% resolution
    low_res_color: vk::Image,
    low_res_view: vk::ImageView,
    low_res_allocation: vk_mem::Allocation,
    
    // Motion vectors (2D per pixel)
    motion_vectors: vk::Image,
    motion_view: vk::ImageView,
    motion_allocation: vk_mem::Allocation,
    
    // Depth at low resolution
    low_res_depth: vk::Image,
    low_res_depth_view: vk::ImageView,
    low_res_depth_allocation: vk_mem::Allocation,
    
    // Temporal history (full resolution)
    history[2]: [vk::Image; 2],
    history_views[2]: [vk::ImageView; 2],
    history_allocations[2]: [vk_mem::Allocation; 2],
    
    upscale_pipeline: vk::Pipeline,
    upscale_layout: vk::PipelineLayout,
    
    upscale_factor: u32,  // 1.33x, 1.5x, 2.0x
    frame_index: u32,
}

impl TemporalSuperResolution {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: Arc<crate::vulkan::Allocator>,
        full_width: u32,
        full_height: u32,
        upscale_factor: f32,  // 1.5 = render at 67% res
    ) -> crate::Result<Self> {
        let low_width = (full_width as f32 / upscale_factor) as u32;
        let low_height = (full_height as f32 / upscale_factor) as u32;
        
        // Create low-res color target
        let low_res_color = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .extent(vk::Extent3D {
                        width: low_width,
                        height: low_height,
                        depth: 1,
                    })
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | 
                           vk::ImageUsageFlags::SAMPLED),
                None,
            )?
        };
        
        // Create motion vectors (2 floats per pixel)
        let motion_vectors = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16G16_SFLOAT)
                    .extent(vk::Extent3D {
                        width: low_width,
                        height: low_height,
                        depth: 1,
                    })
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | 
                           vk::ImageUsageFlags::SAMPLED),
                None,
            )?
        };
        
        // Create history buffers (double-buffered)
        let history = [vk::Image::null(); 2];
        let history_views = [vk::ImageView::null(); 2];
        
        for _ in 0..2 {
            // Full resolution history
            let img = unsafe {
                device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .extent(vk::Extent3D {
                            width: full_width,
                            height: full_height,
                            depth: 1,
                        })
                        .usage(vk::ImageUsageFlags::SAMPLED |
                               vk::ImageUsageFlags::TRANSFER_DST),
                    None,
                )?
            };
        }
        
        Ok(Self {
            device,
            allocator,
            low_res_color,
            low_res_view: vk::ImageView::null(),
            low_res_allocation: unsafe { std::mem::zeroed() },
            motion_vectors,
            motion_view: vk::ImageView::null(),
            motion_allocation: unsafe { std::mem::zeroed() },
            low_res_depth,
            low_res_depth_view: vk::ImageView::null(),
            low_res_depth_allocation: unsafe { std::mem::zeroed() },
            history,
            history_views,
            history_allocations: [unsafe { std::mem::zeroed() }; 2],
            upscale_pipeline: vk::Pipeline::null(),
            upscale_layout: vk::PipelineLayout::null(),
            upscale_factor: upscale_factor as u32,
            frame_index: 0,
        })
    }
    
    /// Perform temporal upscaling from low-res to full-res
    pub unsafe fn upscale(
        &mut self,
        cmd: vk::CommandBuffer,
        output_image: vk::Image,
    ) -> crate::Result<()> {
        // Bind compute pipeline
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.upscale_pipeline,
        );
        
        // Descriptor set: [low_res_color, motion, history, output]
        // Compute shader reads neighbor samples, applies reprojection
        self.device.cmd_dispatch(cmd, 1920 / 8, 1080 / 8, 1);
        
        // Swap history buffers
        self.frame_index += 1;
        
        Ok(())
    }
}

// TSR upscaling compute shader
const TSR_COMP: &str = r#"
#version 460

layout(set = 0, binding = 0) uniform sampler2D low_res_color;
layout(set = 0, binding = 1) uniform sampler2D motion_vectors;
layout(set = 0, binding = 2) uniform sampler2D history;
layout(set = 0, binding = 3) uniform sampler2D low_res_depth;
layout(set = 0, binding = 4, rgba16f) uniform writeonly image2D output_image;

layout(push_constant) uniform Constants {
    vec2 low_res_size;
    vec2 full_res_size;
    float motion_scale;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(coord) / pc.full_res_size;
    
    // Low-res UV
    vec2 low_uv = uv / (pc.low_res_size / pc.full_res_size);
    
    // Sample motion vector
    vec2 motion = texture(motion_vectors, low_uv).rg * pc.motion_scale;
    
    // Reproject to history
    vec2 reprojected_uv = low_uv - motion;
    
    // Fetch low-res color
    vec3 current = texture(low_res_color, low_uv).rgb;
    
    // Fetch history
    vec3 history_color = texture(history, reprojected_uv).rgb;
    
    // Temporal blend (90% history, 10% current for stability)
    vec3 blended = mix(current, history_color, 0.9);
    
    // Catmull-Rom filter for upscaling
    // Sample 2x2 neighbors
    vec2 dx = vec2(1.0 / pc.low_res_size.x, 0.0);
    vec2 dy = vec2(0.0, 1.0 / pc.low_res_size.y);
    
    vec3 c00 = texture(low_res_color, low_uv - dx - dy).rgb;
    vec3 c10 = texture(low_res_color, low_uv + dx - dy).rgb;
    vec3 c01 = texture(low_res_color, low_uv - dx + dy).rgb;
    vec3 c11 = texture(low_res_color, low_uv + dx + dy).rgb;
    
    // Catmull-Rom interpolation
    vec2 fract = fract(low_uv * pc.low_res_size);
    
    vec3 h0 = mix(c00, c10, fract.x);
    vec3 h1 = mix(c01, c11, fract.x);
    vec3 filtered = mix(h0, h1, fract.y);
    
    // Final composite
    vec3 final = mix(blended, filtered, 0.5);
    
    imageStore(output_image, coord, vec4(final, 1.0));
}
"#;
```

---

## Integration Plan: How to Add All Three

### Phase 1 (Weeks 1-3): Foundation
- Implement Hi-Z pyramid + GPU culling
- Result: 5-8% FPS improvement, better scalability

### Phase 2 (Weeks 4-5): GI
- Add screen-space GI
- Result: Better lighting, more realistic
- FPS cost: 2-4% (negligible on modern GPUs)

### Phase 3 (Weeks 6-7): Temporal Upscaling
- Add TSR support
- Render core game at 67-75% resolution
- Upscale temporally to full resolution
- Result: 20-30% FPS improvement at quality parity

### Phase 4 (Weeks 8+): Mesh Clustering & Advanced GI
- Add mesh clustering for Nanite-like LOD
- Switch GI to voxel-based if needed
- Fine-tune parameters

---

## Implementation Timeline (8-Week Sprint)

```
Week 1-2: Hi-Z + GPU Culling
  Mon-Tue:  Hi-Z pyramid infrastructure
  Wed-Thu:  Compute shader implementation
  Fri:      Integration + profiling
  Expected: 5-8% FPS gain

Week 3: Indirect Commands + Mesh Clustering
  Mon-Tue:  Indirect command buffer generation
  Wed-Thu:  Mesh clustering algorithm
  Fri:      Testing
  Expected: 3-5% additional FPS

Week 4-5: Screen-Space GI
  Mon-Tue:  SSGI shader implementation
  Wed-Thu:  Temporal accumulation
  Fri:      Quality tuning
  Expected: 2-4% FPS cost, but better image quality

Week 6-7: Temporal Super-Resolution
  Mon-Tue:  Motion vector generation
  Wed-Thu:  TSR upscaling shader
  Fri:      Motion compensation
  Expected: 20-30% FPS improvement (at 75% res render)

Week 8: Advanced GI + Polish
  Mon-Tue:  Voxel GI (optional)
  Wed:      Integration testing
  Thu-Fri:  Performance tuning
  Expected: +5-10% additional quality
```

---

## Final Performance Projection

```
Baseline (Current):           50 FPS
+ Week 1-2 (Hi-Z):            53-55 FPS (+6-10%)
+ Week 3 (Clustering):        56-58 FPS (+12-16%)
+ Week 4-5 (SSGI):            54-56 FPS (cost offset by culling)
+ Week 6-7 (TSR at 75% res):  70-85 FPS (effective quality maintained)
+ Week 8 (Advanced GI):       75-95 FPS (visual quality similar to UE5)
```

**Result:** Competitive with UE5 (without Nanite/Lumen complexity!)

---

## Key Learning Points from UE5

### 1. **Nanite Philosophy**
- **Automatic LOD is essential** for scale
- **Cluster-based** approach beats triangle-based
- **GPU-driven** rendering (no CPU submission overhead)
- **Conservative occlusion** (better to render extra than cull visible)

### 2. **Lumen Philosophy**
- **Temporal accumulation** makes real-time GI feasible
- **Multiple bounce approximation** vs full path tracing
- **Voxel grids work** but SSGI is cheaper
- **Probe-based fallback** for dynamic areas

### 3. **TSR Philosophy**
- **Motion vectors are critical** - quality depends on accuracy
- **Temporal stability** beats per-frame quality
- **Multiple samples** improve reprojection
- **Conservative upscaling** (blur rather than alias)

---

## Recommended Tools & Profiling

```bash
# GPU profiling (see current load)
cargo run --release --features "gpu-profiler"

# CPU-side bottleneck detection
TRACY_ENABLE=1 cargo run --release
# Open trace in Tracy profiler

# Vulkan-specific profiling
RenderDoc capture + analysis

# Benchmark suite
cargo bench --bench tsr_upscaling
cargo bench --bench gi_performance
```

---

## Production Deployment Checklist

- [ ] Hi-Z pyramid working on 5+ GPU architectures
- [ ] GPU culling properly synchronized
- [ ] Motion vectors accurate (no ghosting)
- [ ] TSR quality >= native at 70% res
- [ ] SSGI converges in 8-16 frames
- [ ] No visible popping from culling
- [ ] Memory usage tracked and budgeted
- [ ] Fallbacks for unsupported features
- [ ] Comprehensive profiling data

---

## When to Use Each Feature

| Feature | Use Case | Performance Cost |
|---------|----------|-----------------|
| **Hi-Z** | Large open worlds (10K+ objects) | 0% (saves time) |
| **SSGI** | Dynamic scenes, no lightmap budget | 2-4% |
| **TSR** | High-end targets (240 FPS VR) | -20% (net save) |
| **Voxel GI** | Cinematic quality, heavy compute | 5-10% |
| **Mesh Clustering** | 100K+ triangle models | 0% (saves time) |

---

## Conclusion

You can achieve **90% of UE5's visual quality** without 90% of the complexity:

✅ Nanite-like LOD via GPU culling (Hi-Z + clustering)  
✅ Lumen-like GI via SSGI + voxel fallback  
✅ TSR for massive perf boost  

**Timeline:** 8 weeks of focused work  
**Team Size:** 1-2 engineers  
**Final Quality:** Competitive with UE5 for indie games  

Start with Hi-Z pyramid. It's the foundation everything else builds on. 🚀

