// shaders/include/structures.glsl
// Single Source of Truth for shader-side structures
// Matches Rust definitions in src/renderer/model_renderer.rs

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// --- Basic Structs (Leaf nodes first) ---

struct InstanceData {
    vec4 bounds_center;
    vec4 bounds_extents;
    mat4 model;
    uint draw_index;
    uint first_index;
    uint index_count;
    int vertex_offset;
    vec4 color;
    vec4 custom;
    uint parent_index;
    float error_metric;
    uint flags;
    uint material_index;
};

struct MaterialData {
    vec4 base_color_factor;
    vec4 emissive_factor;
    vec4 parameters; // x: metallic, y: roughness, z: occlusion strength, w: normal scale
    ivec4 texture_indices; // x: base_color, y: normal, z: metallic_roughness, w: occlusion
    int emissive_texture_index;
    int tint_index;
    float alpha_cutoff;
    float _padding;
};

struct HemisphereAmbient {
    vec4 sky_color;       // xyz = color, w = intensity
    vec4 ground_color;    // xyz = color, w = unused
};

struct DirectionalLight {
    vec4 direction;       // xyz = direction, w = shadow enabled
    vec4 color_intensity; // xyz = color, w = intensity
};

// GPU Forward+ Light structure
struct Light {
    vec4 position;   // xyz = position, w = radius
    vec4 color;      // rgb = color, a = intensity
    vec4 direction;  // xyz = direction, w = type (0=point, 1=directional, 2=spot)
    vec4 params;     // x = inner, y = outer, z = falloff, w = enabled
};

struct IndirectDrawCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct SceneLighting {
    HemisphereAmbient ambient;
    DirectionalLight directional;
    uint point_light_count;
    uint num_tiles_x;
    uint num_tiles_y;
    uint tile_size;
};

// --- BDA Buffer References (Require structs above) ---

layout(buffer_reference, scalar) readonly buffer FrameData {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 view_proj;
    mat4 prev_view_proj;
    mat4 light_space_matrix;
    mat4 normal_matrix;
    vec4 camera_pos;
    SceneLighting scene_lighting;
};

layout(buffer_reference, scalar) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(buffer_reference, scalar) readonly buffer ObjectBuffer {
    InstanceData objects[];
};

layout(buffer_reference, scalar) readonly buffer MaterialBuffer {
    MaterialData materials[];
};

layout(buffer_reference, scalar) buffer LightBuffer {
    Light lights[];
};

layout(buffer_reference, scalar) buffer TileIndexBuffer {
    uint tileData[];
};

// --- Index Buffer for BDA-based Index Pulling ---
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer IndexBuffer { 
    uint indices[]; 
};

uint load_index(uint64_t ptr, uint logical_index) {
    if (ptr == 0) return 0;
    IndexBuffer ib = IndexBuffer(ptr);
    return ib.indices[logical_index];
}

// --- Push Constants ---

#ifndef SKIP_PUSH_CONSTANTS
// Modern Push Constants - Full Bindless/BDA
layout(push_constant) uniform PushConstants {
    // Pointer stage (0-55)
    uint64_t frame_ptr;
    uint64_t vertex_ptr;
    uint64_t instance_ptr;
    uint64_t material_ptr;
    uint64_t index_ptr;
    uint64_t light_ptr;
    uint64_t tile_ptr;

    // Control stage (64-127)
    layout(offset = 64) mat4 model; 
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint use_instancing;
    layout(offset = 136) uint flags;
    layout(offset = 140) uint debug_path;
    layout(offset = 144) uint debug_visualization_enabled;
} push;
#endif
