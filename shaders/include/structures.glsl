// shaders/include/structures.glsl
// Single Source of Truth for shader-side structures
// Matches Rust definitions in src/renderer/model_renderer.rs

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

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
    uint cluster_offset;
    uint cluster_count;
    uint flags;
    uint _padding;
};

#ifndef SKIP_PUSH_CONSTANTS
// Push Constants - Strict 160-byte block
// Matches DrawPushConstants in Rust
layout(push_constant) uniform PushConstants {
    // Vertex stage (0-127)
    layout(offset = 0) mat4 model;
    layout(offset = 64) uint joint_offset;
    layout(offset = 68) uint use_instancing;
    layout(offset = 72) uint instance_buffer_index;
    layout(offset = 76) uint joint_buffer_index;
    layout(offset = 80) uint64_t vertex_heap_ptr; // BDA pointer to vertex data
    layout(offset = 88) uint is_skinned;
    layout(offset = 92) uint _vertex_padding[9]; // Pad to 128 bytes

    // Fragment stage (128-159)
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint debug_path; // 0: None, 1: GPU-Driven, 2: Legacy
    layout(offset = 136) uint flags; // bit 0: receive_shadows
    layout(offset = 140) uint material_buffer_index;
    layout(offset = 144) uint debug_visualization_enabled;
    layout(offset = 148) float uv_min;
    layout(offset = 152) float uv_max;
    layout(offset = 156) float texel_size;
} push;
#endif

// RAGE Hemisphere Ambient
struct HemisphereAmbient {
    vec4 sky_color;       // xyz = color, w = intensity
    vec4 ground_color;    // xyz = color, w = unused
};

struct DirectionalLight {
    vec4 direction;       // xyz = direction, w = shadow enabled
    vec4 color_intensity; // xyz = color, w = intensity
};

struct SceneLighting {
    HemisphereAmbient ambient;
    DirectionalLight directional;
    uint point_light_count;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
