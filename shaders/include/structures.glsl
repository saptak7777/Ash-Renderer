// shaders/include/structures.glsl
// Single Source of Truth for shader-side structures
// Matches Rust definitions in src/renderer/model_renderer.rs

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

// Push Constants - Strict 160-byte block
// Matches DrawPushConstants in Rust
layout(push_constant) uniform PushConstants {
    // Vertex stage (0-127)
    layout(offset = 0) mat4 model;
    layout(offset = 64) uint joint_offset;
    layout(offset = 68) uint use_instancing;
    layout(offset = 72) uint instance_buffer_index;
    layout(offset = 76) uint joint_buffer_index;
    layout(offset = 80) uint _vertex_padding[12]; // Pad to 128 bytes

    // Fragment stage (128-159)
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint debug_path; // 0: None, 1: GPU-Driven, 2: Legacy
    layout(offset = 136) uint flags; // bit 0: receive_shadows
    layout(offset = 140) uint material_buffer_index;
    layout(offset = 144) uint debug_visualization_enabled;
    layout(offset = 148) uint _fragment_padding[3]; // Pad to 160 bytes
} push;
