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
    uint _pad1;
    uint _pad2;
    uint _pad3;
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

// --- Index Buffer for BDA-based Index Pulling ---
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer IndexBuffer { 
    uint indices[]; 
};

uint load_index(uint64_t ptr, uint logical_index) {
    IndexBuffer ib = IndexBuffer(ptr);
    return ib.indices[logical_index];
}

// --- Push Constants ---

#ifndef SKIP_PUSH_CONSTANTS
// Modern Push Constants - Full Bindless/BDA
layout(push_constant) uniform PushConstants {
    // Pointer stage (0-47)
    uint64_t frame_ptr;
    uint64_t vertex_ptr;
    uint64_t instance_ptr;
    uint64_t material_ptr;
    uint64_t index_ptr;
    uint64_t _ptr_padding; // Padding to 48 bytes

    // Control stage (48-111)
    layout(offset = 48) mat4 model; 
    layout(offset = 112) uint material_index; // Shifted up
    layout(offset = 116) uint use_instancing;
    layout(offset = 120) uint _unused_flags[2]; // Padding to maintain alignment

    // Fragment/Debug stage (128-159)
    layout(offset = 128) uint flags;
    layout(offset = 132) uint debug_path;
    layout(offset = 136) uint debug_visualization_enabled;
} push;
#endif
