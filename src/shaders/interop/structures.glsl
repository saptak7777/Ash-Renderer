// shaders/include/structures.glsl
// Verified Phase 4 Binding Update: Set 0, Binding 4 for Bindless Buffers
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
    int ibl_irradiance_index;
    int ibl_prefilter_index;
    int ibl_brdf_lut_index;
    float ibl_intensity;
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

layout(buffer_reference, scalar) readonly buffer TransformBuffer {
    mat4 matrices[];
};

// Set 0: Unified Bindless consolidated resources
#ifndef SKIP_GLOBAL_BINDED_RESOURCES
layout(set = 0, binding = 0) uniform sampler2D global_textures[];
layout(set = 0, binding = 1) uniform usampler2DArray global_page_tables[];
layout(set = 0, binding = 2) uniform samplerCube global_cubemaps[];

// Binding 3: Global Storage Images (for compute writes)
layout(set = 0, binding = 3, rgba16f) uniform image2D global_storage_images[];

// Binding 4: Bindless Storage Buffers
layout(set = 0, binding = 4, std430) readonly buffer BindlessBuffer {
    vec4 data[];
} bindless_buffers[];
#endif

layout(buffer_reference, scalar) writeonly buffer IndirectBuffer {
    IndirectDrawCommand commands[];
};

layout(buffer_reference, scalar) buffer CountBuffer {
    uint count;
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

    // Texture indices (56-63)
    uint vsm_page_index;
    uint vsm_cache_index;

    // Phase 19: Transient Transform (64-79)
    uint64_t transform_ptr;           // 64
    uint transform_index;             // 72
    uint _padding_ptr;                // 76

    // Control stage (80-111)
    layout(offset = 80) uint material_index;
    layout(offset = 84) uint use_instancing;
    layout(offset = 88) uint flags;
    layout(offset = 92) uint debug_path;
    layout(offset = 96) uint debug_mode;
    layout(offset = 100) uint skybox_index;
} push;
#endif
