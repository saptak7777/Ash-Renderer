// shaders/include/structures.glsl
// Verified Phase 4 Binding Update: Set 0, Binding 4 for Bindless Buffers
// Single Source of Truth for shader-side structures
// Matches Rust definitions in src/renderer/model_renderer.rs
// Verification: Dependency tracking active.

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// --- Basic Structs (Leaf nodes first) ---

struct InstanceData {
    vec4 bounds_center;
    vec4 bounds_extents;
    mat4 model;
    mat4 prev_model;
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

const uint MATERIAL_FLAG_ALPHA_TESTED = 1u << 0;

struct MaterialData {
    vec4 base_color_factor;
    vec4 emissive_factor;
    vec4 parameters; // x: metallic, y: roughness, z: occlusion strength, w: normal scale
    ivec4 texture_indices; // x: base_color, y: normal, z: metallic_roughness, w: occlusion
    int emissive_texture_index;
    int tint_index;
    float alpha_cutoff;
    uint flags;
};

// [REMOVED] struct HemisphereAmbient { ... }

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
//
// Qualifiers guide:
//   restrict  — pointer is not aliased by any other BDA pointer in scope; enables
//               better alias analysis and lowers register pressure in the compiler.
//   readonly  — shader only reads; driver may cache aggressively.
//   writeonly — shader only writes; driver may skip readback.
//
// CountBuffer / TileIndexBuffer are intentionally left without restrict because
// they participate in atomics that may alias shared GPU state across wavefronts.
// LightBuffer is r/w by Forward+ tile-generation passes.

layout(buffer_reference, scalar, buffer_reference_align = 8) restrict readonly buffer FrameData {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 view_proj;
    mat4 prev_view_proj;
    mat4 view_proj_no_jitter;
    mat4 prev_view_proj_no_jitter;
    mat4 light_space_matrix;
    mat4 inv_projection;
    mat4 normal_matrix;
    vec4 camera_pos;
    SceneLighting scene_lighting;
    vec4 screen_params; // width, height, 1/width, 1/height
    uint hiz_levels;
    uint _pad_frame;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer ObjectBuffer {
    InstanceData objects[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer MaterialBuffer {
    MaterialData materials[];
};

// LightBuffer: r/w; restrict omitted intentionally (Forward+ tile writes alias this).
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer LightBuffer {
    Light lights[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer TransformBuffer {
    mat4 matrices[];
};

// Set 0: Unified Bindless consolidated resources
layout(set = 0, binding = 0) uniform sampler2D global_textures[];
layout(set = 0, binding = 1) uniform usampler2DArray global_page_tables[];
layout(set = 0, binding = 2) uniform samplerCube global_cubemaps[];

// Binding 3: Global Storage Images (for compute writes)
layout(set = 0, binding = 3, rgba16f) uniform image2D global_storage_images[];

// Binding 5: Global Storage Image Arrays (for Page Table writes)
layout(set = 0, binding = 5, r32ui) uniform uimage2DArray global_storage_uimages_2d_array[];

// Binding 4: Bindless Storage Buffers
layout(set = 0, binding = 4, std430) readonly buffer BindlessBuffer {
    vec4 data[];
} bindless_buffers[];

// IndirectBuffer: write-only from compute; restrict allows the driver to skip reads.
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict writeonly buffer IndirectBuffer {
    IndirectDrawCommand commands[];
};

layout(buffer_reference, scalar) buffer CountBuffer {
    uint count;
};

layout(buffer_reference, scalar) buffer TileIndexBuffer {
    uint tileData[];
};

// IndexBuffer: read-only, tight 4-byte alignment known at compile time.
layout(buffer_reference, scalar, buffer_reference_align = 4) restrict readonly buffer IndexBuffer { 
    uint indices[]; 
};

// --- VSM Shadow Mapping Structs ---

struct VsmPageRequest {
    uint virtual_x;
    uint virtual_y;
    float priority;
    uint layer;
};

struct VsmPageAllocation {
    uint virtual_x;
    uint virtual_y;
    uint physical_x;
    uint physical_y;
    uint layer;
    uint flags;
    uint _padding0;
    uint _padding1;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer VsmRequestBuffer {
    uint count;
    uint overflow_count;
    uint _padding[2];
    VsmPageRequest data[];
};

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer VsmAllocationBuffer {
    uint count;
    VsmPageAllocation allocations[];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) restrict readonly buffer VsmGlobal {
    mat4 light_view_projections[16];
    mat4 view_proj;
    mat4 inv_view_proj;
    vec4 camera_position;
    vec4 light_dir;
    uint page_table_size;
    uint page_table_index;
    uint physical_cache_index;
    uint scene_depth_index;
    uint page_table_storage_index;
    uint physical_cache_storage_index;
    uint64_t request_ptr;
    uint64_t allocation_ptr;
    uint64_t _pad3;
};

uint load_index(uint64_t ptr, uint logical_index) {
    if (ptr == 0) return 0;
    IndexBuffer ib = IndexBuffer(ptr);
    return ib.indices[logical_index];
}

// --- Push Constants ---

#ifndef SHADER_PUSH_CONSTANT_OVERRIDE
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
    uint light_count;
    uint _pad_vsm2;

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
    layout(offset = 104) uint64_t vsm_ptr;
    
    // Shadow/Culling extensions (112-127)
    layout(offset = 112) uint clipmap_level;
    layout(offset = 116) uint object_count;
    layout(offset = 120) uint base_index;
    layout(offset = 124) uint indirect_start;
} push;
#endif // SHADER_PUSH_CONSTANT_OVERRIDE
