#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define SKIP_PUSH_CONSTANTS
#include "interop/structures.glsl"

// Shadow map fragment shader - outputs variance moments for VSM
layout(location = 0) in vec2 inUV;

// Output variance moments to color attachment 0 (R32G32_SFLOAT)
layout(location = 0) out vec2 outMoments;

layout(push_constant) uniform ShadowPushConstants {
    // Pointer stage (0-55)
    uint64_t frame_ptr;    // 0
    uint64_t vertex_ptr;   // 8
    uint64_t instance_ptr; // 16
    uint64_t material_ptr; // 24
    uint64_t index_ptr;    // 32
    uint64_t light_ptr;    // 40
    uint64_t tile_ptr;     // 48

    // Texture indices (56-63)
    uint vsm_page_index;   // 56
    uint vsm_cache_index;  // 60

    // Control bits (64-127)
    layout(offset = 64) mat4 model; 
    
    // Material & Flags (128-159)
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint use_instancing;
    layout(offset = 136) uint flags;
    layout(offset = 140) uint debug_path;
    layout(offset = 144) uint debug_visualization_enabled;
    layout(offset = 148) uint skybox_index;
    
    // VSM-specific (160-223)
    layout(offset = 160) mat4 lightSpaceMatrix;
} pc;

// Binding 0: Textures (Set 0)
layout(set = 0, binding = 0) uniform sampler2D textures[];

void main() {
    // Alpha testing for transparent materials
    if (pc.material_ptr != 0) {
        MaterialBuffer material_ctx = MaterialBuffer(pc.material_ptr);
        MaterialData mat = material_ctx.materials[pc.material_index & 0xFFFFu];
        
        int base_color_idx = mat.texture_indices.x;
        if (base_color_idx >= 0) {
            float alpha = texture(textures[nonuniformEXT(base_color_idx)], inUV).a;
            if (alpha * mat.base_color_factor.a < mat.alpha_cutoff) {
                discard;
            }
        }
    }
    
    float depth = gl_FragCoord.z;
    outMoments = vec2(depth, depth * depth);
}
