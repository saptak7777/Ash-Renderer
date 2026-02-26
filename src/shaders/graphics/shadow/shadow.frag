#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "interop/structures.glsl"

// Shadow map fragment shader - outputs variance moments for VSM
layout(location = 0) in vec2 inUV;

// Output variance moments to color attachment 0 (R32G32_SFLOAT)
layout(location = 0) out vec2 outMoments;


void main() {
    // Alpha testing for transparent materials
    if (push.material_ptr != 0) {
        MaterialBuffer material_ctx = MaterialBuffer(push.material_ptr);
        MaterialData mat = material_ctx.materials[push.material_index & 0xFFFFu];
        
        int base_color_idx = mat.texture_indices.x;
        if (base_color_idx >= 0) {
            float alpha = texture(global_textures[nonuniformEXT(base_color_idx)], inUV).a;
            if (alpha * mat.base_color_factor.a < mat.alpha_cutoff) {
                discard;
            }
        }
    }
    
    float depth = gl_FragCoord.z;
    outMoments = vec2(depth, depth * depth);
}
