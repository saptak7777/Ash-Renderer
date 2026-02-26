#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "interop/structures.glsl"

layout(location = 0) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Correctly sample from bindless cubemap array
    // skybox_index is passed from Rust via push constants
    // nonuniformEXT is required for dynamically indexed bindless arrays
    vec4 skyColor = texture(global_cubemaps[nonuniformEXT(push.skybox_index)], normalize(fragPos));
    
    // Simple tone mapping / gamma correction if needed, 
    // but the main shader handles HDR-to-SDR for the whole scene.
    outColor = skyColor;
}
