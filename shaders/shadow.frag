#version 450
#extension GL_EXT_nonuniform_qualifier : enable

// Shadow map fragment shader - outputs depth to R32_SFLOAT color attachment
// VSM uses a physical cache texture (Color) instead of a Depth Buffer

layout(location = 0) in vec2 inUV;

// Output depth to color attachment (R32_SFLOAT)
layout(location = 0) out float outDepth;

layout(push_constant) uniform PushConstants {
    // Skip Vertex push constants (0-143)
    // We'll put base_color_index at offset 144
    layout(offset = 144) int base_color_index; 
} pc;

// Binding 0: Textures (Set 1)
layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {
    // Alpha testing for transparent materials
    if (pc.base_color_index >= 0) {
        float alpha = texture(textures[nonuniformEXT(pc.base_color_index)], inUV).a;
        if (alpha < 0.1) {
            discard;
        }
    }
    
    // Write depth to color attachment
    outDepth = gl_FragCoord.z;
}
