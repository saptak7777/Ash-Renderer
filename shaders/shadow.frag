#version 450
#extension GL_EXT_nonuniform_qualifier : enable

// Shadow map fragment shader - depth-only, no color output
// The fragment shader can be empty for depth-only passes,
// Explicitly included for precise control over the fragment stage.

layout(location = 0) in vec2 inUV;

layout(push_constant) uniform PushConstants {
    // Skip Vertex push constants (0-143)
    // We'll put base_color_index at offset 144
    layout(offset = 144) int base_color_index; 
} pc;

// Binding 0: Textures (Set 1)
layout(set = 1, binding = 0) uniform sampler2D textures[];

void main() {
    if (pc.base_color_index >= 0) {
        float alpha = texture(textures[nonuniformEXT(pc.base_color_index)], inUV).a;
        if (alpha < 0.1) {
            discard;
        }
    }
}
