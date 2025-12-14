#version 450

// Shadow map fragment shader - depth-only, no color output
// The fragment shader can be empty for depth-only passes,
// but we include it for explicit control.

layout(location = 0) in vec2 inUV;

layout(push_constant) uniform PushConstants {
    layout(offset = 128) int base_color_index; // Offset 128 to skip Vertex push constants
} pc;

#extension GL_EXT_nonuniform_qualifier : require
layout(set = 2, binding = 0) uniform sampler2D textures[];

void main() {
    if (pc.base_color_index >= 0) {
        float alpha = texture(textures[nonuniformEXT(pc.base_color_index)], inUV).a;
        if (alpha < 0.1) {
            discard;
        }
    }
}
