#version 450

// Shadow map fragment shader - depth-only, no color output
// The fragment shader can be empty for depth-only passes,
// but we include it for explicit control.

layout(location = 0) in vec2 inUV;

// Binding 0 in Set 1 (matches material_texture_layout binding 0)
// We need to match the descriptor set layout index used in the pipeline
layout(set = 1, binding = 0) uniform sampler2D textureSampler;

void main() {
    float alpha = texture(textureSampler, inUV).a;
    if (alpha < 0.1) {
        discard;
    }
}
