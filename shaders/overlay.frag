#version 450

// Simple overlay fragment shader - solid color output

layout(location = 0) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;
}
