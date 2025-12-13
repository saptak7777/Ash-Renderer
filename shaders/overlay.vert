#version 450

// Simple overlay vertex shader for diagnostics text
// Input: 2D position in NDC, color

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

void main() {
    gl_Position = vec4(inPos, 0.0, 1.0);
    fragColor = inColor;
}
