#version 450

// Shadow map vertex shader - transforms vertices to light space

layout(location = 0) in vec3 inPosition;

layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 outUV;

// Light-space matrix (projection * view from light's POV)
layout(push_constant) uniform PushConstants {
    mat4 lightSpaceMatrix;
    mat4 model;
} pc;

void main() {
    gl_Position = pc.lightSpaceMatrix * pc.model * vec4(inPosition, 1.0);
    outUV = inUV;
}
