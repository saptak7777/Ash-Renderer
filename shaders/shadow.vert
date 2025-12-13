#version 450

// Shadow map vertex shader - transforms vertices to light space

layout(location = 0) in vec3 inPosition;

// Light-space matrix (projection * view from light's POV)
layout(push_constant) uniform PushConstants {
    mat4 lightSpaceMatrix;
    mat4 model;
} pc;

void main() {
    gl_Position = pc.lightSpaceMatrix * pc.model * vec4(inPosition, 1.0);
}
