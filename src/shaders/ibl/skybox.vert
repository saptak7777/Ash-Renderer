#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 outPos;

layout(push_constant) uniform PushConstants {
    mat4 view;
    mat4 projection;
} push;

void main() {
    outPos = inPosition;
    gl_Position = push.projection * push.view * vec4(inPosition, 1.0);
}
