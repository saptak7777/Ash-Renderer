#version 450

//! Motion Vector Vertex Shader
//!
//! Outputs both current and previous frame clip-space positions
//! for motion vector calculation in the fragment shader.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(push_constant) uniform MotionPushConstants {
    mat4 current_mvp;
    mat4 previous_mvp;
} pc;

layout(location = 0) out vec4 outCurrentPos;
layout(location = 1) out vec4 outPreviousPos;

void main() {
    vec4 worldPos = vec4(inPosition, 1.0);
    
    // Calculate current and previous clip-space positions
    outCurrentPos = pc.current_mvp * worldPos;
    outPreviousPos = pc.previous_mvp * worldPos;
    
    gl_Position = outCurrentPos;
}
