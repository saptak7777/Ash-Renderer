#version 450

//! SDFGI Voxelization Vertex Shader
//!
//! Transforms vertices for voxelization pass.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;        // View-projection matrix
    vec3 cascadeOrigin;   // Cascade origin
    float voxelSize;      // Voxel size
    uint cascadeIndex;    // Cascade index
} pc;

void main() {
    outWorldPos = inPosition;
    outNormal = inNormal;
    outTexCoord = inTexCoord;
    
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
}
