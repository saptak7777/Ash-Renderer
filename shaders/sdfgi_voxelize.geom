#version 450

//! SDFGI Voxelization Geometry Shader
//!
//! Projects triangles to dominant axis for conservative voxelization.

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 inWorldPos[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec2 inTexCoord[];

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;
layout(location = 3) flat out uint outAxis;

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec3 cascadeOrigin;
    float voxelSize;
    uint cascadeIndex;
} pc;

void main() {
    // Calculate triangle normal
    vec3 edge1 = inWorldPos[1] - inWorldPos[0];
    vec3 edge2 = inWorldPos[2] - inWorldPos[0];
    vec3 triNormal = abs(normalize(cross(edge1, edge2)));
    
    // Select dominant axis
    uint axis = 0;
    if (triNormal.y > triNormal.x && triNormal.y > triNormal.z) {
        axis = 1;  // Y-axis
    } else if (triNormal.z > triNormal.x && triNormal.z > triNormal.y) {
        axis = 2;  // Z-axis
    }
    
    // Project to 2D based on dominant axis
    for (int i = 0; i < 3; i++) {
        outWorldPos = inWorldPos[i];
        outNormal = inNormal[i];
        outTexCoord = inTexCoord[i];
        outAxis = axis;
        
        vec3 voxelPos = (inWorldPos[i] - pc.cascadeOrigin) / pc.voxelSize;
        
        // Project to dominant axis
        if (axis == 0) {
            gl_Position = vec4(voxelPos.yz, 0.0, 1.0);
        } else if (axis == 1) {
            gl_Position = vec4(voxelPos.xz, 0.0, 1.0);
        } else {
            gl_Position = vec4(voxelPos.xy, 0.0, 1.0);
        }
        
        // Map to [-1, 1] NDC
        gl_Position.xy = gl_Position.xy * 2.0 - 1.0;
        
        EmitVertex();
    }
    
    EndPrimitive();
}
