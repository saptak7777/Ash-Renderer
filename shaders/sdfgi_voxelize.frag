#version 450

//! SDFGI Voxelization Fragment Shader
//!
//! Writes voxel data to 3D textures.

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) flat in uint inAxis;

// 3D storage images
layout(set = 0, binding = 0, rgba16f) uniform image3D albedoVoxels;
layout(set = 0, binding = 1, rgba16f) uniform image3D normalVoxels;

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec3 cascadeOrigin;
    float voxelSize;
    uint cascadeIndex;
} pc;

const uint VOXEL_RESOLUTION = 32;

void main() {
    // Convert world position to voxel coordinates
    vec3 voxelPos = (inWorldPos - pc.cascadeOrigin) / pc.voxelSize;
    ivec3 voxelCoord = ivec3(voxelPos * float(VOXEL_RESOLUTION));
    
    // Bounds check
    if (any(lessThan(voxelCoord, ivec3(0))) || any(greaterThanEqual(voxelCoord, ivec3(VOXEL_RESOLUTION)))) {
        discard;
    }
    
    // For now, use simple white albedo and store normal
    // In real implementation, this would sample textures
    vec4 albedo = vec4(0.8, 0.8, 0.8, 1.0);
    vec4 normal = vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
    
    // Write to 3D textures
    // NOTE: In real implementation, would use imageAtomicMax for proper blending
    imageStore(albedoVoxels, voxelCoord, albedo);
    imageStore(normalVoxels, voxelCoord, normal);
}
