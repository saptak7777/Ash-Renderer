#version 450
// BDA Skinned Vertex Pulling Implementation
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "include/structures.glsl"
#include "include/skinned_vertex_pulling.glsl"

// Output attributes
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) centroid out vec3 fragNormal;
layout(location = 3) sample out vec3 fragWorldPos;
layout(location = 4) out vec4 fragPosLightSpace;
layout(location = 6) out vec2 motionVector;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 view_proj;
    mat4 prev_view_proj;
    mat4 light_space_matrix;
    mat4 normal_matrix;
    vec4 camera_pos;
    SceneLighting scene_lighting;
} mvp;

layout(set = 1, binding = 3) readonly buffer JointBuffers {
    mat4 joints[];
} joint_buffers[];

void main() {
    // BDA Skinned Vertex Pulling: Load vertex data from global vertex heap
    SkinnedVertexBuffer vertex = load_skinned_vertex(push.vertex_heap_ptr, gl_VertexIndex);
    
    vec3 inPosition = vertex.position;
    vec3 inNormal = vertex.normal;
    vec2 inUV = vertex.uv;
    uvec4 inJointIndices = uvec4(vertex.joint_indices); // Convert u16vec4 to uvec4
    vec4 inJointWeights = vertex.joint_weights;

    // Linear Blend Skinning
    mat4 skinMatrix = mat4(0.0);
    uint base_offset = push.joint_offset;
    skinMatrix += inJointWeights.x * joint_buffers[nonuniformEXT(push.joint_buffer_index)].joints[inJointIndices.x + base_offset];
    skinMatrix += inJointWeights.y * joint_buffers[nonuniformEXT(push.joint_buffer_index)].joints[inJointIndices.y + base_offset];
    skinMatrix += inJointWeights.z * joint_buffers[nonuniformEXT(push.joint_buffer_index)].joints[inJointIndices.z + base_offset];
    skinMatrix += inJointWeights.w * joint_buffers[nonuniformEXT(push.joint_buffer_index)].joints[inJointIndices.w + base_offset];

    vec4 skinnedPosition = skinMatrix * vec4(inPosition, 1.0);
    vec3 skinnedNormal = mat3(skinMatrix) * inNormal;

    vec4 worldPosition = push.model * skinnedPosition;

    gl_Position = mvp.view_proj * worldPosition;

    fragColor = vec3(1.0);
    fragUV = inUV;
    mat3 normalMatrix = mat3(mvp.normal_matrix);
    fragNormal = normalize(normalMatrix * skinnedNormal);
    fragWorldPos = worldPosition.xyz;
    fragPosLightSpace = mvp.light_space_matrix * worldPosition;

    // Motion vectors for TAA
    vec4 currentClip = mvp.view_proj * worldPosition;
    vec4 prevClip = mvp.prev_view_proj * worldPosition;
    
    vec2 currentNdc = currentClip.xy / currentClip.w;
    vec2 prevNdc = prevClip.xy / prevClip.w;
    
    motionVector = (currentNdc - prevNdc) * 0.5;
}
