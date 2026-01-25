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

void main() {
    // Access Frame Data via BDA
    FrameData frame = FrameData(push.frame_ptr);
    // Access Joint Data via BDA
    JointBuffer joint_buffer = JointBuffer(push.joint_ptr);

    // BDA Skinned Vertex Pulling: Load vertex data from global vertex heap
    SkinnedVertexBuffer vertex = load_skinned_vertex(push.vertex_ptr, gl_VertexIndex);
    
    vec3 inPosition = vertex.position;
    vec3 inNormal = vertex.normal;
    vec2 inUV = vertex.uv;
    uvec4 inJointIndices = uvec4(vertex.joint_indices); // Convert u16vec4 to uvec4
    vec4 inJointWeights = vertex.joint_weights;

    // Linear Blend Skinning
    mat4 skinMatrix = mat4(0.0);
    uint base_offset = push.joint_offset;
    
    // Access joints directly from BDA buffer
    skinMatrix += inJointWeights.x * joint_buffer.joints[inJointIndices.x + base_offset];
    skinMatrix += inJointWeights.y * joint_buffer.joints[inJointIndices.y + base_offset];
    skinMatrix += inJointWeights.z * joint_buffer.joints[inJointIndices.z + base_offset];
    skinMatrix += inJointWeights.w * joint_buffer.joints[inJointIndices.w + base_offset];

    vec4 skinnedPosition = skinMatrix * vec4(inPosition, 1.0);
    vec3 skinnedNormal = mat3(skinMatrix) * inNormal;

    vec4 worldPosition = push.model * skinnedPosition;

    gl_Position = frame.view_proj * worldPosition;

    fragColor = vec3(1.0);
    fragUV = inUV;
    // Normal matrix from frame data (or derived from model matrix if instancing, but skinning implies model matrix)
    // For skinning, the normal is transformed by the skinMatrix which includes model transform if joints are world space,
    // or we apply model matrix rotation.
    // The previous code used mvp.normal_matrix * skinnedNormal.
    // However, skinMatrix usually transforms to Model space or World space depending on implementation.
    // If joints are in Model space, we need to apply Model matrix (or Normal matrix) afterwards.
    // Assuming standard glTF: joints are relative to root. push.model transforms to world.
    
    // Previous code:
    // vec4 worldPosition = push.model * skinnedPosition; 
    // gl_Position = mvp.view_proj * worldPosition;
    // mat3 normalMatrix = mat3(mvp.normal_matrix);
    // fragNormal = normalize(normalMatrix * skinnedNormal);
    
    // So we use frame.normal_matrix.
    mat3 normalMatrix = mat3(frame.normal_matrix);
    fragNormal = normalize(normalMatrix * skinnedNormal);
    
    fragWorldPos = worldPosition.xyz;
    fragPosLightSpace = frame.light_space_matrix * worldPosition;

    // Motion vectors for TAA
    vec4 currentClip = frame.view_proj * worldPosition;
    vec4 prevClip = frame.prev_view_proj * worldPosition;
    
    vec2 currentNdc = currentClip.xy / currentClip.w;
    vec2 prevNdc = prevClip.xy / prevClip.w;
    
    motionVector = (currentNdc - prevNdc) * 0.5;
}
