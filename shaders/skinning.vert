#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in uvec4 inJointIndices;
layout(location = 4) in vec4 inJointWeights;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragWorldPos;
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
    vec4 light_direction;
    vec4 light_color;
    vec4 ambient_color;
} mvp;

layout(push_constant) uniform PushConstants {
    mat4 model;
    uint joint_offset;
    uint _padding;
} push;

layout(set = 5, binding = 0) readonly buffer JointMatrices {
    mat4 joints[];
};

void main() {
    // Linear Blend Skinning
    mat4 skinMatrix = mat4(0.0);
    uint base_offset = push.joint_offset;
    skinMatrix += inJointWeights.x * joints[inJointIndices.x + base_offset];
    skinMatrix += inJointWeights.y * joints[inJointIndices.y + base_offset];
    skinMatrix += inJointWeights.z * joints[inJointIndices.z + base_offset];
    skinMatrix += inJointWeights.w * joints[inJointIndices.w + base_offset];

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
