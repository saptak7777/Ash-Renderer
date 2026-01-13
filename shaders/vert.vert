#version 450
// Force recompile for push constant fix
#extension GL_GOOGLE_include_directive : require

#include "include/structures.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;
layout(location = 4) in vec4 inTangent;
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) centroid out vec3 fragNormal;
layout(location = 3) sample out vec3 fragWorldPos;
layout(location = 4) out vec4 fragPosLightSpace;
layout(location = 5) out vec4 fragTangent;
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

// Set 1: Bindless consolidated resources
#extension GL_EXT_nonuniform_qualifier : enable

// Binding 2: Instances
layout(set = 1, binding = 2) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

void main() {
    mat4 modelMatrix;
    if (push.use_instancing != 0) {
        modelMatrix = instance_buffers[nonuniformEXT(push.instance_buffer_index)].instances[gl_InstanceIndex].model;
    } else {
        modelMatrix = push.model;
    }

    vec4 worldPosition = modelMatrix * vec4(inPosition, 1.0);
    gl_Position = mvp.view_proj * worldPosition;

    fragColor = inColor;
    if (push.use_instancing != 0) {
        fragColor *= instance_buffers[nonuniformEXT(push.instance_buffer_index)].instances[gl_InstanceIndex].color.rgb;
    }
    fragUV = inUV;
    
    mat3 normalMat;
    if (push.use_instancing != 0) {
        // PER-INSTANCE NORMAL MATRIX: Calculate from instance model matrix to fix lighting on rotated objects
        // In Phase 2, we will pass this pre-calculated from CPU to avoid inverse-transpose in shader.
        normalMat = transpose(inverse(mat3(modelMatrix)));
    } else {
        normalMat = mat3(mvp.normal_matrix);
    }
    
    fragNormal = normalize(normalMat * inNormal);
    fragTangent = vec4(normalize(normalMat * inTangent.xyz), inTangent.w);
    
    fragWorldPos = worldPosition.xyz;
    fragPosLightSpace = mvp.light_space_matrix * worldPosition;

    vec4 currentClip = mvp.view_proj * worldPosition;
    vec4 prevClip = mvp.prev_view_proj * worldPosition;
    motionVector = (currentClip.xy / currentClip.w - prevClip.xy / prevClip.w) * 0.5;
}
