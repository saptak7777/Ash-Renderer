#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;
layout(location = 3) out vec3 fragWorldPos;
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
    vec4 light_direction;
    vec4 light_color;
    vec4 ambient_color;
} mvp;

// Set 1: Bindless consolidated resources
struct InstanceData {
    vec4 bounds_center;
    vec4 bounds_extents;
    mat4 model;
    uint draw_index;
    uint first_index;
    uint index_count;
    int vertex_offset;
    vec4 color;
    vec4 custom;
    uint cluster_offset;
    uint cluster_count;
    uint flags;
    uint _padding;
};

#extension GL_EXT_nonuniform_qualifier : enable

// Binding 2: Instances
layout(set = 1, binding = 2) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

layout(push_constant) uniform PushConstants {
    // Vertex stage (0-127)
    layout(offset = 0) mat4 model;
    layout(offset = 64) uint joint_offset;
    layout(offset = 68) uint use_instancing;
    layout(offset = 72) uint instance_buffer_index;
    layout(offset = 76) uint joint_buffer_index;

    // Fragment stage (128-255)
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint _material_padding[3];
} push;

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
    
    mat3 normalMatrix = mat3(transpose(inverse(modelMatrix)));
    fragNormal = normalize(normalMatrix * inNormal);
    fragTangent = vec4(normalize(normalMatrix * inTangent.xyz), inTangent.w);
    
    fragWorldPos = worldPosition.xyz;
    fragPosLightSpace = mvp.light_space_matrix * worldPosition;

    vec4 currentClip = mvp.view_proj * worldPosition;
    vec4 prevClip = mvp.prev_view_proj * worldPosition;
    motionVector = (currentClip.xy / currentClip.w - prevClip.xy / prevClip.w) * 0.5;
}
