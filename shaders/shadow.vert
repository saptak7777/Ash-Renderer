#version 450
#extension GL_EXT_nonuniform_qualifier : enable

// Shadow map vertex shader - transforms vertices to light space

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inJointIndices;
layout(location = 3) in vec4 inJointWeights;

layout(location = 0) out vec2 outUV;

// Light-space matrix (projection * view from light's POV)
layout(push_constant) uniform PushConstants {
    // Vertex stage (0-143)
    layout(offset = 0) mat4 lightSpaceMatrix;
    layout(offset = 64) mat4 model;
    layout(offset = 128) uint joint_offset;
    layout(offset = 132) uint use_instancing;
    layout(offset = 136) uint instance_buffer_index;
    layout(offset = 140) uint joint_buffer_index;
} pc;

// Bindless resources (Set 1)
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

// Binding 2: Instances
layout(set = 1, binding = 2) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

// Binding 3: Joint Matrices
layout(set = 1, binding = 3) readonly buffer JointBuffers {
    mat4 joints[];
} joint_buffers[];

void main() {
    mat4 modelMatrix;
    if (pc.use_instancing != 0) {
        modelMatrix = instance_buffers[nonuniformEXT(pc.instance_buffer_index)].instances[gl_InstanceIndex].model;
    } else {
        modelMatrix = pc.model;
    }

    vec4 localPosition;
    if (inJointWeights.x > 0.0) {
        mat4 skinMatrix = 
            inJointWeights.x * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + uint(inJointIndices.x)] +
            inJointWeights.y * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + uint(inJointIndices.y)] +
            inJointWeights.z * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + uint(inJointIndices.z)] +
            inJointWeights.w * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + uint(inJointIndices.w)];
        localPosition = skinMatrix * vec4(inPosition, 1.0);
    } else {
        localPosition = vec4(inPosition, 1.0);
    }

    gl_Position = pc.lightSpaceMatrix * modelMatrix * localPosition;
    outUV = inUV;
}
