#version 450
#extension GL_GOOGLE_include_directive : enable
#line 1 "shaders/shadow.vert"

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require


#line 1 "shaders/include/structures.glsl"




#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

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



























struct HemisphereAmbient {
    vec4 sky_color;
    vec4 ground_color;
};

struct DirectionalLight {
    vec4 direction;
    vec4 color_intensity;
};

struct SceneLighting {
    HemisphereAmbient ambient;
    DirectionalLight directional;
    uint point_light_count;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};
#line 8 "shaders/shadow.vert"
#line 1 "shaders/include/skinned_vertex_pulling.glsl"




#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require



layout(buffer_reference, scalar) buffer SkinnedVertexBuffer {
    vec3 position;
    vec3 normal;
    vec2 uv;
    u16vec4 joint_indices;
    vec4 joint_weights;

};


SkinnedVertexBuffer load_skinned_vertex(uint64_t base_address, uint vertex_index) {

    uint64_t vertex_address = base_address + uint64_t(vertex_index * 56);
    return SkinnedVertexBuffer(vertex_address);
}
#line 9 "shaders/shadow.vert"

layout(location = 0) out vec2 outUV;



layout(push_constant) uniform ShadowPushConstants {

    layout(offset = 0) mat4 model;
    layout(offset = 64) uint joint_offset;
    layout(offset = 68) uint use_instancing;
    layout(offset = 72) uint instance_buffer_index;
    layout(offset = 76) uint joint_buffer_index;
    layout(offset = 80) uint64_t vertex_heap_ptr;


    layout(offset = 160) mat4 lightSpaceMatrix;
} pc;


#extension GL_EXT_nonuniform_qualifier : enable


layout(set = 1, binding = 2) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];


layout(set = 1, binding = 3) readonly buffer JointBuffers {
    mat4 joints[];
} joint_buffers[];
void main() {

    SkinnedVertexBuffer vertex = load_skinned_vertex(pc.vertex_heap_ptr, gl_VertexIndex);

    vec3 inPosition = vertex.position;
    vec2 inUV = vertex.uv;
    uvec4 inJointIndices = uvec4(vertex.joint_indices);
    vec4 inJointWeights = vertex.joint_weights;

    mat4 modelMatrix;
    if (pc.use_instancing != 0) {
        modelMatrix = instance_buffers[nonuniformEXT(pc.instance_buffer_index)].instances[gl_InstanceIndex].model;
    } else {
        modelMatrix = pc.model;
    }

    vec4 localPosition;
    if (inJointWeights.x > 0.0) {
        mat4 skinMatrix =
            inJointWeights.x * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + inJointIndices.x] +
            inJointWeights.y * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + inJointIndices.y] +
            inJointWeights.z * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + inJointIndices.z] +
            inJointWeights.w * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[pc.joint_offset + inJointIndices.w];
        localPosition = skinMatrix * vec4(inPosition, 1.0);
    } else {
        localPosition = vec4(inPosition, 1.0);
    }

    gl_Position = pc.lightSpaceMatrix * modelMatrix * localPosition;
    outUV = inUV;
}
