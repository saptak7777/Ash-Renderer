#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#define SKIP_PUSH_CONSTANTS
#include "include/structures.glsl"
#include "include/vertex_pulling.glsl"
#include "include/skinned_vertex_pulling.glsl"

layout(location = 0) out vec2 outUV;

// Push constants matching DrawPushConstants layout
layout(push_constant) uniform ShadowPushConstants {
    // Vertex stage (0-127) - matches DrawPushConstants
    layout(offset = 0) mat4 model;
    layout(offset = 64) uint joint_offset;
    layout(offset = 68) uint use_instancing;
    layout(offset = 72) uint instance_buffer_index;
    layout(offset = 76) uint joint_buffer_index;
    layout(offset = 80) uint64_t vertex_heap_ptr;
    layout(offset = 88) uint is_skinned;
    
    // VSM-specific (160-223)
    layout(offset = 160) mat4 lightSpaceMatrix;
} pc;

// Bindless resources (Set 1) requires nonuniform qualifier
#extension GL_EXT_nonuniform_qualifier : enable

// Binding 2: Instances
layout(set = 1, binding = 2) readonly buffer InstanceBuffers {
    InstanceData instances[];
} instance_buffers[];

// Binding 3: Joint Matrices (Skinned Animation)
layout(set = 1, binding = 3) readonly buffer JointBuffers {
    mat4 joints[];
} joint_buffers[];

void main() {
    mat4 modelMatrix = pc.model;
    if (pc.use_instancing != 0) {
        modelMatrix = instance_buffers[nonuniformEXT(pc.instance_buffer_index)].instances[gl_InstanceIndex].model;
    }

    vec4 localPosition;
    vec2 inUV;

    if (pc.is_skinned != 0) {
        // BDA Skinned Vertex Pulling (56-byte stride)
        SkinnedVertexBuffer vertex = load_skinned_vertex(pc.vertex_heap_ptr, gl_VertexIndex);
        
        vec3 inPosition = vertex.position;
        inUV = vertex.uv;
        
        uint joint_base = pc.joint_offset;
        if (pc.use_instancing != 0) {
            joint_base = instance_buffers[nonuniformEXT(pc.instance_buffer_index)].instances[gl_InstanceIndex].draw_index;
        }

        mat4 skinMatrix = 
            vertex.joint_weights.x * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[joint_base + vertex.joint_indices.x] +
            vertex.joint_weights.y * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[joint_base + vertex.joint_indices.y] +
            vertex.joint_weights.z * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[joint_base + vertex.joint_indices.z] +
            vertex.joint_weights.w * joint_buffers[nonuniformEXT(pc.joint_buffer_index)].joints[joint_base + vertex.joint_indices.w];
            
        localPosition = skinMatrix * vec4(inPosition, 1.0);
    } else {
        // BDA Static Vertex Pulling (60-byte stride)
        VertexBuffer vertex = load_vertex(pc.vertex_heap_ptr, gl_VertexIndex);
        localPosition = vec4(vertex.position, 1.0);
        inUV = vertex.uv;
    }
    
    gl_Position = pc.lightSpaceMatrix * modelMatrix * localPosition;
    outUV = inUV;
}
