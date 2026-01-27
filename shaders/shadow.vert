// shadow.vert
// Optimized VSM shadow pass with BDA vertex pulling

#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define SKIP_PUSH_CONSTANTS
#include "include/structures.glsl"
#include "include/vertex_pulling.glsl"

layout(location = 0) out vec2 outUV;

// Combined push constants for shadow pass
layout(push_constant) uniform ShadowPushConstants {
    // Standard DrawPushConstants (0-159)
    uint64_t frame_ptr;
    uint64_t vertex_ptr;
    uint64_t instance_ptr;
    uint64_t material_ptr;
    uint64_t _ptr_padding[2];

    layout(offset = 48) mat4 model; 
    layout(offset = 112) uint material_index;
    layout(offset = 116) uint use_instancing;
    layout(offset = 120) uint _unused_flags[2];

    layout(offset = 128) uint flags;
    layout(offset = 132) uint debug_path;
    layout(offset = 136) uint debug_visualization_enabled;
    
    // VSM-specific (160-223)
    layout(offset = 160) mat4 lightSpaceMatrix;
} pc;

void main() {
    mat4 modelMatrix = pc.model;
    int vertex_offset = 0;
    
    if (pc.use_instancing != 0 && pc.instance_ptr != 0) {
        InstanceBuffer instance_buffer = InstanceBuffer(pc.instance_ptr);
        InstanceData instance = instance_buffer.instances[gl_InstanceIndex];
        modelMatrix = instance.model;
        vertex_offset = instance.vertex_offset;
    }

    // BDA Static Vertex Pulling
    VertexBuffer vertex = load_vertex(pc.vertex_ptr, gl_VertexIndex + vertex_offset);
    vec4 localPosition = vec4(vertex.position, 1.0);
    vec2 inUV = vertex.uv;
    
    gl_Position = pc.lightSpaceMatrix * modelMatrix * localPosition;
    outUV = inUV;
}
