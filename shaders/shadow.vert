// shadow.vert
// Optimized VSM shadow pass with BDA manual index pulling

#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define SKIP_PUSH_CONSTANTS
#include "include/structures.glsl"
#include "include/vertex_pulling.glsl"

layout(location = 0) out vec2 outUV;

// Combined push constants for shadow pass (Lean 128-byte version)
layout(push_constant) uniform ShadowPushConstants {
    uint64_t vertex_ptr;   // 0
    uint64_t instance_ptr; // 8
    uint64_t index_ptr;    // 16
    uint64_t transform_ptr;// 24
    
    uint transform_index;  // 32
    uint use_instancing;   // 36
    
    // Mat4 at offset 64
    layout(offset = 64) mat4 lightSpaceMatrix;
} pc;

void main() {
    mat4 modelMatrix;
    if (pc.transform_ptr != 0) {
        modelMatrix = TransformBuffer(pc.transform_ptr).matrices[pc.transform_index];
    } else {
        modelMatrix = mat4(1.0);
    }
    
    // Manual Index Pulling (Phase 5)
    // We use gl_VertexIndex as the absolute index into the Index Buffer
    uint actualIndex = load_index(pc.index_ptr, gl_VertexIndex);

    int vertexOffset = 0;
    if (pc.use_instancing != 0 && pc.instance_ptr != 0) {
        InstanceBuffer instance_buffer = InstanceBuffer(pc.instance_ptr);
        InstanceData instance = instance_buffer.instances[gl_InstanceIndex];
        modelMatrix = instance.model;
        vertexOffset = int(instance.vertex_offset);
    }
    
    // BDA Vertex Pulling - APPLY BASE VERTEX (Phase 5 Fix)
    // vertex_index = actualIndex (from index buffer) + vertexOffset (BaseVertex)
    VertexBuffer vertex = load_vertex(pc.vertex_ptr, uint(int(actualIndex) + vertexOffset));
    vec4 localPosition = vec4(vertex.position, 1.0);
    vec2 inUV = vertex.uv;
    
    gl_Position = pc.lightSpaceMatrix * modelMatrix * localPosition;
    outUV = inUV;
}
