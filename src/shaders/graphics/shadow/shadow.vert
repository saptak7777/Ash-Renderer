#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define SHADER_PUSH_CONSTANT_OVERRIDE
#include "interop/structures.glsl"
#include "common/vertex_pulling.glsl"

layout(location = 0) out vec2 outUV;

// Specialized Shadow Push Constants (Matching Rust ShadowPushConstants)
layout(push_constant) uniform ShadowPushBlock {
    uint64_t frame_ptr;        // 0
    uint64_t vertex_ptr;       // 8
    uint64_t instance_ptr;     // 16
    uint64_t index_ptr;        // 24
    uint64_t transform_ptr;    // 32
    uint transform_index;      // 40
    uint use_instancing;       // 44
    uint clipmap_level;        // 48
    // Offset 64 ensures 16-byte alignment for mat4 and matches Rust padding
    layout(offset = 64) mat4 light_space_matrix; 
} push;

void main() {
    // Access Frame Data via BDA
    FrameData frame = FrameData(push.frame_ptr);
    
    mat4 modelMatrix;
    if (push.transform_ptr != 0) {
        modelMatrix = TransformBuffer(push.transform_ptr).matrices[push.transform_index];
    } else {
        modelMatrix = mat4(1.0);
    }
    
    // Manual Index Pulling (Phase 5)
    // We use gl_VertexIndex as the absolute index into the Index Buffer
    uint actualIndex = load_index(push.index_ptr, gl_VertexIndex);

    int vertexOffset = 0;
    if (push.use_instancing != 0 && push.instance_ptr != 0) {
        InstanceBuffer instance_buffer = InstanceBuffer(push.instance_ptr);
        InstanceData instance = instance_buffer.instances[gl_InstanceIndex];
        modelMatrix = instance.model;
        vertexOffset = int(instance.vertex_offset);
    }
    
    // BDA Vertex Pulling - APPLY BASE VERTEX (Phase 5 Fix)
    // vertex_index = actualIndex (from index buffer) + vertexOffset (BaseVertex)
    VertexBuffer vertex = load_vertex(push.vertex_ptr, uint(int(actualIndex) + vertexOffset));
    vec4 localPosition = vec4(vertex.position, 1.0);
    vec2 inUV = vertex.uv;
    
    gl_Position = push.light_space_matrix * modelMatrix * localPosition;
    outUV = inUV;
}
