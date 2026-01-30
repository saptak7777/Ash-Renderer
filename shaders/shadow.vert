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

// Combined push constants for shadow pass
layout(push_constant) uniform ShadowPushConstants {
    // Pointer stage (0-55)
    uint64_t frame_ptr;    // 0
    uint64_t vertex_ptr;   // 8
    uint64_t instance_ptr; // 16
    uint64_t material_ptr; // 24
    uint64_t index_ptr;    // 32
    uint64_t light_ptr;    // 40
    uint64_t tile_ptr;     // 48

    // Texture indices (56-63)
    uint vsm_page_index;   // 56
    uint vsm_cache_index;  // 60

    // Control his (64-127)
    layout(offset = 64) mat4 model; 
    
    // Material & Flags (128-159)
    layout(offset = 128) uint material_index;
    layout(offset = 132) uint use_instancing;
    layout(offset = 136) uint flags;
    layout(offset = 140) uint debug_path;
    layout(offset = 144) uint debug_visualization_enabled;
    layout(offset = 148) uint skybox_index;
    
    // VSM-specific (160-223)
    layout(offset = 160) mat4 lightSpaceMatrix;
} pc;

void main() {
    mat4 modelMatrix = pc.model;
    
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
