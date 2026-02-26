// shadow.vert
// Optimized VSM shadow pass with BDA manual index pulling

#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "interop/structures.glsl"
#include "common/vertex_pulling.glsl"

layout(location = 0) out vec2 outUV;

// Unified ABI: PushConstants block now from structure.glsl
// layout(push_constant) uniform PushConstants { ... } push;

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
    
    // VSM Matrix via BDA
    mat4 lightSpaceMatrix = VsmGlobal(push.vsm_ptr).light_view_projections[push.clipmap_level];
    
    gl_Position = lightSpaceMatrix * modelMatrix * localPosition;
    outUV = inUV;
}
