#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#define SKIP_PUSH_CONSTANTS
#include "interop/structures.glsl"
#include "common/vertex_pulling.glsl"

layout(push_constant) uniform MotionPushConstants {
    mat4 current_mvp;
    mat4 previous_mvp;
    uint64_t vertex_heap_ptr;
} pc;

layout(location = 0) out vec4 outCurrentPos;
layout(location = 1) out vec4 outPreviousPos;

void main() {
    VertexBuffer vertex = load_vertex(pc.vertex_heap_ptr, gl_VertexIndex);
    vec4 worldPos = vec4(vertex.position, 1.0);
    
    // Calculate current and previous clip-space positions
    outCurrentPos = pc.current_mvp * worldPos;
    outPreviousPos = pc.previous_mvp * worldPos;
    
    gl_Position = outCurrentPos;
}
