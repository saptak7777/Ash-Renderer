#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "interop/structures.glsl"
#include "common/vertex_pulling.glsl"

layout(buffer_reference, scalar) restrict readonly buffer ObjectMotionData {
    mat4 current_mvp;
    mat4 previous_mvp;
    uint64_t vertex_ptr;
};

layout(location = 0) out vec4 outCurrentPos;
layout(location = 1) out vec4 outPreviousPos;

void main() {
    ObjectMotionData data = ObjectMotionData(push.transform_ptr);
    
    VertexBuffer vertex = load_vertex(data.vertex_ptr, gl_VertexIndex);
    vec4 worldPos = vec4(vertex.position, 1.0);
    
    // Calculate current and previous clip-space positions
    outCurrentPos = data.current_mvp * worldPos;
    outPreviousPos = data.previous_mvp * worldPos;
    
    gl_Position = outCurrentPos;
}
