#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "interop/structures.glsl"
#include "common/vertex_pulling.glsl"

layout(location = 0) out vec3 outUVW;

void main() {
    FrameData frame = FrameData(push.frame_ptr);
    VertexBuffer vertex = load_vertex(push.vertex_ptr, gl_VertexIndex);
    vec3 pos = vertex.position;
    
    outUVW = pos;
    // Remove translation from view matrix to keep skybox centered on camera
    mat4 view_no_pos = frame.view;
    view_no_pos[3] = vec4(0.0, 0.0, 0.0, 1.0);
    
    vec4 clip_pos = frame.projection * view_no_pos * vec4(pos, 1.0);
    
    // Force depth to 0.0 (far plane for Reverse-Z)
    gl_Position = clip_pos.xyww;
    gl_Position.z = 0.0;
}
