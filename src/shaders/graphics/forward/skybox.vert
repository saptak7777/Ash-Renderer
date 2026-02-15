#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define SKIP_PUSH_CONSTANTS
#include "interop/structures.glsl"
#include "common/vertex_pulling.glsl"

layout(push_constant) uniform SkyboxPush {
    uint64_t frame_ptr;      // Offset 0
    uint skybox_index;       // Offset 8
    layout(offset = 80) uint64_t vertex_ptr; // Offset 80
} push;

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
    
    // Force depth to 1.0 (far plane for Standard Z: Near=0, Far=1)
    // Clear depth is 1.0, and depth test is LESS_OR_EQUAL.
    gl_Position = clip_pos.xyww;
}
