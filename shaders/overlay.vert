#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar) readonly buffer TextVertexHeap {
    vec2 pos;
    vec2 uv;
    vec4 color;
};

layout(push_constant) uniform PushConstants {
    uint64_t vertex_heap_ptr;
} pc;

layout(location = 0) out vec4 fragColor;

void main() {
    TextVertexHeap vertex = TextVertexHeap(pc.vertex_heap_ptr + gl_VertexIndex * 32);
    gl_Position = vec4(vertex.pos, 0.0, 1.0);
    fragColor = vertex.color;
}
