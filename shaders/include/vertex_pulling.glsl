// shaders/include/vertex_pulling.glsl
// BDA Vertex Pulling Helper
// Requires GL_EXT_buffer_reference and GL_EXT_scalar_block_layout

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Vertex struct matching Rust's Vertex (60 bytes, tightly packed)
// CRITICAL: Uses scalar layout to match Rust's #[repr(C)] memory layout
layout(buffer_reference, scalar) buffer VertexBuffer {
    vec3 position;    // offset 0, 12 bytes
    vec3 normal;      // offset 12, 12 bytes
    vec2 uv;          // offset 24, 8 bytes
    vec3 color;       // offset 32, 12 bytes
    vec4 tangent;     // offset 44, 16 bytes
    // Total: 60 bytes
};

// Load vertex data from BDA pointer
VertexBuffer load_vertex(uint64_t base_address, uint vertex_index) {
    // BDA SAFETY: Check for null pointer to prevent DEVICE_LOST
    if (base_address == 0) {
        // Return a safe zero address that will be caught by the caller
        return VertexBuffer(0);
    }
    
    // Calculate vertex address: base + (index * 60)
    uint64_t vertex_address = base_address + uint64_t(vertex_index * 60);
    return VertexBuffer(vertex_address);
}

// Safe vertex data fetch with null check
bool load_vertex_safe(uint64_t base_address, uint vertex_index, out vec3 position, out vec3 normal, out vec2 uv, out vec3 color, out vec4 tangent) {
    if (base_address == 0) {
        // Return safe default values
        position = vec3(0.0);
        normal = vec3(0.0, 1.0, 0.0);
        uv = vec2(0.0);
        color = vec3(1.0);
        tangent = vec4(1.0, 0.0, 0.0, 1.0);
        return false;
    }
    
    VertexBuffer vertex = load_vertex(base_address, vertex_index);
    position = vertex.position;
    normal = vertex.normal;
    uv = vertex.uv;
    color = vertex.color;
    tangent = vertex.tangent;
    return true;
}
