// shaders/include/skinned_vertex_pulling.glsl
// BDA Skinned Vertex Pulling Helper
// Requires GL_EXT_buffer_reference and GL_EXT_scalar_block_layout

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

// SkinnedVertex struct matching Rust's SkinnedVertex (56 bytes, tightly packed)
// CRITICAL: Uses scalar layout to match Rust's #[repr(C)] memory layout
layout(buffer_reference, scalar) buffer SkinnedVertexBuffer {
    vec3 position;        // offset 0, 12 bytes
    vec3 normal;          // offset 12, 12 bytes
    vec2 uv;              // offset 24, 8 bytes
    u16vec4 joint_indices; // offset 32, 8 bytes (u16x4)
    vec4 joint_weights;   // offset 40, 16 bytes
    // Total: 56 bytes
};

// Load skinned vertex data from BDA pointer
SkinnedVertexBuffer load_skinned_vertex(uint64_t base_address, uint vertex_index) {
    // BDA SAFETY: Check for null pointer to prevent DEVICE_LOST
    if (base_address == 0) {
        // Return a safe zero address that will be caught by the caller
        return SkinnedVertexBuffer(0);
    }
    
    // Calculate vertex address: base + (index * 56)
    uint64_t vertex_address = base_address + uint64_t(vertex_index * 56);
    return SkinnedVertexBuffer(vertex_address);
}

// Safe skinned vertex data fetch with null check
bool load_skinned_vertex_safe(uint64_t base_address, uint vertex_index, out vec3 position, out vec3 normal, out vec2 uv, out u16vec4 joint_indices, out vec4 joint_weights) {
    if (base_address == 0) {
        // Return safe default values
        position = vec3(0.0);
        normal = vec3(0.0, 1.0, 0.0);
        uv = vec2(0.0);
        joint_indices = u16vec4(0);
        joint_weights = vec4(0.0);
        return false;
    }
    
    SkinnedVertexBuffer vertex = load_skinned_vertex(base_address, vertex_index);
    position = vertex.position;
    normal = vertex.normal;
    uv = vertex.uv;
    joint_indices = vertex.joint_indices;
    joint_weights = vertex.joint_weights;
    return true;
}
