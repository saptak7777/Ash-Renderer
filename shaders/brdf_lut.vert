#version 450

// Fullscreen triangle shader for BRDF LUT generation
layout(location = 0) out vec2 fragTexCoord;

void main() {
    // Generate fullscreen triangle vertices
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    // This covers more than the screen but is clipped to viewport
    fragTexCoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(fragTexCoord * 2.0 - 1.0, 0.0, 1.0);
}
