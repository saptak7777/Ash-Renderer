#version 450

layout(location = 0) out vec2 outUV;

void main() {
    // Generate a full-screen triangle
    // (0,0) -> (2,0) -> (0,2) in NDC covers the screen
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
    
    // Flip Y for Vulkan (NDC Y is down)
    // Actually, for full-screen quads often we just use this and it maps correctly
    // or we might need gl_Position.y *= -1.0;
    // Let's stick to standard fullscreen triangle for now.
}
