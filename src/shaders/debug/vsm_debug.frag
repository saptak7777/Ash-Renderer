#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Set 0: Debug textures
layout(set = 0, binding = 0, r32ui) uniform uimage2DArray u_PageTable;
layout(set = 0, binding = 1, rg32f) uniform image2D u_PhysicalMemory;

layout(push_constant) uniform PC {
    uint mode;  // 0: Physical Atlas, 1: Page Table
    float layer;
} pc;

vec3 hash(uint x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    
    float r = float(x & 0xFF) / 255.0;
    float g = float((x >> 8) & 0xFF) / 255.0;
    float b = float((x >> 16) & 0xFF) / 255.0;
    return vec3(r, g, b);
}

void main() {
    if (pc.mode == 0) {
        // Mode 0: Physical Atlas
        ivec2 size = imageSize(u_PhysicalMemory);
        ivec2 coord = ivec2(inUV * vec2(size));
        vec4 data = imageLoad(u_PhysicalMemory, coord);
        
        // Depth is in Red channel
        float depth = data.r;
        
        // Visualize depth with high contrast (Reverse-Z)
        // If depth is 1.0 (clear), it will be 1.0. 
        // We use pow to see the small variations near 1.0
        float visual = pow(depth, 100.0);
        outColor = vec4(vec3(visual), 1.0);
        
        // Debug: use green for Variance component (data.g)
        // outColor.g = data.g * 10.0;
    } else {
        // Mode 1: Page Table
        ivec3 size = imageSize(u_PageTable);
        ivec3 coord = ivec3(ivec2(inUV * vec2(size.xy)), int(pc.layer));
        uint value = imageLoad(u_PageTable, coord).r;
        
        if (value == 0) {
            outColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            outColor = vec4(hash(value), 1.0);
        }
    }
}
