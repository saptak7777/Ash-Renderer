#version 450
// 128x128 Page Size / 16x16 Local Size = 8x8 Workgroups per Page
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

struct PageAllocation {
    uint virtual_x;
    uint virtual_y;
    uint physical_x;
    uint physical_y;
    uint layer;
    uint flags;
    uint _padding[2];
};

layout(std430, set = 0, binding = 2) buffer AllocationBuffer {
    uint count;
    PageAllocation allocations[];
};

// The Physical Memory Texture (Atlas)
layout(set = 0, binding = 4, rg32f) uniform image2D u_PhysicalMemory;

void main() {
    // Z-dimension of WorkGroup ID corresponds to the Allocation Index
    uint alloc_idx = gl_WorkGroupID.z;
    if (alloc_idx >= count) return;

    PageAllocation alloc = allocations[alloc_idx];
    
    // Calculate physical pixel coordinate
    ivec2 p_base = ivec2(alloc.physical_x, alloc.physical_y);
    p_base *= 128; // Scale by page size

    ivec2 pixel_offset = ivec2(gl_LocalInvocationID.xy); 
    ivec2 target_coord = p_base + pixel_offset;

    // Clear to Max Depth (1.0) and Zero Moments
    imageStore(u_PhysicalMemory, target_coord, vec4(1.0, 1.0, 0.0, 0.0));
}
