#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct PageAllocation {
    uint virtual_x;
    uint virtual_y;
    uint physical_x;
    uint physical_y;
    uint layer;
    uint flags;
    uint _padding[2];
};

// Input: List of new allocations from CPU
layout(std430, set = 0, binding = 2) buffer AllocationBuffer {
    uint count; // Header
    PageAllocation allocations[];
};

// Output: The Page Table (R32UI)
// Maps Virtual Coords -> Physical Coords + Flags
layout(set = 0, binding = 3, r32ui) uniform uimage2DArray u_PageTable;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= count) return;

    PageAllocation alloc = allocations[idx];
    
    // Virtual coordinates
    ivec2 v_coord = ivec2(alloc.virtual_x, alloc.virtual_y);
    uint layer = alloc.layer;
    
    // Pack physical coordinates and flags for the page table
    // Format: physical_x (16 bits) | physical_y (16 bits)
    // Using 16/16 packing to match the sampler and support larger physical caches.
    uint p_packed = (alloc.physical_x & 0xFFFF) | ((alloc.physical_y & 0xFFFF) << 16);
    
    // Write to Page Table
    // If flags == 0, this is an invalidation request for an evicted page.
    uint entry_to_write = (alloc.flags == 0) ? 0xFFFFFFFFu : p_packed;
    imageStore(u_PageTable, ivec3(v_coord, layer), uvec4(entry_to_write, 0, 0, 0)); 
}
