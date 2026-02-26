#version 450
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../../interop/structures.glsl"

// 128x128 Page Size / 16x16 Local Size = 8x8 Workgroups per Page
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    VsmGlobal u_Global = VsmGlobal(push.vsm_ptr);
    VsmAllocationBuffer requests = VsmAllocationBuffer(u_Global.allocation_ptr);

    // Z-dimension of WorkGroup ID corresponds to the Allocation Index
    uint alloc_idx = gl_WorkGroupID.z;
    if (alloc_idx >= requests.count) return;

    VsmPageAllocation alloc = requests.allocations[alloc_idx];
    
    // Calculate physical pixel coordinate
    ivec2 p_base = ivec2(alloc.physical_x, alloc.physical_y);
    p_base *= 128; // Scale by page size

    ivec2 pixel_offset = ivec2(gl_LocalInvocationID.xy); 
    ivec2 target_coord = p_base + pixel_offset;

    // Clear to Max Depth (1.0) and Zero Moments
    // Use global_storage_images from structures.glsl
    imageStore(global_storage_images[nonuniformEXT(u_Global.physical_cache_storage_index)], target_coord, vec4(1.0, 1.0, 0.0, 0.0));
}
