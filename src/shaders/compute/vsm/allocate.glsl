#version 450
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../../interop/structures.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    VsmGlobal u_Global = VsmGlobal(push.vsm_ptr);
    VsmAllocationBuffer requests = VsmAllocationBuffer(u_Global.allocation_ptr);

    uint idx = gl_GlobalInvocationID.x;
    if (idx >= requests.count) return;

    VsmPageAllocation alloc = requests.allocations[idx];
    
    // Virtual coordinates
    ivec2 v_coord = ivec2(alloc.virtual_x, alloc.virtual_y);
    uint layer = alloc.layer;
    
    // Pack physical coordinates and flags for the page table
    uint p_packed = (alloc.physical_x & 0xFFFF) | ((alloc.physical_y & 0xFFFF) << 16);
    
    // Write to Page Table
    uint entry_to_write = (alloc.flags == 0) ? 0xFFFFFFFFu : p_packed;
    
    // Use the unified bindless storage image array from structures.glsl
    imageStore(global_storage_uimages_2d_array[nonuniformEXT(u_Global.page_table_storage_index)], ivec3(v_coord, layer), uvec4(entry_to_write, 0, 0, 0)); 
}
