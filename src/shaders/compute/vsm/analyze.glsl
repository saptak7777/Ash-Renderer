#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
// Default limit. Matches VsmConfig in Rust. Overridden via Specialization Constants at pipeline creation.
layout(constant_id = 0) const uint MAX_REQUESTS = 1024;

layout(std140, set = 0, binding = 0) uniform VsmGlobal {
    mat4 light_view_projections[16];
    mat4 view_proj;
    mat4 inv_view_proj;
    vec4 camera_position;
    vec4 light_dir;
    uint page_table_size;
    uint _pad0;
    uint _pad1;
    uint _pad2;
} u_Global;

struct PageRequest {
    uint virtual_x;
    uint virtual_y;
    float priority;
    uint layer;
};

layout(std430, set = 0, binding = 1) buffer RequestBuffer {
    uint count;
    uint overflow_count;
    uint _padding[2];
    PageRequest data[];
} requests;

// Page Table (R32UI)
layout(set = 0, binding = 3, r32ui) uniform uimage2DArray u_PageTable;

// Scene Depth Buffer
layout(set = 0, binding = 5) uniform sampler2D u_SceneDepth;

void main() {
    ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 screen_size = textureSize(u_SceneDepth, 0);
    
    if (pixel_coord.x >= screen_size.x || pixel_coord.y >= screen_size.y) return;

    vec2 uv = (vec2(pixel_coord) + 0.5) / vec2(screen_size);

    // 1. Sample Depth
    float depth = texture(u_SceneDepth, uv).r;
    
    // Skip skybox
    if (depth >= 1.0) return; 

    // 2. Reconstruct World Position
    vec4 clip_pos = vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, depth, 1.0);
    vec4 view_pos = u_Global.inv_view_proj * clip_pos;
    vec4 world_pos = view_pos / view_pos.w;

    // 3. Project to Light Space (Level 0 for now)
    vec4 shadow_pos = u_Global.light_view_projections[0] * world_pos;
    
    // 4. Check Bounds
    if (shadow_pos.x >= -1.0 && shadow_pos.x <= 1.0 &&
        shadow_pos.y >= -1.0 && shadow_pos.y <= 1.0 &&
        shadow_pos.z >= -1.0 && shadow_pos.z <= 1.0) 
    {
        // 5. Convert to Page Coordinates
        vec2 page_uv = shadow_pos.xy * 0.5 + 0.5;
        uint page_x = uint(page_uv.x * float(u_Global.page_table_size));
        uint page_y = uint(page_uv.y * float(u_Global.page_table_size));
        
        page_x = min(page_x, u_Global.page_table_size - 1);
        page_y = min(page_y, u_Global.page_table_size - 1);
        
        uint layer = 0;
        
        // 6. Check if already allocated or needs update (Simplified for now)
        // We'll just request and let PageManager handle deduplication.
        
        uint idx = atomicAdd(requests.count, 1);
        if (idx < MAX_REQUESTS) { 
            requests.data[idx].virtual_x = page_x;
            requests.data[idx].virtual_y = page_y;
            requests.data[idx].priority = distance(world_pos.xyz, u_Global.camera_position.xyz);
            requests.data[idx].layer = layer;
        } else {
            atomicAdd(requests.overflow_count, 1);
        }
    }
}
