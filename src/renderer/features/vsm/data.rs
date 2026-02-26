use glam::{Mat4, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmGlobalInfo {
    pub light_view_projections: [Mat4; 16], // Array for cascade/clipmap levels
    pub view_proj: Mat4,                    // Camera ViewProj
    pub inv_view_proj: Mat4,                // Camera Inverse ViewProj (for position reconstruction)
    pub camera_position: Vec4,
    pub light_dir: Vec4, // .xyz = dir, .w = time/padding
    pub page_table_size: u32,
    pub page_table_index: u32,
    pub physical_cache_index: u32,
    pub scene_depth_index: u32,
    pub page_table_storage_index: u32,
    pub physical_cache_storage_index: u32,
    pub request_ptr: u64,
    pub allocation_ptr: u64,
    pub _pad3: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmLightData {
    pub view_proj: Mat4,
    pub position: Vec4,
    pub direction: Vec4,
    pub color: Vec4,
    pub intensity: f32,
    pub radius: f32,
    pub _padding: [f32; 2], // Ensure 16-byte alignment if needed
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmPageData {
    pub page_address: u64,
    pub is_dirty: u32,
    pub last_visible_frame: u32,
}
