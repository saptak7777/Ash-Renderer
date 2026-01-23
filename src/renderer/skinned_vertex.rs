/// Vertex with skeletal animation data (bone indices + weights)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joint_indices: [u16; 4],
    pub joint_weights: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skinned_vertex_size() {
        // position(12) + normal(12) + uv(8) + indices(8) + weights(16) = 56
        assert_eq!(std::mem::size_of::<SkinnedVertex>(), 56);
    }

    #[test]
    fn skinned_vertex_alignment() {
        assert_eq!(std::mem::align_of::<SkinnedVertex>(), 4);
    }
}
