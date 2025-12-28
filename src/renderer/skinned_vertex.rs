use ash::vk;

/// Vertex with skeletal animation data (bone indices + weights)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joint_indices: [u16; 4],
    pub joint_weights: [f32; 4],
}

impl SkinnedVertex {
    /// Vulkan vertex binding description
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<SkinnedVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// Vulkan vertex attribute descriptions
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 5] {
        [
            // Position
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            // Normal
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            // UV
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
            // Joint Indices (u16x4)
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R16G16B16A16_UINT,
                offset: 32,
            },
            // Joint Weights
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 40,
            },
        ]
    }
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

    #[test]
    fn binding_description_valid() {
        let desc = SkinnedVertex::binding_description();
        assert_eq!(desc.binding, 0);
        assert_eq!(desc.stride, 56);
        assert_eq!(desc.input_rate, vk::VertexInputRate::VERTEX);
    }

    #[test]
    fn attribute_descriptions_count() {
        let attrs = SkinnedVertex::attribute_descriptions();
        assert_eq!(attrs.len(), 5);
    }

    #[test]
    fn attribute_offsets_correct() {
        let attrs = SkinnedVertex::attribute_descriptions();
        assert_eq!(attrs[0].offset, 0); // position
        assert_eq!(attrs[1].offset, 12); // normal
        assert_eq!(attrs[2].offset, 24); // uv
        assert_eq!(attrs[3].offset, 32); // joint_indices
        assert_eq!(attrs[4].offset, 40); // joint_weights
    }
}
