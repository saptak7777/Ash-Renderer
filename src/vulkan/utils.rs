use ash::vk;

/// Returns true if the provided format contains a stencil component.
pub fn has_stencil_component(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
    )
}
