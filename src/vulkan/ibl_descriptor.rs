use super::descriptor_set::DescriptorSet;
use crate::Result;
use ash::vk;

/// Resource handles for IBL textures
#[derive(Clone, Copy, Debug)]
pub struct IBLResources {
    pub irradiance_view: vk::ImageView,
    pub prefiltered_view: vk::ImageView,
    pub brdf_lut_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

/// Helper for binding IBL resources to Set 3
pub struct IBLDescriptorSet;

impl IBLDescriptorSet {
    /// Update the descriptor set with IBL textures.
    /// Shadow map is binding 0, IBL textures are 1, 2, 3.
    pub fn update(descriptor_set: &DescriptorSet, resources: &IBLResources) -> Result<()> {
        let irradiance_info = vk::DescriptorImageInfo {
            sampler: resources.sampler,
            image_view: resources.irradiance_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let prefiltered_info = vk::DescriptorImageInfo {
            sampler: resources.sampler,
            image_view: resources.prefiltered_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let brdf_lut_info = vk::DescriptorImageInfo {
            sampler: resources.sampler,
            image_view: resources.brdf_lut_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        descriptor_set.update_image_at(
            1,
            0,
            irradiance_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;
        descriptor_set.update_image_at(
            2,
            0,
            prefiltered_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;
        descriptor_set.update_image_at(
            3,
            0,
            brdf_lut_info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        Ok(())
    }
}
