use crate::renderer::{resources, Texture};
use crate::vulkan::{Allocator, BindlessManager};
use crate::Result;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

pub struct AssetManager {
    pub(crate) bindless_manager: BindlessManager,
    pub(crate) texture_registry: HashMap<u32, Arc<Texture>>,
}

impl AssetManager {
    pub fn new(bindless_manager: BindlessManager) -> Self {
        Self {
            bindless_manager,
            texture_registry: HashMap::new(),
        }
    }

    pub fn upload_ibl(
        &mut self,
        allocator: Arc<Allocator>,
        device: &crate::vulkan::VulkanDevice,
        upload_command_pool: vk::CommandPool,
        queue: vk::Queue,
        params: resources::IblUploadParams,
    ) -> Result<(u32, u32, u32)> {
        let name = "Global_IBL"; // Internal name for debug

        // 1. Upload Irradiance (Cubemap, 1 mip)
        let irradiance_texture = unsafe {
            resources::Texture::create_cubemap_from_data(
                allocator.clone(),
                device.device.clone(),
                upload_command_pool,
                queue,
                params.irradiance,
                params.irradiance_size,
                1,
                params.format,
                Some(&(name.to_owned() + "_Irradiance")),
            )?
        };
        let irradiance_idx = self
            .bindless_manager
            .add_cubemap(irradiance_texture.view(), irradiance_texture.sampler())?;

        // 2. Upload Prefilter (Cubemap, N mips)
        let prefilter_texture = unsafe {
            resources::Texture::create_cubemap_from_data(
                allocator.clone(),
                device.device.clone(),
                upload_command_pool,
                queue,
                params.prefilter,
                params.prefilter_size,
                params.prefilter_mips,
                params.format,
                Some(&(name.to_owned() + "_Prefilter")),
            )?
        };
        let prefilter_idx = self
            .bindless_manager
            .add_cubemap(prefilter_texture.view(), prefilter_texture.sampler())?;

        // 3. Upload BRDF LUT (2D Texture)
        let brdf_data = resources::TextureData {
            width: 512, // Standard IBL LUT size
            height: 512,
            pixels: params.brdf.to_vec(),
        };
        let brdf_texture = unsafe {
            resources::Texture::from_data(
                allocator.clone(),
                device.device.clone(),
                upload_command_pool,
                queue,
                &brdf_data,
                vk::Format::R16G16_SFLOAT, // Standard for BRDF LUTs
                Some(&(name.to_owned() + "_BRDF_LUT")),
            )?
        };
        let brdf_idx = self
            .bindless_manager
            .add_sampled_image(brdf_texture.view(), brdf_texture.sampler())?;

        // 4. Register with Registry to keep alive
        self.texture_registry
            .insert(irradiance_idx, Arc::new(irradiance_texture));
        self.texture_registry
            .insert(prefilter_idx, Arc::new(prefilter_texture));
        self.texture_registry
            .insert(brdf_idx, Arc::new(brdf_texture));

        log::info!(
            "IBL maps uploaded to bindless slots (Irradiance: {}, Prefilter: {}, BRDF: {})",
            irradiance_idx,
            prefilter_idx,
            brdf_idx
        );

        Ok((irradiance_idx, prefilter_idx, brdf_idx))
    }
}
