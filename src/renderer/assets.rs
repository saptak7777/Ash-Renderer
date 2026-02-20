use crate::renderer::vram_budget::VramBudget;
use crate::renderer::{resources, Texture, TextureInitContext};
use crate::vulkan::{Allocator, BindlessManager};
use crate::Result;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

pub type BindlessDescriptorSet = BindlessManager;
pub type CpuMesh = resources::Mesh;
pub type StandardMemoryAllocator = Allocator;

pub struct AssetManager {
    pub bindless_manager: BindlessManager,
    pub texture_registry: HashMap<u32, Arc<Texture>>,
    pub vram_budget: VramBudget,
    pub texture_compression: bool,
}

impl AssetManager {
    pub fn new(bindless_manager: BindlessManager, vram_budget: VramBudget) -> Self {
        Self {
            bindless_manager,
            texture_registry: HashMap::new(),
            vram_budget,
            texture_compression: true, // Default enabled
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

        let tex_ctx = TextureInitContext {
            allocator: Arc::clone(&allocator),
            device: Arc::clone(&device.device),
            command_pool: upload_command_pool,
            queue,
        };

        // 1. Upload Irradiance (Cubemap, 1 mip)
        let irradiance_texture = unsafe {
            resources::Texture::create_cubemap_from_data(
                &tex_ctx,
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
                &tex_ctx,
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
                &tex_ctx,
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
            "IBL maps uploaded to bindless slots (Irradiance: {irradiance_idx}, Prefilter: {prefilter_idx}, BRDF: {brdf_idx})"
        );

        Ok((irradiance_idx, prefilter_idx, brdf_idx))
    }
    pub fn ingest_mesh_textures(
        &mut self,
        device: Arc<ash::Device>,
        allocator: Arc<StandardMemoryAllocator>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
        mesh: &mut CpuMesh,
    ) -> Result<()> {
        // 1. Ensure textures are uploaded to GPU using VramBudget
        unsafe {
            mesh.ensure_texture(
                allocator,
                device,
                *command_pool,
                *queue,
                &mut self.vram_budget,
                self.texture_compression,
            )?;
        }

        // 2. Register with BindlessManager and update indices in CpuMesh
        mesh.texture_index =
            Some(self.register_single_texture("base_color", mesh.texture.clone())?);
        mesh.normal_texture_index =
            Some(self.register_single_texture("normal", mesh.normal_texture.clone())?);
        mesh.metallic_roughness_texture_index = Some(self.register_single_texture(
            "metallic_roughness",
            mesh.metallic_roughness_texture.clone(),
        )?);
        mesh.occlusion_texture_index =
            Some(self.register_single_texture("occlusion", mesh.occlusion_texture.clone())?);
        mesh.emissive_texture_index =
            Some(self.register_single_texture("emissive", mesh.emissive_texture.clone())?);

        Ok(())
    }

    fn register_single_texture(
        &mut self,
        texture_name: &str,
        texture: Option<Arc<Texture>>,
    ) -> Result<u32> {
        let index = match texture {
            Some(tex) => {
                match self
                    .bindless_manager
                    .add_sampled_image(tex.view(), tex.sampler())
                {
                    Ok(idx) => {
                        log::debug!("Registered {texture_name} texture at bindless index {idx}");
                        self.texture_registry.insert(idx, tex);
                        idx
                    }
                    Err(e) => {
                        log::warn!("Failed to register {texture_name} texture: {e}");
                        u32::MAX
                    }
                }
            }
            None => {
                log::debug!("{texture_name} texture not provided");
                u32::MAX
            }
        };

        Ok(index)
    }
}
