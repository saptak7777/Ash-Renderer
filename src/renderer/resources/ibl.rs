use ash::vk;
use bytemuck::{Pod, Zeroable};

/// Matches the format in archetype_asset::ibl
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IblAssetHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub cubemap_size: u32,
    pub irradiance_size: u32,
    pub prefiltered_size: u32,
    pub prefiltered_mips: u32,
    pub format: u32, // VK_FORMAT_...
    pub _padding: [u32; 2],
}

#[derive(Clone, Debug)]
pub struct IblUploadParams<'a> {
    pub irradiance: &'a [u8],
    pub prefilter: &'a [u8],
    pub brdf: &'a [u8],
    pub irradiance_size: u32,
    pub prefilter_size: u32,
    pub prefilter_mips: u32,
    pub format: vk::Format,
}
