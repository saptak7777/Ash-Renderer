use crate::renderer::resources::texture::TextureData;
use crate::Result;
use ash::vk;
use intel_tex_2::{bc5, bc7, RgSurface, RgbaSurface};

/// Supported block compression formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    /// No compression (RGBA8)
    None,
    /// BC7 compression (High quality, for color/albedo)
    Bc7,
    /// BC5 compression (Two-channel, for normal maps)
    Bc5,
}

impl CompressionFormat {
    /// Get the corresponding Vulkan format.
    pub fn to_vk_format(self, srgb: bool) -> vk::Format {
        match self {
            Self::None => {
                if srgb {
                    vk::Format::R8G8B8A8_SRGB
                } else {
                    vk::Format::R8G8B8A8_UNORM
                }
            }
            Self::Bc7 => {
                if srgb {
                    vk::Format::BC7_SRGB_BLOCK
                } else {
                    vk::Format::BC7_UNORM_BLOCK
                }
            }
            Self::Bc5 => {
                // BC5 is typically UNORM
                vk::Format::BC5_UNORM_BLOCK
            }
        }
    }

    /// Get the expected compression ratio (uncompressed size / compressed size)
    pub fn compression_ratio(self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Bc7 => 4.0, // RGBA8 (4 bytes) -> BC7 (1 byte per pixel)
            Self::Bc5 => 4.0, // RGBA8 (4 bytes) -> BC5 (1 byte per pixel)
        }
    }
}

/// Helper for compressing texture data on the CPU.
pub struct TextureCompressor;

impl TextureCompressor {
    /// Compress RGBA8 data using BC7.
    pub fn compress_bc7(data: &TextureData) -> Result<Vec<u8>> {
        let width = data.width;
        let height = data.height;

        // BC7 works on 4x4 blocks
        if width % 4 != 0 || height % 4 != 0 {
            log::warn!("Texture dimensions ({width}x{height}) not multiple of 4. BC7 compression might have artifacts or fail.");
        }

        let surface = RgbaSurface {
            data: &data.pixels,
            width,
            height,
            stride: width * 4,
        };

        // BC7 compression with ultra fast settings
        let compressed_data = bc7::compress_blocks(&bc7::opaque_ultra_fast_settings(), &surface);

        Ok(compressed_data)
    }

    /// Compress RGBA8 data using BC5 (for normals).
    /// Extracts R and G channels to create an RG surface for BC5.
    pub fn compress_bc5(data: &TextureData) -> Result<Vec<u8>> {
        let width = data.width;
        let height = data.height;

        let mut rg_pixels = Vec::with_capacity((width * height * 2) as usize);
        for chunk in data.pixels.chunks_exact(4) {
            rg_pixels.push(chunk[0]); // R
            rg_pixels.push(chunk[1]); // G
        }

        let surface = RgSurface {
            data: &rg_pixels,
            width,
            height,
            stride: width * 2,
        };

        let compressed_data = bc5::compress_blocks(&surface);

        Ok(compressed_data)
    }
}
