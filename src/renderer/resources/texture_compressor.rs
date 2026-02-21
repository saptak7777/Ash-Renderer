use crate::Result;
use crate::renderer::resources::texture::TextureData;
use ash::vk;
use intel_tex_2::{RgSurface, RgbaSurface, bc5, bc7};

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
            log::warn!(
                "Texture dimensions ({width}x{height}) not multiple of 4. BC7 compression might have artifacts or fail."
            );
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

    /// Generates a full mip chain and compresses each level.
    ///
    /// Returns a list of compressed data buffers, one for each mip level.
    /// Level 0 is the base image.
    pub fn compress_with_mips(
        data: &TextureData,
        format: CompressionFormat,
    ) -> Result<Vec<Vec<u8>>> {
        // 1. Generate Mip Chain (Uncompressed)
        let mips = Self::generate_mipmaps(data);

        // 2. Compress each level
        let mut compressed_mips = Vec::with_capacity(mips.len());

        for mip in mips {
            let compressed = match format {
                CompressionFormat::Bc7 => Self::compress_bc7(&mip)?,
                CompressionFormat::Bc5 => Self::compress_bc5(&mip)?,
                CompressionFormat::None => mip.pixels, // Fallback (should normally be handled by GPU blit)
            };
            compressed_mips.push(compressed);
        }

        Ok(compressed_mips)
    }

    /// Generate mipmaps using high-quality CPU downscaling (Lanczos3).
    fn generate_mipmaps(data: &TextureData) -> Vec<TextureData> {
        let mut mips = Vec::new();
        mips.push(data.clone());

        let mut width = data.width;
        let mut height = data.height;

        // Wrap raw pixels in image buffer for resizing
        // SAFETY: TextureData guarantees pixels match width * height * 4
        let mut dynamic_image = image::DynamicImage::ImageRgba8(
            image::RgbaImage::from_raw(width, height, data.pixels.clone())
                .expect("Failed to create image buffer from verified TextureData"),
        );

        while width > 1 || height > 1 {
            let new_width = if width > 1 { width / 2 } else { 1 };
            let new_height = if height > 1 { height / 2 } else { 1 };

            dynamic_image =
                dynamic_image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

            width = new_width;
            height = new_height;

            mips.push(TextureData {
                width,
                height,
                pixels: dynamic_image.to_rgba8().into_raw(),
            });
        }

        mips
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mipmap_generation_chain() {
        // Create 64x64 dummy texture
        let width = 64;
        let height = 64;
        let pixels = vec![255u8; (width * height * 4) as usize];
        let data = TextureData::new(width, height, pixels).unwrap();

        // Compress with None to check raw mip sizes
        let mips = TextureCompressor::compress_with_mips(&data, CompressionFormat::None).unwrap();

        // Expected levels: 64, 32, 16, 8, 4, 2, 1 (7 levels)
        assert_eq!(mips.len(), 7, "Should generate 7 mip levels for 64x64");

        let expected_sizes = [
            64 * 64 * 4,
            32 * 32 * 4,
            16 * 16 * 4,
            8 * 8 * 4,
            4 * 4 * 4,
            2 * 2 * 4,
            4,
        ];

        for (i, mip) in mips.iter().enumerate() {
            assert_eq!(
                mip.len(),
                expected_sizes[i],
                "Mip level {i} has incorrect size"
            );
        }
    }

    #[test]
    fn test_non_square_mips() {
        // 32x4 texture
        let width = 32;
        let height = 4;
        let pixels = vec![255; (width * height * 4) as usize];
        let data = TextureData::new(width, height, pixels).unwrap();

        let mips = TextureCompressor::compress_with_mips(&data, CompressionFormat::None).unwrap();

        // Levels:
        // 0: 32x4
        // 1: 16x2
        // 2: 8x1
        // 3: 4x1
        // 4: 2x1
        // 5: 1x1
        // Total 6 levels
        assert_eq!(mips.len(), 6);

        // Check level 2 (8x1)
        assert_eq!(mips[2].len(), 8 * 4);
    }
}
