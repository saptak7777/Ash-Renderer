use ash_renderer::renderer::resources::texture::TextureData;
use ash_renderer::renderer::resources::texture_compressor::{CompressionFormat, TextureCompressor};
use bytemuck::{Pod, Zeroable};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Header for the legacy .ash_tex format, now maintained locally in the cooker.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct AshTexHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub mip_levels: u32,
    pub compression: u32,
    pub _padding: [u32; 3],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: asset_cooker <input_image_path> [output_path]");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = if args.len() >= 3 {
        std::path::PathBuf::from(&args[2])
    } else {
        input_path.with_extension("ash_tex")
    };

    println!("Cooking asset: {input_path:?}");
    let start_time = Instant::now();

    // 1. Load Image
    let img = image::open(input_path).map_err(|e| format!("Failed to open image: {e}"))?;
    let img = img.to_rgba8();
    let width = img.width();
    let height = img.height();

    println!("  Dimensions: {width}x{height}");

    // 2. Determine Format
    // Simple heuristic: if filename contains "normal", use BC5. Otherwise BC7.
    let is_normal_map = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase().contains("normal"))
        .unwrap_or(false);

    let format = if is_normal_map {
        println!("  Detected Normal Map -> Using BC5");
        CompressionFormat::Bc5
    } else {
        println!("  Standard Texture -> Using BC7");
        CompressionFormat::Bc7
    };

    let texture_data = TextureData {
        width,
        height,
        pixels: img.into_raw(),
    };

    // 3. Generate Mips & Compress
    println!("  Generating mips and compressing...");
    let compressed_mips = TextureCompressor::compress_with_mips(&texture_data, format)?;
    let mip_levels = compressed_mips.len() as u32;

    println!("  Generated {mip_levels} mip levels");

    // 4. Write Output
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    // Write Header
    let vk_format = format.to_vk_format(false); // Assume UNORM for now
    let compression_type = match format {
        CompressionFormat::None => 0,
        CompressionFormat::Bc7 => 1,
        CompressionFormat::Bc5 => 2,
    };

    let header = AshTexHeader {
        magic: *b"ASHT",
        version: 1,
        width,
        height,
        format: vk_format.as_raw() as u32,
        mip_levels,
        compression: compression_type,
        _padding: [0; 3],
    };

    let header_bytes = bytemuck::bytes_of(&header);
    writer.write_all(header_bytes)?;

    // Write Mip Data
    for (i, mip) in compressed_mips.iter().enumerate() {
        println!("  Writing Mip {}: {} bytes", i, mip.len());
        writer.write_all(mip)?;
    }

    writer.flush()?;

    let duration = start_time.elapsed();
    println!("Successfully cooked to {output_path:?} in {duration:.2?}");

    Ok(())
}
