use anyhow::{Context, Result};
use ash::vk;
use ash_renderer::renderer::features::ibl_manager::IblManager;
use ash_renderer::renderer::resources::{IblAssetHeader, Texture, TextureData};
use ash_renderer::vulkan::surface_provider::HeadlessSurfaceProvider;
use ash_renderer::vulkan::{Allocator, VulkanDevice, VulkanInstance};
use clap::Parser;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(long, default_value_t = 512)]
    cubemap_size: u32,

    #[arg(long, default_value_t = 32)]
    irradiance_size: u32,

    #[arg(long, default_value_t = 128)]
    prefiltered_size: u32,

    #[arg(long, default_value_t = false)]
    debug: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("🚀 Starting IBL Baker...");
    println!("   Input: {:?}", args.input);
    println!("   Output: {:?}", args.output);

    // 1. Initialize Headless Vulkan
    let surface_provider = HeadlessSurfaceProvider::new(args.cubemap_size, args.cubemap_size);
    let instance = Arc::new(
        VulkanInstance::new(&surface_provider, args.debug).context("Failed to create instance")?,
    );
    let device = Arc::new(
        VulkanDevice::new(Arc::clone(&instance), true).context("Failed to create device")?,
    );
    let allocator =
        Arc::new(unsafe { Allocator::new(&device).context("Failed to create allocator")? });

    // 2. Setup Command Pool
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(device.graphics_queue_family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool = unsafe {
        device
            .device
            .create_command_pool(&command_pool_info, None)?
    };

    // 3. Load Input HDR
    println!("📥 Loading HDR texture...");
    let equirect_tex = load_hdr(
        &args.input,
        Arc::clone(&allocator),
        Arc::clone(&device.device),
        command_pool,
        device.graphics_queue,
    )?;

    // 4. Initialize IBL Manager
    println!("⚙️  Initializing IBL Manager...");
    let mut ibl_manager = IblManager::new(Arc::clone(&device.device), Arc::clone(&allocator))
        .context("Failed to initialize IBL Manager")?;

    // 5. Bake Everything
    println!(
        "🔥 Baking cubemap ({}x{})...",
        args.cubemap_size, args.cubemap_size
    );
    let env_cubemap = ibl_manager
        .create_cubemap_from_equirect(
            &device,
            command_pool,
            equirect_tex.view(),
            equirect_tex.sampler(),
            args.cubemap_size,
        )
        .context("Failed to bake cubemap")?;

    println!("🔥 Baking irradiance map (32x32)...");
    let irradiance_map = ibl_manager
        .generate_irradiance(
            &device,
            command_pool,
            env_cubemap.view(),
            equirect_tex.sampler(),
        )
        .context("Failed to bake irradiance map")?;

    println!(
        "🔥 Baking prefiltered map ({}x{})...",
        args.prefiltered_size, args.prefiltered_size
    );
    let prefiltered_map = ibl_manager
        .generate_prefiltered(
            &device,
            command_pool,
            env_cubemap.view(),
            equirect_tex.sampler(),
        )
        .context("Failed to bake prefiltered map")?;

    println!("🔥 Generating BRDF LUT...");
    let brdf_lut = ash_renderer::renderer::resources::ImageHandle::create_brdf_lut(
        Arc::clone(&device.device),
        Arc::clone(&allocator),
        512,
    )
    .context("Failed to create BRDF LUT image")?;

    // 6. Read back data
    println!("📤 Reading back data from GPU...");
    let cubemap_data = env_cubemap.read_to_buffer(command_pool, device.graphics_queue)?;
    let irradiance_data = irradiance_map.read_to_buffer(command_pool, device.graphics_queue)?;
    let prefiltered_data = prefiltered_map.read_to_buffer(command_pool, device.graphics_queue)?;
    let brdf_lut_data = brdf_lut.read_to_buffer(command_pool, device.graphics_queue)?;

    // 7. Save to Binary
    println!("💾 Saving to {:?}...", args.output);
    let header = IblAssetHeader {
        magic: *b"AIBL",
        version: 1,
        cubemap_size: args.cubemap_size,
        irradiance_size: args.irradiance_size,
        prefiltered_size: args.prefiltered_size,
        prefiltered_mips: prefiltered_map.mip_levels(),
        format: env_cubemap.format().as_raw() as u32,
        _padding: [0; 9],
    };

    save_ibl(
        &args.output,
        &header,
        &cubemap_data,
        &irradiance_data,
        &prefiltered_data,
        &brdf_lut_data,
    )?;

    println!("✅ IBL Baking complete!");

    // Cleanup
    unsafe {
        device.device.destroy_command_pool(command_pool, None);
    }

    Ok(())
}

fn load_hdr(
    path: &Path,
    allocator: Arc<Allocator>,
    device: Arc<ash::Device>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<Texture> {
    // Try loading first
    let rgb32f_result = image::open(path).map(|img| img.into_rgba32f());

    let rgb32f = match rgb32f_result {
        Ok(img) if img.width() > 16 && img.height() > 16 => {
            println!("✅ Loaded HDR: {}x{}", img.width(), img.height());
            img
        }
        Ok(img) => {
            log::warn!(
                "Loaded HDR but dimensions too small ({}x{}), likely garbage. Using fallback.",
                img.width(),
                img.height()
            );
            // Fallback generation (duplicated code, should function-ize but inline is fine)
            generate_fallback_hdr()
        }
        Err(e) => {
            log::warn!("Failed to load HDR {path:?}: {e}");
            println!(
                "⚠️  Failed to load HDR (might be LFS pointer). Generating synthetic fallback."
            );
            generate_fallback_hdr()
        }
    };

    let width = rgb32f.width();
    let height = rgb32f.height();
    let raw_pixels = rgb32f.into_raw();
    let raw_bytes = bytemuck::cast_slice(&raw_pixels);

    // Bypass TextureData::new validation which assumes 4 bytes/pixel
    let data = TextureData {
        width,
        height,
        pixels: raw_bytes.to_vec(),
    };

    unsafe {
        Texture::from_data(
            allocator,
            device,
            command_pool,
            queue,
            &data,
            vk::Format::R32G32B32A32_SFLOAT,
            Some("HDR Input"),
        )
        .map_err(|e| anyhow::anyhow!(e))
    }
}

fn save_ibl(
    path: &Path,
    header: &IblAssetHeader,
    cubemap: &[u8],
    irradiance: &[u8],
    prefiltered: &[u8],
    brdf: &[u8],
) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytemuck::bytes_of(header))?;

    let lengths = [
        cubemap.len() as u64,
        irradiance.len() as u64,
        prefiltered.len() as u64,
        brdf.len() as u64,
    ];
    file.write_all(bytemuck::cast_slice(&lengths))?;

    file.write_all(cubemap)?;
    file.write_all(irradiance)?;
    file.write_all(prefiltered)?;
    file.write_all(brdf)?;

    Ok(())
}

fn generate_fallback_hdr() -> image::Rgba32FImage {
    // Generate a simple gradient or flat color HDR
    let width = 64;
    let height = 32;
    let mut buffer = image::Rgba32FImage::new(width, height);
    for (_x, y, pixel) in buffer.enumerate_pixels_mut() {
        // Simple gradient based on Y to simulate sky
        let t = y as f32 / height as f32;
        let r = 0.5;
        let g = 0.5 + t * 0.5; // Blue-ish top
        let b = 1.0;
        *pixel = image::Rgba([r, g, b, 1.0]);
    }
    buffer
}
