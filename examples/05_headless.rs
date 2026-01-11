//! Headless rendering example.
//!
//! Demonstrates how to use the renderer without a window,
//! rendering a single frame and saving it to an image file.

use ash_renderer::prelude::*;
use glam::{Mat4, Vec3};
use std::path::Path;

fn main() -> Result<()> {
    env_logger::init();

    log::info!("Starting headless rendering example...");

    // 1. Initialize Headless Surface Provider
    let width = 1280;
    let height = 720;
    let surface_provider = ash_renderer::vulkan::HeadlessSurfaceProvider::new(width, height);

    // 2. Create Renderer
    let mut renderer = Renderer::new(&surface_provider)?;
    renderer.enable_post_processing()?;
    log::info!("Renderer initialized in headless mode with HDR post-processing.");

    // 3. Set up scene (Cube)
    let cube = Mesh::create_cube();
    let mesh_handle = renderer.upload_mesh(cube)?;

    let material = Material {
        color: [0.2, 0.8, 0.2, 1.0], // Green cube
        metallic: 0.1,
        roughness: 0.9,
        ..Default::default()
    };

    // CRITICAL FIX: Register and upload material
    let material_handle = renderer.register_and_upload_material(material)?;
    log::info!(
        "✓ Registered and uploaded green material with handle {:?}",
        material_handle
    );

    // 4. Set up Camera
    let camera_pos = Vec3::new(3.0, 3.0, 3.0);
    let target = Vec3::ZERO;
    let up = Vec3::Y;
    let aspect = width as f32 / height as f32;

    let view = Mat4::look_at_rh(camera_pos, target, up);
    let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
    proj.y_axis.y *= -1.0; // Vulkan Y-flip

    // 5. Render a single frame
    log::info!("Rendering frame...");
    renderer.submit_render_commands(&[ash_renderer::renderer::RenderCommand {
        mesh_handle,
        material_handle,
        transform: Mat4::IDENTITY,
        ..Default::default()
    }])?;

    renderer.render_frame(view, proj, camera_pos, None)?;

    // 6. Read back the image data
    log::info!("Reading back image data...");
    let image_data = renderer.read_headless_image()?;
    log::info!("Image readback complete. Size: {} bytes", image_data.len());

    // 7. Save to file using the `image` crate
    let output_path = "headless_output.png";
    log::info!("Saving to file: {}", output_path);
    image::save_buffer(
        Path::new(output_path),
        &image_data,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .map_err(|e| AshError::VulkanError(format!("Failed to save image: {e}")))?;

    log::info!("Successfully saved rendered frame to {}", output_path);

    Ok(())
}
