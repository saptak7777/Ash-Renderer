//! PBR Material Grid Example
//!
//! Demonstrates physically-based rendering with a 5x5 grid of materials.
//! Shows how metallic and roughness values affect appearance under identical lighting.
//!
//! Grid Layout:
//! - Rows (Y-axis): Metallic values from 0.0 (dielectric) to 1.0 (metal)
//! - Columns (X-axis): Roughness values from 0.05 (mirror) to 1.0 (matte)
//!
//! This example is perfect for:
//! - Understanding PBR material properties
//! - Debugging material rendering
//! - Testing lighting and reflection systems

use ash_renderer::prelude::*;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::features::PointLight;
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    tint_buffer: Option<Arc<parking_lot::Mutex<StorageBuffer<Vec4>>>>,
    renderer: Option<Renderer>,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            tint_buffer: None,
            renderer: None,
            render_commands: Vec::new(),
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - PBR Material Grid")
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // Create a cube mesh
                let mut cube = Mesh::create_cube();
                // Override vertex colors to WHITE so they don't affect the material color
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }
                cube.name = Arc::from("PbrCube");
                cube.texture_data = None;

                // Upload mesh
                let mesh_handle = renderer.upload_mesh_single(cube).unwrap_or(0);
                log::info!("✓ Mesh uploaded to GPU");

                // Register bindless storage buffer (prevents crash)
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                if let Ok((tint_buffer, _tint_index)) =
                    renderer.register_bindless_storage_buffer(&tint_colors, "DefaultTint")
                {
                    self.tint_buffer = Some(tint_buffer);
                    log::info!("✓ Registered default tint buffer");
                }

                // Enable post-processing for HDR/Tonemapping
                if let Err(e) = renderer.enable_post_processing() {
                    log::warn!("Post-processing failed: {e}");
                    renderer.tonemapping_enabled = true;
                }

                // Load pre-baked IBL environment map for realistic PBR lighting
                if let Ok(asset) =
                    archetype_asset::ibl::MappedIblAsset::load("assets/textures/skybox.ibl")
                {
                    log::info!("✓ IBL asset loaded via Zero-Copy MappedIblAsset");

                    let params = ash_renderer::renderer::resources::IblUploadParams {
                        irradiance: asset.irradiance_data(),
                        prefilter: asset.prefiltered_data(),
                        brdf: asset.brdf_data(),
                        irradiance_size: asset.header.irradiance_size,
                        prefilter_size: asset.header.prefiltered_size,
                        prefilter_mips: asset.header.prefiltered_mips,
                        format: ash::vk::Format::from_raw(asset.header.format as i32),
                    };

                    if let Err(e) = renderer.upload_ibl(params) {
                        log::warn!("Failed to upload IBL: {e}");
                    }
                } else {
                    log::warn!("Failed to load IBL asset or file not found");
                    log::warn!("Continuing without IBL - cubes will have minimal ambient lighting");
                }

                // Create 5x5 PBR material grid
                // Rows: Metallic 0.0 to 1.0 (dielectric to metal)
                // Columns: Roughness 0.05 to 1.0 (mirror to matte)
                let grid_size = 5;
                let spacing = 2.5;
                // let mesh_handle = renderer.get_mesh_handle("PbrCube").unwrap_or(0); // Already have handle

                for row in 0..grid_size {
                    for col in 0..grid_size {
                        // Calculate metallic and roughness values
                        let metallic = row as f32 / (grid_size - 1) as f32;
                        let roughness = 0.05 + (col as f32 / (grid_size - 1) as f32) * 0.95;

                        // Create material with varying color for visual distinction
                        let hue = (row as f32 * 0.2 + col as f32 * 0.1) % 1.0;
                        let color = hsl_to_rgb(hue, 0.7, 0.5);

                        let material = Material {
                            name: format!("PBR_M{metallic:.2}_R{roughness:.2}"),
                            color: [color.x, color.y, color.z, 1.0],
                            metallic,
                            roughness,
                            ..Default::default()
                        };

                        let material_handle =
                            renderer.register_and_upload_material(material).unwrap();

                        // Position cube in grid
                        let x = (col as f32 - (grid_size - 1) as f32 / 2.0) * spacing;
                        let y = (row as f32 - (grid_size - 1) as f32 / 2.0) * spacing;
                        let transform = Mat4::from_translation(Vec3::new(x, y, 0.0));

                        self.render_commands
                            .push(ash_renderer::renderer::RenderCommand {
                                mesh_handle,
                                material_handle,
                                transform,
                                cast_shadows: true,
                                receive_shadows: true,
                                ..Default::default()
                            });
                    }
                }

                log::info!(
                    "✓ Created {}x{} PBR material grid ({} cubes)",
                    grid_size,
                    grid_size,
                    grid_size * grid_size
                );

                // Set up lighting to highlight PBR properties (RAGE approach)
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::IndoorLit)
                    .with_directional(
                        Vec3::new(1.0, -1.0, -1.0).normalize(),
                        Vec3::splat(2.0),
                        1.0,
                    )
                    .build();

                renderer.set_lighting(&lighting);

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
                log::info!("PBR Material Grid initialized!");
            }
            Err(e) => {
                log::error!("Failed to create renderer: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, exiting...");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.request_swapchain_resize(ash::vk::Extent2D {
                        width: new_size.width,
                        height: new_size.height,
                    });
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(window)) =
                    (self.renderer.as_mut(), self.window.as_ref())
                {
                    // Update camera position to view the entire grid
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let camera_radius = 15.0;
                    let camera_x = camera_radius * (elapsed * 0.15).cos();
                    let camera_z = camera_radius * (elapsed * 0.15).sin();
                    let camera_pos = Vec3::new(camera_x, 8.0, camera_z);

                    renderer.set_view(camera_pos, Vec3::ZERO, Vec3::Y);

                    // Update light positions (focused around the center grid)
                    let p1 = Vec3::new(
                        (elapsed * 0.5).cos() * 6.0,
                        5.0,
                        (elapsed * 0.5).sin() * 6.0,
                    );
                    let p2 = Vec3::new(
                        (elapsed * 0.6).sin() * 6.0,
                        -3.0,
                        (elapsed * 0.6).cos() * 6.0,
                    );
                    let p3 = Vec3::new(
                        -(elapsed * 0.4).cos() * 6.0,
                        2.0,
                        -(elapsed * 0.4).sin() * 6.0,
                    );
                    let p4 = Vec3::new(
                        -(elapsed * 0.7).sin() * 6.0,
                        -1.0,
                        -(elapsed * 0.7).cos() * 6.0,
                    );

                    renderer.update_light(
                        0,
                        PointLight {
                            position: p1,
                            color: Vec3::new(1.0, 0.9, 0.8),
                            intensity: 200.0,
                            radius: 30.0,
                        },
                    );
                    renderer.update_light(
                        1,
                        PointLight {
                            position: p2,
                            color: Vec3::new(0.8, 0.9, 1.0),
                            intensity: 150.0,
                            radius: 25.0,
                        },
                    );
                    renderer.update_light(
                        2,
                        PointLight {
                            position: p3,
                            color: Vec3::new(1.0, 1.0, 1.0),
                            intensity: 180.0,
                            radius: 28.0,
                        },
                    );
                    renderer.update_light(
                        3,
                        PointLight {
                            position: p4,
                            color: Vec3::new(1.0, 0.5, 0.5),
                            intensity: 120.0,
                            radius: 20.0,
                        },
                    );

                    // Calculate camera matrices
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;
                    let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                    // MODERN PROJECTION: Infinite Reverse Z
                    let mut proj = Mat4::perspective_infinite_reverse_rh(
                        45.0_f32.to_radians(),
                        aspect,
                        0.5, // Near Plane
                    );
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    // Rotate the whole scene slowly
                    // renderer.transform.rotation = glam::Quat::from_rotation_y(elapsed * 0.03);

                    // Submit draw calls (Required for Bindless-Only Renderer)
                    if let Err(e) = renderer.submit_render_commands(&self.render_commands) {
                        log::error!("Failed to submit render commands: {e}");
                    }

                    if let Err(e) = renderer.render_frame(view, proj, camera_pos, None) {
                        log::error!("Failed to render frame: {e}");
                    }
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => (),
        }
    }
}

/// Convert HSL color to RGB
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Vec3::new(r + m, g + m, b + m)
}

fn main() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");

    Ok(())
}
