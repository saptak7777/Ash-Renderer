//! PBR Material Showcase Example
//!
//! Demonstrates physically-based rendering with a grid of materials.
//! Shows how metallic and roughness values affect appearance.
//!
//! Grid Layout:
//! - Rows (Y-axis): Metallic values from 0.0 (dielectric) to 1.0 (metal)
//! - Columns (X-axis): Roughness values from 0.05 (mirror) to 1.0 (matte)

use ash_renderer::prelude::*;
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
            .with_title("ASH Renderer - PBR Material Showcase")
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

                // Set mesh FIRST so mesh_data exists
                let mesh_handle = renderer.upload_mesh(cube).unwrap_or_else(|e| {
                    log::error!("Failed to upload mesh: {e}");
                    0 // Fallback
                });
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
                    renderer.set_tonemapping_enabled(true);
                }

                // Load environment map for IBL (Image-Based Lighting)
                // This provides realistic ambient lighting and reflections
                // Using pre-baked .ibl asset for fast loading
                // TEMPORARILY DISABLED FOR DEBUGGING
                /*
                if let Err(e) = renderer.load_environment_map("assets/textures/skybox.ibl") {
                    log::warn!("Failed to load environment map: {e}");
                    log::warn!("Continuing without IBL - cubes will have minimal ambient lighting");
                }
                */

                // Create a single red metallic material
                let material = Material {
                    name: "MetallicRed".to_string(),
                    color: [1.0, 0.05, 0.05, 1.0], // Deep red
                    metallic: 1.0,
                    roughness: 0.1,
                    ..Default::default()
                };

                let material_handle = renderer.register_and_upload_material(material).unwrap();
                // let mesh_handle = renderer.get_mesh_handle("PbrCube").unwrap_or(0); // We already have mesh_handle from upload

                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: Mat4::IDENTITY,
                        cast_shadows: true,
                        receive_shadows: true,
                        ..Default::default()
                    });

                log::info!("✓ Created single red metallic cube");

                // Set up lighting to highlight PBR properties
                renderer.set_lighting(
                    Vec3::new(1.0, -1.0, -1.0).normalize(),
                    [1.0, 1.0, 1.0, 1.0], // Standard white light
                    0.05,                 // Lower ambient to make specular pop
                );

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
                log::info!("PBR Material Showcase initialized!");
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
                    // Static camera position (player's view)
                    let camera_pos = Vec3::new(0.0, 3.0, 8.0);
                    let elapsed = self.start_time.elapsed().as_secs_f32();

                    renderer.set_view(camera_pos, Vec3::ZERO, Vec3::Y);

                    // Static light positions (one on each side)
                    let p1 = Vec3::new(5.0, 2.0, 0.0); // Right side
                    let p2 = Vec3::new(0.0, 2.0, 5.0); // Front side
                    let p3 = Vec3::new(-5.0, 2.0, 0.0); // Left side
                    let p4 = Vec3::new(0.0, 2.0, -5.0); // Back side

                    renderer.update_light(
                        0,
                        PointLight {
                            position: p1,
                            color: Vec3::new(1.0, 0.9, 0.8),
                            intensity: 30.0,
                            radius: 25.0,
                        },
                    );
                    renderer.update_light(
                        1,
                        PointLight {
                            position: p2,
                            color: Vec3::new(0.8, 0.9, 1.0),
                            intensity: 20.0,
                            radius: 20.0,
                        },
                    );
                    renderer.update_light(
                        2,
                        PointLight {
                            position: p3,
                            color: Vec3::new(1.0, 1.0, 1.0),
                            intensity: 25.0,
                            radius: 22.0,
                        },
                    );
                    renderer.update_light(
                        3,
                        PointLight {
                            position: p4,
                            color: Vec3::new(1.0, 0.5, 0.5),
                            intensity: 15.0,
                            radius: 15.0,
                        },
                    );
                    // Calculate camera matrices
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;
                    let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                    let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.5, 100.0);
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    // Rotate the cube
                    let rotation = glam::Quat::from_rotation_y(elapsed * 0.5);
                    let transform = Mat4::from_quat(rotation);

                    // Update render command with rotation
                    self.render_commands[0].transform = transform;

                    // Submit draw calls (Required for Bindless-Only Renderer)
                    if let Err(e) = renderer.submit_render_commands(&self.render_commands) {
                        log::error!("Failed to submit render commands: {e}");
                    }

                    if let Err(e) = renderer.render_frame(view, proj, camera_pos, Some(transform)) {
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

fn main() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");

    Ok(())
}
