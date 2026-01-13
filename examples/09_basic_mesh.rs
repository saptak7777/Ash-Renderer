//! Minimal basic mesh rendering test.
//!
//! Simplest possible example: non-textured cube with basic lighting.
//! No Forward+, no post-processing, no complexity.
//!

use ash_renderer::prelude::*;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Basic Mesh Test")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // Create simple cube
                let mut cube = Mesh::create_cube();

                // White vertex colors (don't tint)
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }

                cube.name = Arc::from("BasicCube");
                cube.texture_data = None;

                // Upload mesh
                let mesh_handle = renderer.upload_mesh(cube).unwrap_or(0);

                // Simple grey matte material
                let material = Material {
                    color: [0.5, 0.5, 0.5, 1.0], // Grey
                    metallic: 0.0,               // Non-metallic
                    roughness: 0.8,              // Matte
                    ..Default::default()
                };

                // Register and upload material
                let material_handle = renderer.register_and_upload_material(material).unwrap();

                // Setup render command
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: Mat4::IDENTITY,
                        ..Default::default()
                    });

                // Simple directional lighting (RAGE approach)
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::IndoorLit)
                    .with_directional(
                        Vec3::new(1.0, -1.0, -1.0).normalize(),
                        Vec3::splat(3.0),
                        1.0,
                    )
                    .build();

                renderer.set_lighting(&lighting);

                log::info!("✓ Basic mesh renderer initialized");
                self.renderer = Some(renderer);
                self.window = Some(window);
            }
            Err(e) => {
                log::error!("Failed to create renderer: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        let aspect = size.width as f32 / size.height as f32;

                        // Fixed camera
                        let camera_pos = Vec3::new(0.0, 2.0, 5.0);
                        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                        let mut proj =
                            Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
                        proj.y_axis.y *= -1.0;

                        // Submit commands
                        if let Err(e) = renderer.submit_render_commands(&self.render_commands) {
                            log::error!("Failed to submit render commands: {e}");
                        }

                        if let Err(e) = renderer.render_frame(view, proj, camera_pos, None) {
                            log::error!("Render error: {e}");
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    log::info!("Starting basic mesh test...");
    log::info!("Expected: Green cube on black background");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");

    Ok(())
}
