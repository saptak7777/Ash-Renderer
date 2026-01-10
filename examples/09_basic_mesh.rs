//! Minimal basic mesh rendering test.
//!
//! Simplest possible example: non-textured cube with basic lighting.
//! No Forward+, no post-processing, no complexity.

use ash_renderer::prelude::*;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
        }
    }
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

                // Simple grey matte material
                let material = Material {
                    color: [0.5, 0.5, 0.5, 1.0], // Grey
                    metallic: 0.0,               // Non-metallic
                    roughness: 0.8,              // Matte
                    ..Default::default()
                };

                if let Err(e) = renderer.set_mesh(cube) {
                    log::error!("Failed to set mesh: {e}");
                    event_loop.exit();
                    return;
                }

                *renderer.material_mut() = material.clone();

                // Register and upload material
                let material_handle = renderer
                    .material_manager_mut()
                    .register_material(material.clone());
                if let Err(e) =
                    renderer.upload_material_to_gpu(material_handle.index as u32, &material)
                {
                    log::error!("Failed to upload material: {e}");
                }

                if let Some(mesh_data) = renderer.get_mesh_data_mut(0) {
                    mesh_data.material_handle = material_handle;
                }

                // Simple directional lighting (NO Forward+)
                renderer.set_lighting(
                    Vec3::new(1.0, -1.0, -1.0).normalize(),
                    [3.0, 3.0, 3.0, 1.0], // Bright white
                    0.3,                  // Strong ambient
                );

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

                        if let Err(e) = renderer.render_frame(view, proj, camera_pos) {
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
