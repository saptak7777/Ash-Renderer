//! GLTF model loading example.
//!
//! Demonstrates loading and rendering GLTF models with PBR materials.
//! Shows how to control the camera from the application.

use ash_renderer::{prelude::*, WindowSurfaceProvider};
use glam::{Mat4, Vec3};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - GLTF Model")
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

        let window = event_loop.create_window(window_attrs).unwrap();
        let size = window.inner_size();
        let surface_provider = WindowSurfaceProvider::new(&window, size.width, size.height);

        match Renderer::new(&surface_provider) {
            Ok(renderer) => {
                // Load GLTF model
                // TODO: Implement GLTF loading via renderer.load_gltf("path/to/model.gltf")
                log::info!("GLTF loading example - model loading not yet implemented");

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
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
                    // Application-side camera control (replaces auto_rotate)
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Orbiting camera around the origin
                    let radius = 5.0;
                    let camera_x = radius * elapsed.sin();
                    let camera_z = radius * elapsed.cos();
                    let camera_pos = Vec3::new(camera_x, 2.0, camera_z);
                    let target = Vec3::ZERO;
                    let up = Vec3::Y;

                    let view = Mat4::look_at_rh(camera_pos, target, up);
                    let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.5, 100.0);
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    if let Err(e) = renderer.render_frame(view, proj, camera_pos) {
                        log::error!("Render error: {e}");
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.request_swapchain_resize(ash::vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    });
                }
            }
            _ => {}
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
