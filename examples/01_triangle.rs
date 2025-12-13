//! Basic triangle example.
//!
//! Demonstrates minimal renderer setup and rendering a simple triangle.

use ash_renderer::{prelude::*, WindowSurfaceProvider};
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
            .with_title("ASH Renderer - Triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = event_loop.create_window(window_attrs).unwrap();
        let size = window.inner_size();
        let surface_provider = WindowSurfaceProvider::new(&window, size.width, size.height);

        match Renderer::new(&surface_provider) {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.window = Some(window);
                log::info!("Renderer initialized successfully!");
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
                    let aspect = size.width as f32 / size.height.max(1) as f32;

                    // Simple static camera
                    let camera_pos = glam::Vec3::new(0.0, 0.0, 3.0);
                    let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                    let mut proj =
                        glam::Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.5, 100.0);
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
