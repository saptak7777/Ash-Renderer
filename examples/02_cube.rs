//! Cube with textures example.
//!
//! Demonstrates textured cube rendering with materials.

use ash_renderer::prelude::*;
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
            .with_title("ASH Renderer - Textured Cube")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();

        match Renderer::new(&window) {
            Ok(mut renderer) => {
                // Create a cube mesh
                let cube = Mesh::create_cube();

                // Set up material
                let material = Material {
                    color: [0.8, 0.2, 0.2, 1.0],
                    metallic: 0.5,
                    roughness: 0.5,
                    ..Default::default()
                };

                renderer.set_mesh(cube);

                // Material is set via the mesh or can be configured on the renderer
                *renderer.material_mut() = material;

                self.renderer = Some(renderer);
                self.window = Some(window);
                log::info!("Cube renderer initialized!");
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
                if let Some(renderer) = &mut self.renderer {
                    if let Err(e) = renderer.render_frame() {
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
