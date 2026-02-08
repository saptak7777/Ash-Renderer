//! GLTF Model Loading example.
//!
//! Demonstrates loading a GLTF/GLB model using the renderer's gltf_loader utility.

use ash_renderer::prelude::*;
use ash_renderer::renderer::resources::gltf_loader;
use ash_renderer::renderer::Scene;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    mesh_handles: Vec<u32>,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: None,
            mesh_handles: Vec::new(),
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Model Loading")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                let mut scene = Scene::new(
                    Arc::clone(&renderer.device.device),
                    Arc::clone(&renderer.alloc),
                    renderer.geometry_buffer(),
                );

                // Path to a GLB model
                let glb_path = "assets/models/test.glb";

                if std::path::Path::new(glb_path).exists() {
                    log::info!("Loading model from: {glb_path}");
                    match gltf_loader::load_model(glb_path) {
                        Ok(meshes) => {
                            for (i, mut mesh) in meshes.into_iter().enumerate() {
                                let handle = (i + 1) as u32;
                                if renderer
                                    .register_mesh_handle_single(&mut scene, handle, &mut mesh)
                                    .is_ok()
                                {
                                    self.mesh_handles.push(handle);
                                    log::info!(
                                        "Registered mesh {} with handle {}",
                                        mesh.name,
                                        handle
                                    );
                                }
                            }
                        }
                        Err(e) => log::error!("Failed to load model: {e}"),
                    }
                } else {
                    log::warn!("Model not found at {glb_path}. Using default cube.");
                    let mut cube = Mesh::create_cube();
                    if renderer
                        .register_mesh_handle_single(&mut scene, 1, &mut cube)
                        .is_ok()
                    {
                        self.mesh_handles.push(1);
                    }
                }

                self.renderer = Some(renderer);
                self.scene = Some(scene);
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
                if let (Some(renderer), Some(window), Some(scene)) =
                    (&mut self.renderer, &self.window, &mut self.scene)
                {
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;
                    let elapsed = self.start_time.elapsed().as_secs_f32();

                    let camera_pos = Vec3::new(3.0 * elapsed.sin(), 2.0, 3.0 * elapsed.cos());
                    let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                    let mut proj = Mat4::perspective_infinite_reverse_rh(
                        45.0_f32.to_radians(),
                        aspect,
                        0.1, // Near Plane
                    );
                    proj.y_axis.y *= -1.0;

                    let mut commands = Vec::new();
                    for &handle in &self.mesh_handles {
                        commands.push(ash_renderer::renderer::RenderCommand {
                            mesh_handle: handle,
                            transform: Mat4::IDENTITY,
                            ..Default::default()
                        });
                    }

                    let _ = renderer.submit_render_commands(scene, &commands);
                    let _ = renderer.render_frame(scene, view, proj, camera_pos, None);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
