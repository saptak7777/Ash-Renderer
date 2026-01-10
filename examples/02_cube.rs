//! Cube with textures example.
//!
//! Demonstrates textured cube rendering with materials.
//! Shows how to control the camera from the application.

use ash_renderer::prelude::*;
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
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            tint_buffer: None,
            renderer: None,
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Textured Cube")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

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
                // RENAMING is critical because the renderer caches meshes by name!
                cube.name = Arc::from("TexturedCube");
                // Clear texture data so material color shows through
                cube.texture_data = None;

                // Set up material
                let material = Material {
                    color: [0.8, 0.2, 0.2, 1.0],
                    metallic: 0.5,
                    roughness: 0.5,
                    ..Default::default()
                };

                if let Err(e) = renderer.set_mesh(cube) {
                    log::error!("Failed to set mesh: {e}");
                    event_loop.exit();
                    return;
                }

                // CRITICAL FIX: Set renderer material AFTER set_mesh, then register and upload
                *renderer.material_mut() = material.clone();
                renderer.material_mut().tint_index = -1; // Disable tint buffer usage

                // Register material with material manager
                let material_handle = renderer
                    .material_manager_mut()
                    .register_material(material.clone());

                // Upload the material to GPU
                if let Err(e) =
                    renderer.upload_material_to_gpu(material_handle.index as u32, &material)
                {
                    log::error!("Failed to upload material to GPU: {e}");
                    event_loop.exit();
                    return;
                }

                // Update mesh_data so draw_items use the correct material
                if let Some(mesh_data) = renderer.get_mesh_data_mut(0) {
                    mesh_data.material_handle = material_handle;
                }

                log::info!(
                    "✓ Uploaded red material to GPU with handle {:?}",
                    material_handle
                );

                // CRITICAL: Set lighting for visibility (default ambient is too dark)
                renderer.set_lighting(
                    Vec3::new(1.0, -1.0, -1.0).normalize(),
                    [2.0, 2.0, 2.0, 1.0], // Bright white light
                    0.2,                  // Ambient strength for better visibility
                );

                // Register bindless storage buffer (prevents crash)
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                if let Ok((tint_buffer, _tint_index)) =
                    renderer.register_bindless_storage_buffer(&tint_colors, "DefaultTint")
                {
                    self.tint_buffer = Some(tint_buffer);
                    log::info!("✓ Registered default tint buffer");
                }

                // CRITICAL: Must call enable_post_processing() to initialize HDR/Tonemapping pipelines!
                if let Err(e) = renderer.enable_post_processing() {
                    log::warn!("Post-processing failed: {e}");
                    renderer.set_tonemapping_enabled(true);
                }
                renderer.refresh_draw_items();

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
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
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    // Application-side camera control (instead of auto_rotate)
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
