//! Bindless Storage Buffer verification example.
//!
//! Demonstrates how to use Binding 2 (Storage Buffers) in the BindlessManager
//! to provide per-object or per-material configuration (like tints) without
//! using standard uniforms.

use ash_renderer::prelude::*;
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use glam::{Mat4, Vec3, Vec4};
use log;
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
    _start_time: Instant,
    frame_count: u32,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            _start_time: Instant::now(),
            tint_buffer: None,
            frame_count: 0,
            render_commands: Vec::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Bindless Buffer Test")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // 1. Register bindless storage buffer FIRST to get the index
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                let (tint_buffer_gpu, tint_index) = renderer
                    .register_bindless_storage_buffer(&tint_colors, "CubeTintBuffer")
                    .expect("Failed to register bindless storage buffer");

                log::info!("✓ Registered bindless tint buffer at index {}", tint_index);

                // 2. Set up material with MATTE ORANGE color (Phase 3)
                // Use the tint_index we just got!
                let material = Material {
                    color: [1.0, 0.5, 0.0, 1.0], // SOLID ORANGE
                    metallic: 0.0,               // Non-metallic
                    roughness: 0.7,              // Matte finish
                    tint_index: tint_index as i32,
                    ..Default::default()
                };
                log::info!("✓ Material set to MATTE ORANGE [Metallic 0.0, Roughness 0.7]");

                // Register and upload material
                let material_handle = renderer.register_and_upload_material(material).unwrap();
                log::info!(
                    "✓ Registered orange material with handle {:?}",
                    material_handle
                );

                // 3. Create a cube
                let mut cube = Mesh::create_cube();
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }
                cube.name = Arc::from("OrangeCube");
                cube.texture_data = None;
                log::info!("✓ Cube mesh created and renamed to 'OrangeCube' for Phase 1");

                // Upload mesh
                let mesh_handle = renderer.upload_mesh(cube).unwrap_or(0);
                log::info!("✓ Mesh uploaded to GPU");

                // 4. Setup render command
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: Mat4::IDENTITY,
                        ..Default::default()
                    });

                // 5. Phase 2 Settings: Proper HDR + Tonemapping
                if let Err(e) = renderer.enable_post_processing() {
                    log::warn!("Post-processing failed: {e}");
                    renderer.set_tonemapping_enabled(true);
                }

                // 6. Setup PHASE 2 Lighting: Balanced HDR
                renderer.set_lighting(
                    Vec3::new(-1.0, -1.0, -1.0).normalize(),
                    [2.5, 2.5, 2.5, 1.0], // Correct: White light with 2.5 brightness
                    0.2,                  // Improved ambient strength for better visibility
                );

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.tint_buffer = Some(tint_buffer_gpu);
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
                    let time = self._start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        let aspect = size.width as f32 / size.height as f32;
                        let radius = 5.0;
                        let camera_x = radius * time.sin();
                        let camera_z = radius * time.cos();
                        let camera_pos = Vec3::new(camera_x, 2.0, camera_z);

                        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                        let mut proj =
                            Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
                        proj.y_axis.y *= -1.0;

                        // Dynamic Lighting
                        let light_angle = time * 0.5;
                        let light_dir =
                            Vec3::new(-light_angle.cos(), -1.0, -light_angle.sin()).normalize();

                        renderer.set_lighting(light_dir, [2.5, 2.5, 2.5, 1.0], 0.2);

                        // Submit commands
                        if let Err(e) = renderer.submit_render_commands(&self.render_commands) {
                            log::error!("Failed to submit render commands: {e}");
                        }

                        if let Err(e) = renderer.render_frame(view, proj, camera_pos, None) {
                            log::error!("Render error: {e}");
                        }
                        self.frame_count += 1;
                    }
                }
                if let Some(window) = &self.window {
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
