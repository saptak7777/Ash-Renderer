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
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            _start_time: Instant::now(),
            tint_buffer: None,
            frame_count: 0,
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
                // 1. Create a cube with NO texture data initially
                let mut cube = Mesh::create_cube();
                // Override vertex colors to WHITE so they don't affect the material color
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }
                // RENAMING is critical because the renderer caches meshes by name!
                cube.name = Arc::from("OrangeCube");
                cube.texture_data = None;
                log::info!("✓ Cube mesh created and renamed to 'OrangeCube' for Phase 1");

                // 2. Set up material with MATTE ORANGE color (Phase 3)
                let material = Material {
                    color: [1.0, 0.5, 0.0, 1.0], // SOLID ORANGE
                    metallic: 0.0,               // Non-metallic
                    roughness: 0.7,              // Matte finish
                    ..Default::default()
                };
                log::info!("✓ Material set to MATTE ORANGE [Metallic 0.0, Roughness 0.7]");

                if let Err(e) = renderer.set_mesh(cube) {
                    log::error!("Failed to set mesh: {e}");
                    event_loop.exit();
                    return;
                }
                log::info!("✓ Mesh uploaded to GPU");
                *renderer.material_mut() = material.clone();

                // CRITICAL FIX 1: Register material with bindless buffer so shader uses orange, not grey
                let material_index = 1u32; // Index 0 is reserved for default grey material
                renderer.register_material_handle(material_index, &material);
                
                // CRITICAL: Upload the material data to GPU so shader can access it
                if let Err(e) = renderer.upload_material_to_gpu(material_index, &material) {
                    log::error!("Failed to upload material to GPU: {e}");
                }
                
                // Update mesh_data so the draw_items path uses the correct material
                if let Some(mesh_data) = renderer.get_mesh_data_mut(0) {
                    mesh_data.material_handle = material_index;
                }
                
                log::info!("✓ Registered orange material at index {}", material_index);

                // 3. Register bindless storage buffer
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                let (tint_buffer_gpu, tint_index) = renderer
                    .register_bindless_storage_buffer(&tint_colors, "CubeTintBuffer")
                    .expect("Failed to register bindless storage buffer");

                log::info!("✓ Registered bindless tint buffer at index {}", tint_index);

                // 4. Phase 2 Settings: Proper HDR + Tonemapping
                renderer.material_mut().tint_index = -1;
                // CRITICAL: Must call enable_post_processing() to initialize HDR/Tonemapping pipelines!
                if let Err(e) = renderer.enable_post_processing() {
                    log::warn!("Post-processing failed: {e}");
                    renderer.set_tonemapping_enabled(true);
                }
                renderer.refresh_draw_items();

                // 5. Setup PHASE 2 Lighting: Balanced HDR
                // Initial lighting setup (will be updated per frame)
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

                        // Dynamic Lighting: Rotate light with time for moving shadows
                        let light_angle = time * 0.5;
                        let light_dir =
                            Vec3::new(-light_angle.cos(), -1.0, -light_angle.sin()).normalize();

                        renderer.set_lighting(
                            light_dir,
                            [2.5, 2.5, 2.5, 1.0], // Balanced white HDR light
                            0.2,                  // Improved ambient strength for better visibility
                        );

                        if let Err(e) = renderer.render_frame(view, proj, camera_pos) {
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
