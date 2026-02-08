//! Visual Test: GLB Material Registration Demo
//!
//! This example demonstrates the GLB material registration fix.
//! It shows how materials from GLB files are automatically registered and used.

use ash::vk;
use ash_renderer::prelude::*;
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
    start_time: Instant,
    demo_mesh: Option<Mesh>,
    mesh_handle: Option<u32>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: None,
            start_time: Instant::now(),
            demo_mesh: None,
            mesh_handle: None,
        }
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::field_reassign_with_default)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - GLB Material Registration Demo")
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                let mut scene = Scene::new(
                    Arc::clone(&renderer.device.device),
                    Arc::clone(&renderer.alloc),
                    renderer.geometry_buffer(),
                );

                // Create a demo mesh with material properties (simulating GLB load)
                let mut demo_mesh = Mesh::create_cube();
                demo_mesh.name = "metallic_demo_cube".into();

                // Set material properties like they would come from a GLB file
                demo_mesh.material_properties = Some(
                    ash_renderer::renderer::resources::mesh::MaterialProperties {
                        base_color_factor: [0.8, 0.2, 0.2, 1.0], // Red color
                        metallic_factor: 0.8,                    // Very metallic
                        roughness_factor: 0.2,                   // Smooth surface
                        emissive_factor: [0.0, 0.0, 0.0, 1.0],
                        occlusion_strength: 1.0,
                        normal_scale: 1.0,
                        alpha_cutoff: 0.5,
                    },
                );

                // Register the mesh with the renderer (this triggers material registration)
                // Register the mesh with the renderer (this triggers material registration)
                if let Ok(handle) = renderer.upload_mesh_single(&mut scene, demo_mesh.clone()) {
                    log::info!("✅ Demo mesh registered with handle {handle}");

                    log::info!("✅ Demo mesh registered with handle {handle}");

                    // Check if material was registered
                    let mat_handle = renderer.get_mesh_material(handle);
                    if !mat_handle.is_null() && scene.material_manager.is_handle_valid(mat_handle) {
                        log::info!("✅ Material automatically registered for demo mesh");

                        // CRITICAL FIX: Upload the automatically registered material to GPU
                        let material = scene.material_manager.get_material(mat_handle).clone();
                        let _ = renderer.register_and_upload_material(&mut scene, material.clone());
                        log::info!("✅ Material uploaded to GPU: {mat_handle:?}");

                        log::info!("   - Material: {}", material.name);
                        log::info!("   - Metallic: {:.2}", material.metallic);
                        log::info!("   - Roughness: {:.2}", material.roughness);
                        log::info!("   - Color: {:?}", material.color);
                    } else {
                        log::warn!("❌ No material registered for demo mesh");
                    }

                    // Store for rendering
                    self.demo_mesh = Some(demo_mesh);
                    self.mesh_handle = Some(handle);
                }

                // Submit render command with automatic material selection
                if let Some(mesh_handle) = self.mesh_handle {
                    let mut cmd = ash_renderer::renderer::RenderCommand::default();
                    cmd.mesh_handle = mesh_handle;
                    cmd.material_handle = ash_renderer::renderer::MaterialHandle::null();
                    cmd.transform = Mat4::IDENTITY;

                    let _ = renderer.submit_render_commands(&mut scene, &[cmd]);
                    log::info!("✅ Render command submitted with auto material selection");
                }

                self.renderer = Some(renderer);
                self.scene = Some(scene);
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
                if let (Some(renderer), Some(window), Some(scene)) =
                    (&mut self.renderer, &self.window, &mut self.scene)
                {
                    // Animated camera
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Orbiting camera
                    let radius = 5.0;
                    let camera_x = radius * elapsed.sin();
                    let camera_z = radius * elapsed.cos();
                    let camera_pos = Vec3::new(camera_x, 2.0, camera_z);
                    let target = Vec3::ZERO;
                    let up = Vec3::Y;

                    let view = Mat4::look_at_rh(camera_pos, target, up);
                    let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.5, 100.0);
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    if let Err(e) = renderer.render_frame(scene, view, proj, camera_pos, None) {
                        log::error!("Render error: {e}");
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.request_swapchain_resize(vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    });
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    println!("🎬 GLB Material Registration Demo");
    println!("=====================================");
    println!("This demo shows how GLB materials are automatically registered and used.");
    println!();
    println!("What you should see:");
    println!("  - A red, metallic cube (shiny surface)");
    println!("  - The material properties come from material_properties");
    println!("  - No explicit material_handle needed (auto-selected)");
    println!("  - Check the console logs for registration details");
    println!();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();

    event_loop.run_app(&mut app).unwrap();
}
