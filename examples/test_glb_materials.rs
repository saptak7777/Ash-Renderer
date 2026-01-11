//! Test GLB material registration
//!
//! This example tests if GLB materials are properly registered and rendered.

use ash::vk;
use ash_renderer::prelude::*;
use glam::{Mat4, Vec3};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
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
            .with_title("ASH Renderer - GLB Material Test")
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // Try to load a GLB file if it exists
                let glb_paths = ["assets/models/test.glb", "test.glb", "assets/test.glb"];

                let mut loaded_model = false;
                for path in &glb_paths {
                    if std::path::Path::new(path).exists() {
                        log::info!("Loading GLB model from: {}", path);
                        match Mesh::load_all_from_gltf(path) {
                            Ok(meshes) => {
                                log::info!("Loaded {} meshes from GLB", meshes.len());

                                // Register each mesh with the renderer
                                for (i, mut mesh) in meshes.into_iter().enumerate() {
                                    let handle = (i + 1) as u32; // Use 1-based handles
                                    if let Err(e) = renderer.register_mesh_handle(handle, &mut mesh)
                                    {
                                        log::error!("Failed to register mesh {}: {}", i, e);
                                    } else {
                                        log::info!(
                                            "Registered mesh '{}' with handle {}",
                                            mesh.name,
                                            handle
                                        );

                                        // Check if material was registered
                                        let mesh_data = renderer.mesh_data();
                                        if (handle as usize) < mesh_data.len() {
                                            let mat_handle =
                                                mesh_data[handle as usize].material_handle;
                                            if renderer
                                                .material_manager()
                                                .is_handle_valid(mat_handle)
                                            {
                                                log::info!(
                                                    "✅ Material registered for mesh '{}' (handle {:?})",
                                                    mesh.name,
                                                    mat_handle
                                                );

                                                // CRITICAL FIX: Upload the automatically registered material to GPU
                                                let material = renderer
                                                    .material_manager()
                                                    .get_material(mat_handle)
                                                    .clone();
                                                let _ = renderer.upload_material_to_gpu(
                                                    mat_handle.index as u32,
                                                    &material,
                                                );
                                                log::info!(
                                                    "✅ Material uploaded to GPU: {:?}",
                                                    mat_handle
                                                );
                                            } else {
                                                log::warn!("❌ No material registered for mesh '{}' (handle {:?})", mesh.name, mat_handle);
                                            }
                                        }
                                    }
                                }
                                loaded_model = true;
                                break;
                            }
                            Err(e) => {
                                log::error!("Failed to load GLB from {}: {}", path, e);
                            }
                        }
                    }
                }

                if !loaded_model {
                    log::info!("No GLB file found. Using default cube.");
                    // Create a test cube with material properties
                    let mut cube = Mesh::create_cube();
                    cube.material_properties = Some(
                        ash_renderer::renderer::resources::mesh::MaterialProperties {
                            base_color_factor: [0.8, 0.2, 0.2, 1.0],
                            metallic_factor: 0.8,
                            roughness_factor: 0.2,
                            emissive_factor: [0.0, 0.0, 0.0, 1.0],
                            occlusion_strength: 1.0,
                            normal_scale: 1.0,
                            alpha_cutoff: 0.5,
                        },
                    );

                    if let Err(e) = renderer.register_mesh_handle(1, &mut cube) {
                        log::error!("Failed to register test cube: {}", e);
                    } else {
                        log::info!("Test cube registered with material properties");

                        // Check if material was registered
                        let mesh_data = renderer.mesh_data();
                        if !mesh_data.is_empty() {
                            let mat_handle = mesh_data[0].material_handle;
                            if renderer.material_manager().is_handle_valid(mat_handle) {
                                log::info!(
                                    "✅ Material registered for test cube (handle {:?})",
                                    mat_handle
                                );
                            } else {
                                log::warn!(
                                    "❌ No material registered for test cube (handle {:?})",
                                    mat_handle
                                );
                            }
                        }
                    }

                    // Submit render command with null handle to test fallback
                    let _ =
                        renderer.submit_render_commands(&[ash_renderer::renderer::RenderCommand {
                            mesh_handle: 1,
                            material_handle: ash_renderer::renderer::MaterialHandle::null(), // Should fallback to mesh's registered material
                            transform: Mat4::IDENTITY,
                            is_skinned: false,
                            joint_offset: 0,
                            cast_shadows: true,
                            receive_shadows: true,
                            is_transparent: false,
                            is_hidden: false,
                        }]);
                }

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
                    // Simple camera setup
                    let _elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Static camera position
                    let camera_pos = Vec3::new(0.0, 0.0, 3.0);
                    let target = Vec3::ZERO;
                    let up = Vec3::Y;

                    let view = Mat4::look_at_rh(camera_pos, target, up);
                    let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.5, 100.0);
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    if let Err(e) = renderer.render_frame(view, proj, camera_pos, None) {
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

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();

    event_loop.run_app(&mut app).unwrap();
}
