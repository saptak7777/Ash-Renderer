//! Retro Muscle Car PBR Model example.
//!
//! Demonstrates loading and rendering a GLB model with PBR materials.
//! Features: GLB loading, proper material registration, HDR post-processing.

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
            .with_title("ASH Renderer - Retro Muscle Car")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // Load the car model from GLB file
                let glb_path = r"C:\Users\tilok\Downloads\car retro muscle\base_basic_pbr.glb";
                
                log::info!("Loading car model from: {}", glb_path);
                
                // Load the first mesh from the GLB file
                let mesh = match Mesh::from_gltf(glb_path) {
                    Ok(m) => {
                        log::info!("✓ Loaded mesh '{}' from GLB file", m.name);
                        m
                    }
                    Err(e) => {
                        log::error!("Failed to load GLB file: {e}");
                        event_loop.exit();
                        return;
                    }
                };

                // Extract material properties from mesh BEFORE moving it to set_mesh
                let material_props = mesh.material_properties;
                let mesh_name = mesh.name.clone();
                let texture_indices = [
                    mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.metallic_roughness_texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
                ];
                let emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

                // Register mesh with renderer (this also registers the material)
                if let Err(e) = renderer.set_mesh(mesh) {
                    log::error!("Failed to set mesh: {e}");
                    event_loop.exit();
                    return;
                }
                log::info!("✓ Mesh uploaded to GPU");

                // Create material from mesh properties (loaded from GLB)
                let material = if let Some(props) = material_props {
                    Material {
                        name: format!("{}_material", mesh_name),
                        color: props.base_color_factor,
                        metallic: props.metallic_factor,
                        roughness: props.roughness_factor,
                        emissive: [0.0, 0.0, 0.0, 1.0],
                        occlusion_strength: 1.0,
                        normal_scale: 1.0,
                        alpha_cutoff: 0.1,
                        tint_index: -1,
                        is_transparent: props.base_color_factor[3] < 1.0,
                        // Use texture indices extracted from mesh
                        texture_index: if texture_indices[0] >= 0 { Some(texture_indices[0] as u32) } else { None },
                        normal_texture_index: if texture_indices[1] >= 0 { Some(texture_indices[1] as u32) } else { None },
                        metallic_roughness_texture_index: if texture_indices[2] >= 0 { Some(texture_indices[2] as u32) } else { None },
                        occlusion_texture_index: if texture_indices[3] >= 0 { Some(texture_indices[3] as u32) } else { None },
                        emissive_texture_index: if emissive_index >= 0 { Some(emissive_index as u32) } else { None },
                    }
                } else {
                    // Fallback default material
                    Material {
                        name: format!("{}_default", mesh_name),
                        color: [0.8, 0.8, 0.8, 1.0],
                        metallic: 0.5,
                        roughness: 0.5,
                        emissive: [0.0, 0.0, 0.0, 1.0],
                        occlusion_strength: 1.0,
                        normal_scale: 1.0,
                        alpha_cutoff: 0.1,
                        tint_index: -1,
                        is_transparent: false,
                        texture_index: if texture_indices[0] >= 0 { Some(texture_indices[0] as u32) } else { None },
                        normal_texture_index: if texture_indices[1] >= 0 { Some(texture_indices[1] as u32) } else { None },
                        metallic_roughness_texture_index: if texture_indices[2] >= 0 { Some(texture_indices[2] as u32) } else { None },
                        occlusion_texture_index: if texture_indices[3] >= 0 { Some(texture_indices[3] as u32) } else { None },
                        emissive_texture_index: if emissive_index >= 0 { Some(emissive_index as u32) } else { None },
                    }
                };

                // Upload the material that set_mesh just registered
                if let Some(mesh_data) = renderer.mesh_data().first() {
                    let material_handle = mesh_data.material_handle;
                    if let Err(e) = renderer.upload_material_to_gpu(material_handle.index as u32, &material) {
                        log::error!("Failed to upload material to GPU: {e}");
                        event_loop.exit();
                        return;
                    }
                    log::info!(
                        "✓ Uploaded material to GPU with handle {:?}",
                        material_handle
                    );
                } else {
                    log::error!("Mesh data missing after set_mesh; aborting");
                    event_loop.exit();
                    return;
                }

                // Register bindless storage buffer (prevents crash like example 06)
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                if let Ok((tint_buffer, _tint_index)) =
                    renderer.register_bindless_storage_buffer(&tint_colors, "DefaultTint")
                {
                    self.tint_buffer = Some(tint_buffer); // Keep buffer alive!
                    log::info!("✓ Registered default tint buffer");
                }

                // Update renderer material state once all resources exist
                *renderer.material_mut() = material.clone();
                renderer.material_mut().tint_index = -1;

                if let Err(e) = renderer.enable_post_processing() {
                    log::warn!("Post-processing failed: {e}");
                    renderer.set_tonemapping_enabled(true);
                }
                renderer.refresh_draw_items();

                // Setup lighting for car model
                renderer.set_lighting(
                    Vec3::new(-0.5, -1.0, -0.5).normalize(), // Light from top-front-left
                    [3.0, 3.0, 3.0, 1.0], // Bright white light
                    0.3, // Moderate ambient
                );

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
                log::info!("Car model renderer initialized!");
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
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        let aspect = size.width as f32 / size.height as f32;

                        // Orbiting camera around the car
                        let radius = 8.0;
                        let camera_x = radius * (elapsed * 0.3).sin();
                        let camera_z = radius * (elapsed * 0.3).cos();
                        let camera_y = 3.0 + (elapsed * 0.2).sin() * 1.0;
                        
                        let camera_pos = Vec3::new(camera_x, camera_y, camera_z);
                        let target = Vec3::new(0.0, 0.5, 0.0); // Look slightly above ground
                        let up = Vec3::Y;

                        let view = Mat4::look_at_rh(camera_pos, target, up);
                        let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
                        proj.y_axis.y *= -1.0; // Vulkan Y-flip

                        if let Err(e) = renderer.render_frame(view, proj, camera_pos) {
                            log::error!("Render error: {e}");
                        }
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

    log::info!("Starting Retro Muscle Car example...");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");

    Ok(())
}
