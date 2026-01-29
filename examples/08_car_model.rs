//! Retro Muscle Car PBR Model example.
//!
//! Demonstrates loading and rendering a GLB model with PBR materials.
//! Features: GLB loading, proper material registration, HDR post-processing.

use ash_renderer::prelude::*;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::resources::gltf_loader;
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
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            start_time: Instant::now(),
            render_commands: Vec::new(),
        }
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::field_reassign_with_default)]
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

                log::info!("Loading car model from: {glb_path}");

                // Load the model using the utility bridge
                let mut meshes = match gltf_loader::load_model(glb_path) {
                    Ok(m) if !m.is_empty() => {
                        log::info!("✓ Loaded {} meshes from GLB file", m.len());
                        m
                    }
                    Ok(_) => {
                        log::error!("GLTF file contains no meshes");
                        event_loop.exit();
                        return;
                    }
                    Err(e) => {
                        log::error!("Failed to load GLB file: {e}");
                        event_loop.exit();
                        return;
                    }
                };

                // Take the first mesh for this example
                let mesh = meshes.remove(0);
                let mesh_name = mesh.name.clone();

                // Extract material properties from mesh BEFORE moving it
                let material_props = mesh.material_properties;
                let texture_indices = [
                    mesh.texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.normal_texture_index.map(|i| i as i32).unwrap_or(-1),
                    mesh.metallic_roughness_texture_index
                        .map(|i| i as i32)
                        .unwrap_or(-1),
                    mesh.occlusion_texture_index.map(|i| i as i32).unwrap_or(-1),
                ];
                let emissive_index = mesh.emissive_texture_index.map(|i| i as i32).unwrap_or(-1);

                // Upload mesh
                let mesh_handle = renderer.upload_mesh(mesh).unwrap_or(0);
                log::info!("✓ Mesh uploaded to GPU");

                // Create material from properties
                let material = if let Some(props) = material_props {
                    Material {
                        name: format!("{mesh_name}_material"),
                        color: props.base_color_factor,
                        metallic: props.metallic_factor,
                        roughness: props.roughness_factor,
                        emissive: [
                            props.emissive_factor[0],
                            props.emissive_factor[1],
                            props.emissive_factor[2],
                            1.0,
                        ],
                        occlusion_strength: props.occlusion_strength,
                        normal_scale: props.normal_scale,
                        alpha_cutoff: props.alpha_cutoff,
                        tint_index: -1,
                        is_transparent: props.base_color_factor[3] < 1.0,
                        texture_index: if texture_indices[0] >= 0 {
                            Some(texture_indices[0] as u32)
                        } else {
                            None
                        },
                        normal_texture_index: if texture_indices[1] >= 0 {
                            Some(texture_indices[1] as u32)
                        } else {
                            None
                        },
                        metallic_roughness_texture_index: if texture_indices[2] >= 0 {
                            Some(texture_indices[2] as u32)
                        } else {
                            None
                        },
                        occlusion_texture_index: if texture_indices[3] >= 0 {
                            Some(texture_indices[3] as u32)
                        } else {
                            None
                        },
                        emissive_texture_index: if emissive_index >= 0 {
                            Some(emissive_index as u32)
                        } else {
                            None
                        },
                    }
                } else {
                    Material {
                        name: format!("{mesh_name}_default"),
                        color: [0.8, 0.8, 0.8, 1.0],
                        metallic: 0.5,
                        roughness: 0.5,
                        ..Default::default()
                    }
                };

                // Register and upload material
                let _ = renderer.register_and_upload_material(material).unwrap();

                // Submit render command with all texture indices passed to shader
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle: ash_renderer::renderer::MaterialHandle::null(),
                        transform: Mat4::from_scale(Vec3::splat(1.0)),
                        ..Default::default()
                    });

                // Setup lighting (RAGE approach)
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::OutdoorDay)
                    .with_directional(
                        Vec3::new(-0.5, -1.0, -0.5).normalize(),
                        Vec3::splat(3.0),
                        1.0,
                    )
                    .build();

                renderer.set_lighting(&lighting);

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
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        let aspect = size.width as f32 / size.height as f32;
                        let radius = 8.0;
                        let camera_x = radius * (elapsed * 0.3).sin();
                        let camera_z = radius * (elapsed * 0.3).cos();
                        let camera_pos = Vec3::new(camera_x, 3.0, camera_z);
                        let target = Vec3::new(0.0, 0.5, 0.0);
                        let view = Mat4::look_at_rh(camera_pos, target, Vec3::Y);
                        let mut proj = Mat4::perspective_infinite_reverse_rh(
                            45.0_f32.to_radians(),
                            aspect,
                            0.1, // Near Plane
                        );
                        proj.y_axis.y *= -1.0;

                        let _ = renderer.submit_render_commands(&self.render_commands);
                        let _ = renderer.render_frame(view, proj, camera_pos, None);
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

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
