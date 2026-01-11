//! GPU Skinning example.
//!
//! Demonstrates bone matrix updates and skinned vertex rendering.
//! A simple animated arm with two joints.

use ash_renderer::prelude::*;
use ash_renderer::renderer::MaterialHandle;
use glam::{Mat4, Quat, Vec3};
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
    material_handle: MaterialHandle,
    mesh_handle: u32,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            start_time: Instant::now(),
            material_handle: MaterialHandle::default(),
            mesh_handle: 0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - GPU Skinning")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // Create a simple "arm" mesh (two segments)
                // Bottom half (indices 0..3) influenced by Bone 0
                // Top half (indices 4..7) influenced by Bone 1
                let mut mesh = Mesh::default();
                mesh.name = "Arm".into();
                mesh.skinned_vertices = vec![
                    // Bottom Segment (Stationary)
                    SkinnedVertex {
                        position: [-0.5, 0.0, -0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.0, 0.0],
                        joint_indices: [0, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SkinnedVertex {
                        position: [0.5, 0.0, -0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [1.0, 0.0],
                        joint_indices: [0, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SkinnedVertex {
                        position: [0.5, 0.0, 0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [1.0, 1.0],
                        joint_indices: [0, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SkinnedVertex {
                        position: [-0.5, 0.0, 0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.0, 1.0],
                        joint_indices: [0, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    // Top Segment (Animated)
                    SkinnedVertex {
                        position: [-0.5, 2.0, -0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.0, 0.0],
                        joint_indices: [1, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SkinnedVertex {
                        position: [0.5, 2.0, -0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [1.0, 0.0],
                        joint_indices: [1, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SkinnedVertex {
                        position: [0.5, 2.0, 0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [1.0, 1.0],
                        joint_indices: [1, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SkinnedVertex {
                        position: [-0.5, 2.0, 0.5],
                        normal: [0.0, 1.0, 0.0],
                        uv: [0.0, 1.0],
                        joint_indices: [1, 0, 0, 0],
                        joint_weights: [1.0, 0.0, 0.0, 0.0],
                    },
                ];
                mesh.indices = Some(vec![
                    0, 1, 2, 2, 3, 0, // Bottom face
                    4, 5, 6, 6, 7, 4, // Top face
                    0, 4, 1, 1, 4, 5, // Side
                    1, 5, 2, 2, 5, 6, // Side
                    2, 6, 3, 3, 6, 7, // Side
                    3, 7, 0, 0, 7, 4, // Side
                ]);

                // Upload mesh
                let mesh_handle = renderer.upload_mesh(mesh).unwrap_or(0);

                let green_material = Material {
                    color: [0.2, 0.8, 0.2, 1.0],
                    metallic: 0.1,
                    roughness: 0.8,
                    ..Default::default()
                };
                // Register and upload material
                let material_handle = renderer
                    .register_and_upload_material(green_material)
                    .unwrap();

                log::info!(
                    "✓ Registered and uploaded green material with handle {:?}",
                    material_handle
                );

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
                self.material_handle = material_handle;
                self.mesh_handle = mesh_handle;
                log::info!("Skinning test initialized!");
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
                    let aspect = size.width as f32 / size.height as f32;

                    // Update Joint Matrices
                    // Joint 0: Identity (Base)
                    // Joint 1: Bending arm
                    let joint0 = Mat4::IDENTITY;
                    let joint1 = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
                        * Mat4::from_quat(Quat::from_rotation_z(elapsed.sin() * 0.5))
                        * Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0));

                    unsafe {
                        renderer.update_joint_ssbo(&[joint0, joint1]).unwrap();
                    }

                    // Camera
                    let camera_pos = Vec3::new(5.0, 5.0, 5.0);
                    let target = Vec3::new(0.0, 1.0, 0.0);
                    let view = Mat4::look_at_rh(camera_pos, target, Vec3::Y);
                    let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 1000.0);
                    proj.y_axis.y *= -1.0;

                    // Draw
                    renderer.draw_skinned_mesh(
                        self.mesh_handle,
                        self.material_handle,
                        Mat4::IDENTITY,
                        0,
                    );
                    renderer.render_frame(view, proj, camera_pos, None).unwrap();
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
            _ => (),
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
