//! Basic triangle example.
//!
//! Demonstrates minimal renderer setup and rendering a simple triangle.

use ash_renderer::prelude::*;
use ash_renderer::renderer::Scene;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Window>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Triangle")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);
        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                let mut scene = Scene::new(
                    Arc::clone(&renderer.context.device.device),
                    Arc::clone(&renderer.context.alloc),
                    renderer.geometry_buffer(),
                )
                .expect("Failed to create scene");

                // Add a default directional light so the PBR shader has something to render
                scene.add_directional_light(ash_renderer::renderer::features::DirectionalLight {
                    direction: glam::Vec3::new(-1.0, -1.0, -1.0),
                    color: glam::Vec3::new(1.0, 1.0, 1.0),
                    intensity: 2.0,
                });

                // Create a simple triangle mesh
                let mut mesh = Mesh {
                    name: std::sync::Arc::from("Triangle"),
                    vertices: vec![
                        Vertex {
                            position: [0.0, -0.5, 0.0],
                            color: [1.0, 0.0, 0.0],
                            uv: [0.5, 0.0],
                            normal: [0.0, 0.0, 1.0],
                            tangent: [1.0, 0.0, 0.0, 1.0],
                            _padding: 0,
                        },
                        Vertex {
                            position: [0.5, 0.5, 0.0],
                            color: [0.0, 1.0, 0.0],
                            uv: [1.0, 1.0],
                            normal: [0.0, 0.0, 1.0],
                            tangent: [1.0, 0.0, 0.0, 1.0],
                            _padding: 0,
                        },
                        Vertex {
                            position: [-0.5, 0.5, 0.0],
                            color: [0.0, 0.0, 1.0],
                            uv: [0.0, 1.0],
                            normal: [0.0, 0.0, 1.0],
                            tangent: [1.0, 0.0, 0.0, 1.0],
                            _padding: 0,
                        },
                    ],
                    indices: Some(vec![0, 1, 2]),
                    ..Default::default()
                };

                // Upload mesh
                let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                let cmd_context = renderer.frame.cmds.context(upload_cmd);
                cmd_context
                    .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .unwrap();

                let mut staging_resources = Vec::new();

                let mesh_handle = scene
                    .upload_mesh(ash_renderer::renderer::MeshUploadInfo {
                        device: Arc::clone(&renderer.context.device.device),
                        allocator: Arc::clone(&renderer.context.alloc),
                        command_pool: renderer.frame.cmds.upload_command_pool_handle(),
                        command_buffer: upload_cmd,
                        queue: renderer.context.device.graphics_queue,
                        mesh: &mut mesh,
                        asset_manager: &mut renderer.resources.assets,
                        staging_resources: &mut staging_resources,
                        material_override: None,
                    })
                    .unwrap_or(0);

                cmd_context.end().unwrap();

                // Submit and wait
                let cmds = [upload_cmd];
                let submit_info = ash::vk::SubmitInfo::default().command_buffers(&cmds);
                unsafe {
                    renderer
                        .context
                        .device
                        .device
                        .queue_submit(
                            renderer.context.device.graphics_queue,
                            &[submit_info],
                            ash::vk::Fence::null(),
                        )
                        .unwrap();
                    renderer
                        .context
                        .device
                        .device
                        .queue_wait_idle(renderer.context.device.graphics_queue)
                        .unwrap();
                }

                // Create default material
                let material = Material {
                    name: "TriangleMat".to_string(),
                    color: [1.0, 1.0, 1.0, 1.0],
                    metallic: 0.0,
                    roughness: 1.0,
                    ..Default::default()
                };
                // Register and upload material
                let material_handle = scene.register_material(&material).unwrap();

                // Setup render command
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: glam::Mat4::IDENTITY,
                        ..Default::default()
                    });

                self.renderer = Some(renderer);
                scene.global_cluster_buffer = self
                    .renderer
                    .as_ref()
                    .unwrap()
                    .resources
                    .global_cluster_buffer
                    .clone();
                scene.material_storage_buffer = self
                    .renderer
                    .as_ref()
                    .unwrap()
                    .resources
                    .material_storage_buffer
                    .clone();
                self.scene = Some(scene);
                self.window = Some(window);
                log::info!("Renderer initialized successfully!");
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
                    let aspect = size.width as f32 / size.height.max(1) as f32;

                    // Simple static camera
                    let camera_pos = glam::Vec3::new(0.0, 0.0, 3.0);
                    let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::ZERO, glam::Vec3::Y);
                    // MODERN PROJECTION: Infinite Reverse Z
                    let mut proj = glam::Mat4::perspective_infinite_reverse_rh(
                        45.0_f32.to_radians(),
                        aspect,
                        0.5, // Near Plane
                    );
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    // Submit commands
                    if let Err(e) = renderer.submit_render_commands(scene, &self.render_commands) {
                        log::error!("Failed to submit render commands: {e}");
                    }

                    if let Err(e) = renderer.render_frame(scene, view, proj, camera_pos, None, None)
                    {
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
