//! GLTF Model Loading example.
//!
//! Demonstrates loading a GLTF/GLB model using the renderer's gltf_loader utility.

use ash_renderer::prelude::*;
use ash_renderer::renderer::Scene;
use ash_renderer::renderer::resources::gltf_loader;
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
                    Arc::clone(&renderer.context.device.device),
                    Arc::clone(&renderer.context.alloc),
                    renderer.geometry_buffer(),
                )
                .expect("Failed to create scene");

                // Path to a GLB model
                let glb_path = "assets/models/test.glb";

                if std::path::Path::new(glb_path).exists() {
                    log::info!("Loading model from: {glb_path}");
                    match gltf_loader::load_model(glb_path) {
                        Ok(meshes) => {
                            for mut mesh in meshes.into_iter() {
                                let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                                let cmd_context = renderer.frame.cmds.context(upload_cmd);
                                cmd_context
                                    .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                                    .unwrap();

                                let mut staging_resources = Vec::new();
                                let upload_res =
                                    scene.upload_mesh(ash_renderer::renderer::MeshUploadInfo {
                                        device: Arc::clone(&renderer.context.device.device),
                                        allocator: Arc::clone(&renderer.context.alloc),
                                        command_pool: renderer
                                            .frame
                                            .cmds
                                            .upload_command_pool_handle(),
                                        command_buffer: upload_cmd,
                                        queue: renderer.context.device.graphics_queue,
                                        mesh: &mut mesh,
                                        asset_manager: &mut renderer.resources.assets,
                                        staging_resources: &mut staging_resources,
                                        material_override: None,
                                    });

                                cmd_context.end().unwrap();

                                // Submit and wait
                                let cmds = [upload_cmd];
                                let submit_info =
                                    ash::vk::SubmitInfo::default().command_buffers(&cmds);
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

                                if let Ok(handle) = upload_res {
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
                    let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                    let cmd_context = renderer.frame.cmds.context(upload_cmd);
                    cmd_context
                        .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .unwrap();

                    let mut staging_resources = Vec::new();
                    let upload_res = scene.upload_mesh(ash_renderer::renderer::MeshUploadInfo {
                        device: Arc::clone(&renderer.context.device.device),
                        allocator: Arc::clone(&renderer.context.alloc),
                        command_pool: renderer.frame.cmds.upload_command_pool_handle(),
                        command_buffer: upload_cmd,
                        queue: renderer.context.device.graphics_queue,
                        mesh: &mut cube,
                        asset_manager: &mut renderer.resources.assets,
                        staging_resources: &mut staging_resources,
                        material_override: None,
                    });

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

                    if let Ok(handle) = upload_res {
                        self.mesh_handles.push(handle);
                        log::info!("Registered default cube with handle {handle}");
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
                    let _ = renderer.render_frame(scene, view, proj, camera_pos, None, None);
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
