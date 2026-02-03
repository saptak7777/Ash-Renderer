//! Retro Muscle Car PBR Model example.
//!
//! Demonstrates loading and rendering a GLB model with PBR materials.
//! Features: GLB loading, proper material registration, HDR post-processing.

use ash_renderer::prelude::*;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::resources::gltf_loader;
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use glam::{Mat4, Vec3};
use parking_lot::Mutex;
use std::mem::ManuallyDrop;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

enum LoaderMessage {
    ModelLoaded(Vec<ash_renderer::renderer::Mesh>),
}

struct App {
    window: Option<Window>,
    // Use ManuallyDrop to explicitly control destruction order.
    // This MUST be dropped BEFORE renderer to avoid STATUS_ACCESS_VIOLATION on exit.
    tint_buffer: ManuallyDrop<Option<Arc<Mutex<StorageBuffer<[f32; 4]>>>>>,
    renderer: Option<Renderer>,
    start_time: Instant,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
    upload_fence: Option<ash::vk::Fence>,
    loader_rx: Option<mpsc::Receiver<LoaderMessage>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            tint_buffer: ManuallyDrop::new(None),
            renderer: None,
            start_time: Instant::now(),
            render_commands: Vec::new(),
            upload_fence: None,
            loader_rx: None,
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            // Manually drop the tint buffer FIRST while the renderer/device is still alive.
            log::info!("App: Manually dropping GPU resources...");
            ManuallyDrop::drop(&mut self.tint_buffer);
        }
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::field_reassign_with_default)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Retro Muscle Car (Loading...)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                // Register Global Default Tint Buffer (Required by Shader)
                let tint_data = [[1.0f32, 1.0, 1.0, 1.0]];
                let (tint_buffer, _tint_index) = renderer
                    .register_bindless_storage_buffer(&tint_data, "GlobalTint")
                    .expect("Failed to register global tint buffer");

                // Keep buffer alive
                *self.tint_buffer = Some(tint_buffer);

                // Setup initial lighting (Void state)
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::OutdoorDay)
                    .with_directional(
                        Vec3::new(-0.5, -1.0, -0.5).normalize(),
                        Vec3::splat(3.0),
                        1.0,
                    )
                    .build();
                renderer.set_lighting(&lighting);

                // Spawn Async Loader
                let glb_path = r"C:\Users\tilok\Downloads\car retro muscle\base_basic_pbr.glb";
                let (tx, rx) = mpsc::channel();
                self.loader_rx = Some(rx);

                std::thread::spawn(move || {
                    let result = std::panic::catch_unwind(move || {
                        log::info!("Async Loader: Starting disk I/O for car model...");
                        match gltf_loader::load_model(glb_path) {
                            Ok(meshes) => {
                                let _ = tx.send(LoaderMessage::ModelLoaded(meshes));
                                log::info!("Async Loader: Disk I/O complete.");
                            }
                            Err(e) => {
                                log::error!("Async Loader: Failed to load model: {e}");
                            }
                        }
                    });

                    if let Err(panic) = result {
                        log::error!("Async Loader: Thread panicked: {:?}", panic);
                    }
                });

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
            WindowEvent::CloseRequested => {
                // Cleanup potentially pending upload fence
                if let Some(fence) = self.upload_fence {
                    unsafe {
                        if let Some(renderer) = &self.renderer {
                            renderer.device.device.destroy_fence(fence, None);
                        }
                    }
                    self.upload_fence = None;
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    // 1. ASYNC LOADING: Check if model data has arrived from background thread
                    if let Some(rx) = &self.loader_rx {
                        match rx.try_recv() {
                            Ok(LoaderMessage::ModelLoaded(meshes)) => {
                                log::info!("Async Loader: Data received on main thread. Starting GPU upload...");

                                // Prepare batched upload
                                let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                                let cmd_ctx = renderer.cmds.context(upload_cmd);
                                cmd_ctx
                                    .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                                    .unwrap();

                                let mut staging_resources = Vec::new();

                                // Upload all meshes and generate render commands
                                for (i, mesh) in meshes.into_iter().enumerate() {
                                    let name = mesh.name.clone();
                                    let mesh_handle = renderer
                                        .upload_mesh(mesh, upload_cmd, &mut staging_resources)
                                        .unwrap();
                                    let material_handle = renderer.get_mesh_material(mesh_handle);

                                    log::info!(
                                        "Scheduled Mesh {}: '{}' (Handle: {:?}, Material: {:?})",
                                        i,
                                        name,
                                        mesh_handle,
                                        material_handle
                                    );

                                    self.render_commands.push(
                                        ash_renderer::renderer::RenderCommand {
                                            mesh_handle,
                                            material_handle,
                                            transform: Mat4::from_scale(Vec3::splat(1.0)),
                                            cast_shadows: true,
                                            receive_shadows: true,
                                            ..Default::default()
                                        },
                                    );
                                }

                                // Finalize batch and submit
                                {
                                    let cmd_ctx = renderer.cmds.context(upload_cmd);
                                    cmd_ctx.end().unwrap();
                                }
                                let cmds = [upload_cmd];
                                let submit_info =
                                    ash::vk::SubmitInfo::default().command_buffers(&cmds);

                                let upload_fence_result = unsafe {
                                    renderer
                                        .device
                                        .device
                                        .create_fence(&ash::vk::FenceCreateInfo::default(), None)
                                };

                                let upload_fence = match upload_fence_result {
                                    Ok(f) => f,
                                    Err(e) => {
                                        log::error!(
                                            "Async Loader: Failed to create upload fence: {e}"
                                        );
                                        self.loader_rx = None;
                                        return;
                                    }
                                };

                                let submit_result = unsafe {
                                    renderer.device.device.queue_submit(
                                        renderer.device.graphics_queue,
                                        &[submit_info],
                                        upload_fence,
                                    )
                                };

                                if let Err(e) = submit_result {
                                    log::error!("Async Loader: GPU submit failed: {e}");
                                    unsafe {
                                        renderer.device.device.destroy_fence(upload_fence, None);
                                    }
                                    self.upload_fence = None;
                                } else {
                                    // Update state: Start render loop, stop polling
                                    self.upload_fence = Some(upload_fence);
                                }

                                self.loader_rx = None;

                                // Setup final lighting (RAGE approach)
                                let lighting = LightingBuilder::new()
                                    .with_ambient_preset(AmbientPreset::OutdoorDay)
                                    .with_directional(
                                        Vec3::new(-0.5, -1.0, -0.5).normalize(),
                                        Vec3::splat(3.0),
                                        1.0,
                                    )
                                    .build();
                                renderer.set_lighting(&lighting);

                                window.set_title("ASH Renderer - Retro Muscle Car (Live)");
                                log::info!(
                                    "✓ Car Model GPU Upload scheduled. Switching to rendering."
                                );
                            }
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                log::error!("Async Loader: Thread disconnected unexpectedly (Check logs for panics)");
                                self.loader_rx = None;
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => {}
                        }
                    }

                    // 2. RENDERING: Start assuming ready unless fence says otherwise
                    let mut ready_to_draw = true;

                    // Non-blocking check for upload completion
                    if let Some(fence) = self.upload_fence {
                        unsafe {
                            let status = renderer.device.device.get_fence_status(fence);
                            match status {
                                Ok(true) => {
                                    // Upload complete! Destroy fence and proceed.
                                    renderer.device.device.destroy_fence(fence, None);
                                    self.upload_fence = None;
                                    log::info!("GPU Upload Complete. Starting Render Loop.");
                                }
                                _ => {
                                    // Still uploading or error. Skip frame to keep window responsive.
                                    ready_to_draw = false;
                                }
                            }
                        }
                    }

                    if ready_to_draw {
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
