//! Retro Muscle Car PBR Model example.
//!
//! Demonstrates loading and rendering a GLB model with PBR materials.
//! Features: GLB loading, proper material registration, HDR post-processing.

use ash_renderer::prelude::*;
use ash_renderer::renderer::Scene;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::resources::gltf_loader;
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use glam::{Mat4, Vec3};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

enum LoaderMessage {
    ModelLoaded(Vec<ash_renderer::renderer::Mesh>),
    LoadError(String),
}

struct App {
    window: Option<Window>,
    // DROP ORDER MATTERS: Scene and buffers must drop BEFORE Renderer (destroys Device)
    tint_buffer: Option<Arc<Mutex<StorageBuffer<[f32; 4]>>>>,
    scene: Option<Scene>,
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
            tint_buffer: None,
            scene: None,
            renderer: None,
            start_time: Instant::now(),
            render_commands: Vec::new(),
            upload_fence: None,
            loader_rx: None,
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

        match Renderer::builder()
            .with_environment_map("assets/textures/skybox.hdr")
            .build(&surface_provider)
        {
            Ok(mut renderer) => {
                // Create Scene
                let mut scene = Scene::new(
                    Arc::clone(&renderer.context.device.device),
                    Arc::clone(&renderer.context.alloc),
                    renderer.geometry_buffer(),
                )
                .expect("Failed to create scene");

                // Register Global Default Tint Buffer (Required by Shader)
                let tint_data = [[1.0f32, 1.0, 1.0, 1.0]];
                let (tint_buffer, _tint_index) = renderer
                    .resources
                    .register_bindless_storage_buffer(&renderer.context, &tint_data, "GlobalTint")
                    .expect("Failed to register global tint buffer");

                // Keep buffer alive
                self.tint_buffer = Some(tint_buffer);

                // CRITICAL: Must call enable_post_processing() to initialize HDR/Tonemapping pipelines!
                if let Err(e) = renderer.enable_post_processing(&mut scene) {
                    log::warn!("Post-processing failed: {e}");
                }

                // Set reasonable defaults for PBR
                renderer.set_post_processing_config(
                    ash_renderer::renderer::systems::post_process::PostProcessConfig {
                        exposure: 1.0,
                        gamma: 2.2,
                        bloom_enabled: true,
                        bloom_intensity: 0.04,
                        tonemapping_enabled: true,
                    },
                );

                // Setup clean lighting (Sun + Ambient)
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::OutdoorDay)
                    .with_directional(
                        Vec3::new(-0.5, -1.0, -0.5).normalize(),
                        Vec3::new(1.0, 0.95, 0.8), // Warm sunlight
                        5.0,                       // PBR intensity
                    )
                    .build();
                scene.set_lighting(lighting);

                // Spawn Async Loader
                let glb_path = r"assets/test models/blue muscle car.glb";
                let (tx, rx) = mpsc::channel();
                self.loader_rx = Some(rx);

                if let Err(e) = std::thread::Builder::new()
                    .name("AsyncLoader".to_string())
                    .spawn(move || {
                        let tx_panic = tx.clone();
                        let result = std::panic::catch_unwind(move || {
                            log::info!("Async Loader: Starting disk I/O for car model...");
                            match gltf_loader::load_model(glb_path) {
                                Ok(meshes) => {
                                    let _ = tx.send(LoaderMessage::ModelLoaded(meshes));
                                    log::info!("Async Loader: Disk I/O complete.");
                                }
                                Err(e) => {
                                    log::error!("Async Loader: Failed to load model: {e}");
                                    let _ = tx.send(LoaderMessage::LoadError(e.to_string()));
                                }
                            }
                        });

                        if let Err(panic) = result {
                            log::error!("Async Loader: Thread panicked: {panic:?}");
                            let _ = tx_panic.send(LoaderMessage::LoadError(format!(
                                "Thread panicked: {panic:?}"
                            )));
                        }
                    })
                {
                    log::error!("Async Loader: Failed to spawn thread: {e}");
                    self.loader_rx = None;
                }

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
                            renderer.context.device.device.destroy_fence(fence, None);
                        }
                    }
                    self.upload_fence = None;
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(scene), Some(window)) =
                    (&mut self.renderer, &mut self.scene, &self.window)
                {
                    // 1. ASYNC LOADING: Check if model data has arrived from background thread
                    if let Some(rx) = &self.loader_rx {
                        match rx.try_recv() {
                            Ok(LoaderMessage::ModelLoaded(meshes)) => {
                                if meshes.is_empty() {
                                    log::error!("Async Loader: GLB file contains no meshes.");
                                    window.set_title("ASH Renderer - Error: No meshes found");
                                    self.loader_rx = None;
                                    return;
                                }

                                log::info!(
                                    "Async Loader: Data received on main thread ({} meshes). Starting GPU upload...",
                                    meshes.len()
                                );

                                // Prepare batched upload
                                let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                                let cmd_ctx = renderer.frame.cmds.context(upload_cmd);
                                cmd_ctx
                                    .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                                    .unwrap();

                                let mut staging_resources = Vec::new();

                                // Upload all meshes and generate render commands
                                for (i, mut mesh) in meshes.into_iter().enumerate() {
                                    let name = mesh.name.clone();
                                    let mesh_handle = scene
                                        .upload_mesh(ash_renderer::renderer::MeshUploadInfo {
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
                                        })
                                        .unwrap();

                                    let material_handle =
                                        scene.mesh_data[mesh_handle as usize].material_handle;

                                    log::info!(
                                        "Scheduled Mesh {i}: '{name}' (Handle: {mesh_handle:?}, Material: {material_handle:?})"
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
                                    let cmd_ctx = renderer.frame.cmds.context(upload_cmd);
                                    cmd_ctx.end().unwrap();
                                }
                                let cmds = [upload_cmd];
                                let submit_info =
                                    ash::vk::SubmitInfo::default().command_buffers(&cmds);

                                let upload_fence_result = unsafe {
                                    renderer
                                        .context
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
                                    renderer.context.device.device.queue_submit(
                                        renderer.context.device.graphics_queue,
                                        &[submit_info],
                                        upload_fence,
                                    )
                                };

                                if let Err(e) = submit_result {
                                    log::error!("Async Loader: GPU submit failed: {e}");
                                    unsafe {
                                        renderer
                                            .context
                                            .device
                                            .device
                                            .destroy_fence(upload_fence, None);
                                    }
                                    self.upload_fence = None;
                                } else {
                                    // Update state: Start render loop, stop polling
                                    self.upload_fence = Some(upload_fence);
                                }

                                self.loader_rx = None;
                                window.set_title("ASH Renderer - Retro Muscle Car (Live)");
                                log::info!(
                                    "✓ Car Model GPU Upload scheduled. Switching to rendering."
                                );
                            }
                            Ok(LoaderMessage::LoadError(err)) => {
                                log::error!("Async Loader: Failed to load model: {err}");
                                window.set_title(&format!("ASH Renderer - Error: {err}"));
                                self.loader_rx = None;
                            }
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                log::error!(
                                    "Async Loader: Thread disconnected unexpectedly (Check logs for panics)"
                                );
                                self.loader_rx = None;
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => {}
                        }
                    }

                    // 2. RENDERING: Check if we have something to draw
                    let mut ready_to_draw =
                        !self.render_commands.is_empty() && self.loader_rx.is_none();

                    // Non-blocking check for upload completion
                    if let Some(fence) = self.upload_fence {
                        unsafe {
                            let status = renderer.context.device.device.get_fence_status(fence);
                            match status {
                                Ok(true) => {
                                    // Upload complete! Destroy fence and proceed.
                                    renderer.context.device.device.destroy_fence(fence, None);
                                    self.upload_fence = None;
                                    log::info!("GPU Upload Complete. Starting Render Loop.");
                                }
                                Ok(false) => {
                                    // Still uploading. Skip frame.
                                    ready_to_draw = false;
                                }
                                Err(e) => {
                                    log::error!(
                                        "Failed to check fence status: {e}, destroying fence"
                                    );
                                    renderer.context.device.device.destroy_fence(fence, None);
                                    self.upload_fence = None;
                                    ready_to_draw = false;
                                }
                            }
                        }
                    } else if self.render_commands.is_empty() {
                        ready_to_draw = false;
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

                            let _ = renderer.submit_render_commands(scene, &self.render_commands);
                            let _ =
                                renderer.render_frame(scene, view, proj, camera_pos, None, None);
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
