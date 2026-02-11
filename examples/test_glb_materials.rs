//! Test GLB material registration
//!
//! This example tests if GLB materials are properly registered and rendered.

use ash::vk;
use ash_renderer::prelude::*;
use ash_renderer::renderer::resources::gltf_loader;
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
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: None,
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    #[allow(clippy::field_reassign_with_default)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - GLB Material Test")
            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
                let mut scene = Scene::new(
                    Arc::clone(&renderer.device.device),
                    Arc::clone(&renderer.alloc),
                    renderer.geometry_buffer(),
                )
                .expect("Failed to create scene");

                // Try to load a GLB file if it exists
                let glb_paths = ["assets/models/test.glb", "test.glb", "assets/test.glb"];

                let mut loaded_model = false;
                for path in &glb_paths {
                    if std::path::Path::new(path).exists() {
                        log::info!("Loading GLB model from: {path}");
                        match gltf_loader::load_model(path) {
                            Ok(meshes) => {
                                log::info!("Loaded {} meshes from GLB", meshes.len());

                                // Register each mesh with the renderer
                                for (i, mut mesh) in meshes.into_iter().enumerate() {
                                    let mesh_name = mesh.name.clone();
                                    // Use Scene::upload_mesh directly
                                    let upload_cmd =
                                        renderer.get_transfer_command_buffer().unwrap();
                                    let cmd_context = renderer.cmds.context(upload_cmd);
                                    cmd_context
                                        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                                        .unwrap();

                                    let mut staging_resources = Vec::new();
                                    let upload_res = scene.upload_mesh(
                                        Arc::clone(&renderer.device.device),
                                        Arc::clone(&renderer.alloc),
                                        renderer.cmds.upload_command_pool_handle(),
                                        upload_cmd,
                                        &renderer.device.graphics_queue,
                                        &mut mesh,
                                        &mut renderer.assets,
                                        &mut staging_resources,
                                    );

                                    cmd_context.end().unwrap();

                                    // Submit and wait
                                    let cmds = [upload_cmd];
                                    let submit_info =
                                        vk::SubmitInfo::default().command_buffers(&cmds);
                                    unsafe {
                                        renderer
                                            .device
                                            .device
                                            .queue_submit(
                                                renderer.device.graphics_queue,
                                                &[submit_info],
                                                vk::Fence::null(),
                                            )
                                            .unwrap();
                                        renderer
                                            .device
                                            .device
                                            .queue_wait_idle(renderer.device.graphics_queue)
                                            .unwrap();
                                    }

                                    match upload_res {
                                        Err(e) => {
                                            log::error!("Failed to register mesh {i}: {e}");
                                        }
                                        Ok(handle) => {
                                            log::info!(
                                            "Registered mesh '{mesh_name}' with handle {handle}"
                                        );

                                            // Check if material was registered
                                            let mat_handle =
                                                scene.mesh_data[handle as usize].material_handle;
                                            if !mat_handle.is_null() {
                                                if scene
                                                    .material_manager
                                                    .is_handle_valid(mat_handle)
                                                {
                                                    log::info!(
                                                    "✅ Material registered for mesh '{mesh_name}' (handle {mat_handle:?})"
                                                );
                                                    // Upload the automatically registered material to GPU
                                                    let material = scene
                                                        .material_manager
                                                        .get_material(mat_handle)
                                                        .clone();
                                                    let _ =
                                                        scene.register_material(&material).unwrap();
                                                    log::info!(
                                                    "✅ Material uploaded to GPU: {mat_handle:?}"
                                                );
                                                } else {
                                                    log::warn!("❌ No material registered for mesh '{mesh_name}' (handle {mat_handle:?})");
                                                }
                                            }
                                        }
                                    }
                                }
                                loaded_model = true;
                                break;
                            }
                            Err(e) => {
                                log::error!("Failed to load GLB from {path}: {e}");
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

                    let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                    let cmd_context = renderer.cmds.context(upload_cmd);
                    cmd_context
                        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .unwrap();

                    let mut staging_resources = Vec::new();
                    let upload_res = scene.upload_mesh(
                        Arc::clone(&renderer.device.device),
                        Arc::clone(&renderer.alloc),
                        renderer.cmds.upload_command_pool_handle(),
                        upload_cmd,
                        &renderer.device.graphics_queue,
                        &mut cube,
                        &mut renderer.assets,
                        &mut staging_resources,
                    );

                    cmd_context.end().unwrap();

                    // Submit and wait
                    let cmds = [upload_cmd];
                    let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
                    unsafe {
                        renderer
                            .device
                            .device
                            .queue_submit(
                                renderer.device.graphics_queue,
                                &[submit_info],
                                vk::Fence::null(),
                            )
                            .unwrap();
                        renderer
                            .device
                            .device
                            .queue_wait_idle(renderer.device.graphics_queue)
                            .unwrap();
                    }

                    if let Ok(handle) = upload_res {
                        log::info!("Test cube registered with material properties");

                        // Check if material was registered
                        let mat_handle = scene.mesh_data[handle as usize].material_handle;
                        if !mat_handle.is_null() {
                            if scene.material_manager.is_handle_valid(mat_handle) {
                                log::info!(
                                    "✅ Material registered for test cube (handle {mat_handle:?})"
                                );
                            }
                        }
                    } else {
                        log::error!("Failed to register test cube");
                    }

                    // Submit render command with null handle to test fallback
                    let mut cmd = ash_renderer::renderer::RenderCommand::default();
                    cmd.mesh_handle = 1; // Note: This might be invalid if load failed, but it's a test
                    cmd.material_handle = ash_renderer::renderer::MaterialHandle::null();
                    cmd.transform = Mat4::IDENTITY;

                    let _ = renderer.submit_render_commands(&mut scene, &[cmd]);
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
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    let camera_pos = Vec3::new(0.0, 0.0, 3.0);
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

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::default();

    event_loop.run_app(&mut app).unwrap();
}
