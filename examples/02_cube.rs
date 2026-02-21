//! Cube with textures example.
//!
//! Demonstrates textured cube rendering with materials.
//! Shows how to control the camera from the application.

use ash_renderer::prelude::*;
use ash_renderer::renderer::Scene;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use glam::{Mat4, Vec3, Vec4};
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
    scene: Option<Scene>,
    tint_buffer: Option<Arc<parking_lot::Mutex<StorageBuffer<Vec4>>>>,
    renderer: Option<Renderer>,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            tint_buffer: None,
            renderer: None,
            scene: None,
            render_commands: Vec::new(),
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Textured Cube")
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

                // Create a cube mesh
                let mut cube = Mesh::create_cube();
                // Override vertex colors to WHITE so they don't affect the material color
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }
                // RENAMING is critical because the renderer caches meshes by name!
                cube.name = Arc::from("TexturedCube");
                // Clear texture data so material color shows through
                cube.texture_data = None;

                // Set up material
                let material = Material {
                    color: [0.8, 0.2, 0.2, 1.0],
                    metallic: 0.5,
                    roughness: 0.5,
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
                        mesh: &mut cube,
                        asset_manager: &mut renderer.resources.assets,
                        staging_resources: &mut staging_resources,
                        material_override: None,
                    })
                    .unwrap();

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
                log::info!("✓ Mesh uploaded to GPU");

                // Register and upload material
                let material_handle = scene.register_material(&material).unwrap();

                log::info!("✓ Uploaded red material to GPU with handle {material_handle:?}");

                // Setup initial render command
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: Mat4::IDENTITY,
                        ..Default::default()
                    });

                // CRITICAL: Set lighting for visibility (RAGE approach)
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::IndoorLit)
                    .with_directional(
                        Vec3::new(1.0, -1.0, -1.0).normalize(),
                        Vec3::splat(2.0),
                        1.0,
                    )
                    .build();

                scene.set_lighting(lighting);

                // Register bindless storage buffer (prevents crash)
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                if let Ok((tint_buffer, _tint_index)) =
                    renderer.resources.register_bindless_storage_buffer(
                        &renderer.context,
                        &tint_colors,
                        "DefaultTint",
                    )
                {
                    self.tint_buffer = Some(tint_buffer);
                    log::info!("✓ Registered default tint buffer");
                }

                // CRITICAL: Must call enable_post_processing() to initialize HDR/Tonemapping pipelines!
                if let Err(e) = renderer.enable_post_processing(&mut scene) {
                    log::warn!("Post-processing failed: {e}");
                }

                self.renderer = Some(renderer);
                self.scene = Some(scene);
                self.window = Some(window);
                self.start_time = Instant::now();
                log::info!("Cube renderer initialized!");
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
                if let (Some(_renderer), Some(window)) = (&mut self.renderer, &self.window) {
                    // Application-side camera control (instead of auto_rotate)
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Orbiting camera around the origin
                    let radius = 5.0;
                    let camera_x = radius * elapsed.sin();
                    let camera_z = radius * elapsed.cos();
                    let camera_pos = Vec3::new(camera_x, 2.0, camera_z);
                    let target = Vec3::ZERO;
                    let up = Vec3::Y;

                    let view = Mat4::look_at_rh(camera_pos, target, up);
                    // MODERN PROJECTION: Infinite Reverse Z
                    let mut proj = Mat4::perspective_infinite_reverse_rh(
                        45.0_f32.to_radians(),
                        aspect,
                        0.5, // Near Plane
                    );
                    proj.y_axis.y *= -1.0; // Vulkan Y-flip

                    // Submit commands
                    if let (Some(renderer), Some(scene)) = (&mut self.renderer, &mut self.scene) {
                        if let Err(e) =
                            renderer.submit_render_commands(scene, &self.render_commands)
                        {
                            log::error!("Failed to submit render commands: {e}");
                        }

                        if let Err(e) =
                            renderer.render_frame(scene, view, proj, camera_pos, None, None)
                        {
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

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");

    Ok(())
}
