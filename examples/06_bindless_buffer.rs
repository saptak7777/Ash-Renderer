//! Bindless Storage Buffer verification example.
//!
//! Demonstrates how to use Binding 2 (Storage Buffers) in the BindlessManager
//! to provide per-object or per-material configuration (like tints) without
//! using standard uniforms.
use ash::vk;
use ash_renderer::prelude::*;
use ash_renderer::renderer::Scene;
use ash_renderer::renderer::features::ambient_lighting::LightingBuilder;
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
    tint_buffer: Option<Arc<parking_lot::Mutex<StorageBuffer<Vec4>>>>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    _start_time: Instant,
    frame_count: u32,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: None,
            _start_time: Instant::now(),
            tint_buffer: None,
            frame_count: 0,
            render_commands: Vec::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Bindless Buffer Test")
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

                // 1. Register bindless storage buffer FIRST to get the index
                let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
                let (tint_buffer_gpu, tint_index) = renderer
                    .resources
                    .register_bindless_storage_buffer(
                        &renderer.context,
                        &tint_colors,
                        "CubeTintBuffer",
                    )
                    .expect("Failed to register bindless storage buffer");

                log::info!("✓ Registered bindless tint buffer at index {tint_index}");

                // 2. Set up material with MATTE ORANGE color (Phase 3)
                // Use the tint_index we just got!
                let material = Material {
                    color: [1.0, 0.5, 0.0, 1.0], // SOLID ORANGE
                    metallic: 0.0,               // Non-metallic
                    roughness: 0.7,              // Matte finish
                    tint_index: tint_index as i32,
                    ..Default::default()
                };
                log::info!("✓ Material set to MATTE ORANGE [Metallic 0.0, Roughness 0.7]");

                // Register and upload material
                let material_handle = scene.register_material(&material).unwrap();
                log::info!("✓ Registered orange material with handle {material_handle:?}");

                // 3. Create a cube
                let mut cube = Mesh::create_cube();
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }
                cube.name = Arc::from("OrangeCube");
                cube.texture_data = None;
                log::info!("✓ Cube mesh created and renamed to 'OrangeCube' for Phase 1");

                // Upload mesh
                let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                let cmd_context = renderer.frame.cmds.context(upload_cmd);
                cmd_context
                    .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
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
                    .unwrap_or(0);

                cmd_context.end().unwrap();

                // Submit and wait
                let cmds = [upload_cmd];
                let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
                unsafe {
                    renderer
                        .context
                        .device
                        .device
                        .queue_submit(
                            renderer.context.device.graphics_queue,
                            &[submit_info],
                            vk::Fence::null(),
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

                // 4. Setup render command
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: Mat4::IDENTITY,
                        ..Default::default()
                    });

                // 6. Setup PHASE 2 Lighting: Balanced HDR (RAGE approach)
                let lighting = LightingBuilder::new()
                    .with_directional(
                        Vec3::new(-1.0, -1.0, -1.0).normalize(),
                        Vec3::splat(2.5),
                        1.0,
                    )
                    .build();

                scene.set_lighting(lighting);

                self.renderer = Some(renderer);
                self.scene = Some(scene);
                self.window = Some(window);
                self.tint_buffer = Some(tint_buffer_gpu);
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
                    let time = self._start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        let aspect = size.width as f32 / size.height as f32;
                        let radius = 5.0;
                        let camera_x = radius * time.sin();
                        let camera_z = radius * time.cos();
                        let camera_pos = Vec3::new(camera_x, 2.0, camera_z);

                        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                        let mut proj = Mat4::perspective_infinite_reverse_rh(
                            45.0_f32.to_radians(),
                            aspect,
                            0.1, // Near Plane
                        );
                        proj.y_axis.y *= -1.0; // Vulkan Y-flip
                        // Dynamic Lighting
                        let light_angle = time * 0.5;
                        let light_dir =
                            Vec3::new(-light_angle.cos(), -1.0, -light_angle.sin()).normalize();

                        let lighting = LightingBuilder::new()
                            .with_directional(light_dir, Vec3::splat(2.5), 1.0)
                            .build();

                        scene.set_lighting(lighting);

                        // Submit commands
                        // Submit commands
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
                        self.frame_count += 1;
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => (),
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let is_headless = args.iter().any(|arg| arg == "--headless");
    let max_frames = args
        .iter()
        .position(|arg| arg == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|f| f.parse::<u32>().ok())
        .unwrap_or(u32::MAX);

    if is_headless {
        run_headless(max_frames)
    } else {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App::default();
        event_loop.run_app(&mut app).expect("Event loop error");
        Ok(())
    }
}

fn run_headless(max_frames: u32) -> Result<()> {
    log::info!("Running in HEADLESS mode for {max_frames} frames");
    let width = 1280;
    let height = 720;
    let surface_provider = ash_renderer::vulkan::HeadlessSurfaceProvider::new(width, height);

    let mut renderer = Renderer::new(&surface_provider)?;
    let mut scene = Scene::new(
        Arc::clone(&renderer.context.device.device),
        Arc::clone(&renderer.context.alloc),
        renderer.geometry_buffer(),
    )?;

    // --- SETUP SOURCE (Copied from resumed) ---
    // 1. Register bindless storage buffer FIRST to get the index
    let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
    let (_tint_buffer_gpu, tint_index) = renderer
        .resources
        .register_bindless_storage_buffer(&renderer.context, &tint_colors, "CubeTintBuffer")
        .expect("Failed to register bindless storage buffer");

    log::info!("✓ Registered bindless tint buffer at index {tint_index}");

    // 2. Set up material with MATTE ORANGE color
    let material = Material {
        color: [1.0, 0.5, 0.0, 1.0], // SOLID ORANGE
        metallic: 0.0,
        roughness: 0.7,
        tint_index: tint_index as i32,
        ..Default::default()
    };

    let material_handle = scene.register_material(&material).unwrap();
    let mut cube = Mesh::create_cube();
    for v in &mut cube.vertices {
        v.color = [1.0, 1.0, 1.0];
    }
    cube.name = Arc::from("OrangeCubeHeadless");
    let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
    let cmd_context = renderer.frame.cmds.context(upload_cmd);
    cmd_context
        .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
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
    let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
    unsafe {
        renderer
            .context
            .device
            .device
            .queue_submit(
                renderer.context.device.graphics_queue,
                &[submit_info],
                vk::Fence::null(),
            )
            .unwrap();
        renderer
            .context
            .device
            .device
            .queue_wait_idle(renderer.context.device.graphics_queue)
            .unwrap();
    }

    let mesh_handle = upload_res.unwrap_or(0);

    let render_commands = vec![ash_renderer::renderer::RenderCommand {
        mesh_handle,
        material_handle,
        transform: Mat4::IDENTITY,
        ..Default::default()
    }];

    let lighting = LightingBuilder::new()
        .with_directional(
            Vec3::new(-1.0, -1.0, -1.0).normalize(),
            Vec3::splat(2.5),
            1.0,
        )
        .build();
    scene.set_lighting(lighting);
    // --- END SETUP ---

    let start_time = Instant::now();
    let aspect = width as f32 / height as f32;

    for frame in 0..max_frames {
        let time = start_time.elapsed().as_secs_f32();

        // Orbiting camera
        let radius = 5.0;
        let camera_pos = Vec3::new(radius * time.sin(), 2.0, radius * time.cos());
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let mut proj = Mat4::perspective_infinite_reverse_rh(45.0_f32.to_radians(), aspect, 0.1);
        proj.y_axis.y *= -1.0;

        // Dynamic Lighting
        let light_angle = time * 0.5;
        let light_dir = Vec3::new(-light_angle.cos(), -1.0, -light_angle.sin()).normalize();
        let lighting = LightingBuilder::new()
            .with_directional(light_dir, Vec3::splat(2.5), 1.0)
            .build();
        scene.set_lighting(lighting);

        renderer.submit_render_commands(&mut scene, &render_commands)?;
        renderer.render_frame(&mut scene, view, proj, camera_pos, None, None)?;

        if frame % 100 == 0 {
            log::info!("Headless frame {frame}/{max_frames}");
        }
    }

    log::info!("Headless run complete");
    Ok(())
}
