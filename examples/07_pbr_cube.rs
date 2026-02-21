//! Rotating PBR Cube example.
//!
//! Demonstrates physically-based rendering with a rotating cube.
//! Features: PBR materials, dynamic lighting, HDR post-processing.

use ash::vk;
use ash_renderer::prelude::*;
use ash_renderer::renderer::DebugMode;
use ash_renderer::renderer::Scene;
use ash_renderer::renderer::features::PointLight;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use glam::{Mat4, Quat, Vec3, Vec4};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    scene: Option<Scene>,
    tint_buffer: Option<Arc<parking_lot::Mutex<StorageBuffer<Vec4>>>>,
    renderer: Option<Renderer>,
    render_commands: Vec<ash_renderer::renderer::RenderCommand>,
    start_time: Instant,
    current_debug_mode: DebugMode,
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
            current_debug_mode: DebugMode::None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - Rotating PBR Cube")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        let renderer = Renderer::builder()
            .with_vsync(true)
            .with_shadow_resolution(2048) // Lower shadow res for better perf in example
            .with_environment_map("assets/textures/skybox.hdr")
            .build(&surface_provider);

        match renderer {
            Ok(mut renderer) => {
                let mut scene = Scene::new(
                    Arc::clone(&renderer.context.device.device),
                    Arc::clone(&renderer.context.alloc),
                    renderer.geometry_buffer(),
                )
                .expect("Failed to create scene");

                // CRITICAL: Must assign global buffers BEFORE uploading meshes/materials
                println!("Initializing Scene Buffers...");
                scene.global_cluster_buffer = renderer.resources.global_cluster_buffer.clone();
                scene.material_storage_buffer = renderer.resources.material_storage_buffer.clone();
                println!("✓ Scene buffers synchronized with renderer.");

                // Create a cube mesh
                let mut cube = Mesh::create_cube();
                // Override vertex colors to WHITE so they don't affect the material color
                for v in &mut cube.vertices {
                    v.color = [1.0, 1.0, 1.0];
                }
                // RENAMING is critical because the renderer caches meshes by name!
                cube.name = Arc::from("RedPbrCube");
                cube.texture_data = None;

                // Set up a shiny red PBR material
                let material = Material {
                    name: "ShinyRed".to_string(),
                    color: [1.0, 0.0, 0.0, 1.0], // Pure red
                    metallic: 1.0,               // Fully metallic (Colored reflections)
                    roughness: 0.3,              // Softer highlights (less "plastic" look)
                    ..Default::default()
                };

                // Register and upload material
                let material_handle = scene.register_material(&material).unwrap();
                println!("✓ Uploaded red PBR material to GPU with handle {material_handle:?}");

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
                        material_override: Some(material_handle),
                    })
                    .unwrap();

                cmd_context.end().unwrap();

                // Submit and wait (for simplicity in example)
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
                    println!("Queue submit ok");
                    renderer
                        .context
                        .device
                        .device
                        .queue_wait_idle(renderer.context.device.graphics_queue)
                        .unwrap();
                    println!("Queue wait idle ok");
                }
                println!("✓ Mesh uploaded to GPU");

                // Setup the initial render command
                self.render_commands
                    .push(ash_renderer::renderer::RenderCommand {
                        mesh_handle,
                        material_handle,
                        transform: Mat4::IDENTITY,
                        ..Default::default()
                    });

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
                    println!("✓ Registered default tint buffer");
                }

                // CRITICAL: Must call enable_post_processing() to initialize HDR/Tonemapping pipelines!
                if let Err(e) = renderer.enable_post_processing(&mut scene) {
                    log::warn!("Post-processing failed: {e}");
                    // renderer.set_debug_mode(DebugMode::None);
                }

                // CRITICAL: Set lighting to highlight the PBR properties
                // RAGE Hemisphere Ambient + Global Directional Light
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::IndoorLit)
                    .with_directional(
                        Vec3::new(1.0, -1.0, -1.0).normalize(),
                        Vec3::splat(2.0),
                        1.0,
                    )
                    .build();

                scene.set_lighting(lighting);

                self.renderer = Some(renderer);
                self.scene = Some(scene);
                self.window = Some(window);
                self.start_time = Instant::now();
                println!("PBR Cube renderer initialized!");
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
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Rotate the cube itself
                    self.render_commands[0].transform = Mat4::from_rotation_translation(
                        Quat::from_euler(glam::EulerRot::XYZ, elapsed * 0.5, elapsed * 1.2, 0.0),
                        Vec3::ZERO,
                    );

                    // Add some orbiting point lights to showcase PBR shine
                    let light_distance = 3.0;
                    let lights = vec![
                        PointLight {
                            position: Vec3::new(
                                light_distance * (elapsed * 1.5).cos(),
                                1.0,
                                light_distance * (elapsed * 1.5).sin(),
                            ),
                            color: Vec3::new(1.0, 1.0, 1.0), // White light
                            intensity: 3.0,                  // Reduced from 5.0 to prevent blowout
                            radius: 10.0,
                        },
                        PointLight {
                            position: Vec3::new(
                                light_distance * (elapsed * 2.0 + std::f32::consts::PI).cos(),
                                -1.0,
                                light_distance * (elapsed * 2.0 + std::f32::consts::PI).sin(),
                            ),
                            color: Vec3::new(1.0, 0.5, 0.5), // Pale red light
                            intensity: 4.0,                  // Reduced from 8.0 to prevent blowout
                            radius: 10.0,
                        },
                    ];

                    scene.point_lights = lights;

                    // Fixed camera looking at the rotating cube
                    let camera_pos = Vec3::new(0.0, 2.0, 5.0);
                    let target = Vec3::ZERO;
                    let up = Vec3::Y;

                    let view = Mat4::look_at_rh(camera_pos, target, up);
                    // 1. MODERN PROJECTION: Infinite Reverse Z
                    // Maps Z_Near (0.1) -> 1.0
                    // Maps Infinity     -> 0.0
                    // This matches our pipeline's GREATER_OR_EQUAL test perfectly.
                    let mut proj = Mat4::perspective_infinite_reverse_rh(
                        45.0_f32.to_radians(),
                        aspect,
                        0.1, // Near Plane
                    );

                    // 2. VULKAN FLIP
                    // Glam assumes Y-Up (OpenGL standard). Vulkan uses Y-Down.
                    proj.y_axis.y *= -1.0;

                    // Submit render commands
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyD),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Cycle through debug modes
                self.current_debug_mode = match self.current_debug_mode {
                    DebugMode::None => DebugMode::Albedo,
                    DebugMode::Albedo => DebugMode::Normal,
                    DebugMode::Normal => DebugMode::Metallic,
                    DebugMode::Metallic => DebugMode::Roughness,
                    DebugMode::Roughness => DebugMode::Lighting,
                    DebugMode::Lighting => DebugMode::None,
                };
                log::info!("Debug mode: {:?}", self.current_debug_mode);
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_debug_mode(self.current_debug_mode);
                }
            }
            _ => {}
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

    let mut renderer = Renderer::builder()
        .with_vsync(false) // No vsync for headless profiling
        .with_environment_map("assets/textures/skybox.hdr")
        .build(&surface_provider)?;
    let mut scene = Scene::new(
        Arc::clone(&renderer.context.device.device),
        Arc::clone(&renderer.context.alloc),
        renderer.geometry_buffer(),
    )?;

    // --- SETUP SOURCE (Copied from resumed) ---
    let mut cube = Mesh::create_cube();
    for v in &mut cube.vertices {
        v.color = [1.0, 1.0, 1.0];
    }
    cube.name = Arc::from("RedPbrCubeHeadless");
    let material = Material {
        name: "ShinyRedHeadless".to_string(),
        color: [1.0, 0.0, 0.0, 1.0],
        metallic: 0.9,
        roughness: 0.2,
        ..Default::default()
    };
    let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
    let mut staging_resources = Vec::new();
    let material_handle = scene.register_material(&material).unwrap();
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
            material_override: Some(material_handle),
        })
        .unwrap();

    let render_commands = vec![ash_renderer::renderer::RenderCommand {
        mesh_handle,
        material_handle,
        transform: Mat4::IDENTITY,
        ..Default::default()
    }];

    let _tint_buffer = {
        let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
        renderer
            .resources
            .register_bindless_storage_buffer(
                &renderer.context,
                &tint_colors,
                "DefaultTintHeadless",
            )
            .ok()
            .map(|(b, _)| b)
    };

    if let Err(e) = renderer.enable_post_processing(&mut scene) {
        log::warn!("Post-processing failed: {e}");
    }

    let lighting = LightingBuilder::new()
        .with_ambient_preset(AmbientPreset::IndoorLit)
        .with_directional(
            Vec3::new(1.0, -1.0, -1.0).normalize(),
            Vec3::splat(2.0),
            1.0,
        )
        .build();
    scene.set_lighting(lighting);
    // --- END SETUP ---

    let start_time = Instant::now();
    let aspect = width as f32 / height as f32;

    for frame in 0..max_frames {
        let elapsed = start_time.elapsed().as_secs_f32();

        // Rotate the cube
        let mut frame_commands = render_commands.clone();
        frame_commands[0].transform = Mat4::from_rotation_translation(
            Quat::from_euler(glam::EulerRot::XYZ, elapsed * 0.5, elapsed * 1.2, 0.0),
            Vec3::ZERO,
        );

        // Orbiting point lights
        let light_distance = 3.0;
        let lights = vec![
            PointLight {
                position: Vec3::new(
                    light_distance * (elapsed * 1.5).cos(),
                    1.0,
                    light_distance * (elapsed * 1.5).sin(),
                ),
                color: Vec3::new(1.0, 1.0, 1.0),
                intensity: 5.0,
                radius: 10.0,
            },
            PointLight {
                position: Vec3::new(
                    light_distance * (elapsed * 2.0 + std::f32::consts::PI).cos(),
                    -1.0,
                    light_distance * (elapsed * 2.0 + std::f32::consts::PI).sin(),
                ),
                color: Vec3::new(1.0, 0.5, 0.5),
                intensity: 8.0,
                radius: 10.0,
            },
        ];
        scene.point_lights = lights;

        let camera_pos = Vec3::new(0.0, 2.0, 5.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let mut proj = Mat4::perspective_infinite_reverse_rh(45.0_f32.to_radians(), aspect, 0.1);
        proj.y_axis.y *= -1.0;

        renderer.submit_render_commands(&mut scene, &frame_commands)?;
        renderer.render_frame(&mut scene, view, proj, camera_pos, None, None)?;

        if frame % 100 == 0 {
            log::info!("Headless frame {frame}/{max_frames}");
        }
    }

    log::info!("Headless run complete");
    Ok(())
}
