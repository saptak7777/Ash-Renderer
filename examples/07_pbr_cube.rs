//! Rotating PBR Cube example.
//!
//! Demonstrates physically-based rendering with a rotating cube.
//! Features: PBR materials, dynamic lighting, HDR post-processing.

use ash_renderer::prelude::*;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::features::PointLight;
use ash_renderer::renderer::resources::uniform::StorageBuffer;
use ash_renderer::renderer::DebugMode;
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

        match Renderer::new(&surface_provider) {
            Ok(mut renderer) => {
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
                    metallic: 0.9,               // Highly metallic (but not perfect mirror)
                    roughness: 0.2,              // Smooth but with some blurring
                    ..Default::default()
                };

                // Upload mesh
                let mesh_handle = renderer.upload_mesh_single(cube).unwrap();
                log::info!("✓ Mesh uploaded to GPU");

                // Register and upload material
                let material_handle = renderer.register_and_upload_material(material).unwrap();

                log::info!("✓ Uploaded red PBR material to GPU with handle {material_handle:?}");

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
                    renderer.register_bindless_storage_buffer(&tint_colors, "DefaultTint")
                {
                    self.tint_buffer = Some(tint_buffer);
                    log::info!("✓ Registered default tint buffer");
                }

                // CRITICAL: Must call enable_post_processing() to initialize HDR/Tonemapping pipelines!
                if let Err(e) = renderer.enable_post_processing() {
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

                renderer.set_lighting(&lighting);

                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start_time = Instant::now();
                log::info!("PBR Cube renderer initialized!");
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
                            intensity: 5.0,
                            radius: 10.0,
                        },
                        PointLight {
                            position: Vec3::new(
                                light_distance * (elapsed * 2.0 + std::f32::consts::PI).cos(),
                                -1.0,
                                light_distance * (elapsed * 2.0 + std::f32::consts::PI).sin(),
                            ),
                            color: Vec3::new(1.0, 0.5, 0.5), // Pale red light
                            intensity: 8.0,
                            radius: 10.0,
                        },
                    ];

                    renderer.update_lights(&lights, &[]);

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
                    if let Err(e) = renderer.submit_render_commands(&self.render_commands) {
                        log::error!("Failed to submit render commands: {e}");
                    }

                    if let Err(e) = renderer.render_frame(view, proj, camera_pos, None) {
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

    let mut renderer = Renderer::new(&surface_provider)?;

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
    let mesh_handle = renderer.upload_mesh_single(cube).unwrap();
    let material_handle = renderer.register_and_upload_material(material).unwrap();

    let render_commands = vec![ash_renderer::renderer::RenderCommand {
        mesh_handle,
        material_handle,
        transform: Mat4::IDENTITY,
        ..Default::default()
    }];

    let _tint_buffer = {
        let tint_colors = [Vec4::new(1.0, 1.0, 1.0, 1.0)];
        renderer
            .register_bindless_storage_buffer(&tint_colors, "DefaultTintHeadless")
            .ok()
            .map(|(b, _)| b)
    };

    if let Err(e) = renderer.enable_post_processing() {
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
    renderer.set_lighting(&lighting);
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
        renderer.update_lights(&lights, &[]);

        let camera_pos = Vec3::new(0.0, 2.0, 5.0);
        let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
        let mut proj = Mat4::perspective_infinite_reverse_rh(45.0_f32.to_radians(), aspect, 0.1);
        proj.y_axis.y *= -1.0;

        renderer.submit_render_commands(&frame_commands)?;
        renderer.render_frame(view, proj, camera_pos, None)?;

        if frame % 100 == 0 {
            log::info!("Headless frame {frame}/{max_frames}");
        }
    }

    log::info!("Headless run complete");
    Ok(())
}
