//! VSM Debug Visualization example.
//!
//! Renders the VSM Page Table and Physical Atlas as an overlay.
//! Features: Custom UI callback for debug rendering, VSM internal state visualization.

use ash::vk;
use ash_renderer::prelude::*;
use ash_renderer::renderer::features::ambient_lighting::{AmbientPreset, LightingBuilder};
use ash_renderer::renderer::MeshUploadInfo;
use ash_renderer::renderer::RenderCommand;
use ash_renderer::renderer::Scene;
use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct DebugPipeline {
    pipeline: ash_renderer::vulkan::Pipeline,
    layout: vk::PipelineLayout,
    _descriptor_set_layout: vk::DescriptorSetLayout,
    _descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl DebugPipeline {
    fn new(renderer: &Renderer) -> Result<Self> {
        let device = &renderer.context.device.device;

        // 1. Descriptor Set Layout
        let bindings = [
            // Binding 0: Page Table (Storage Image Array)
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            // Binding 1: Physical Memory (Storage Image)
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        };

        // 2. Pipeline Layout
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: 8, // uint mode, float layer
        };

        let layouts = [descriptor_set_layout];
        let layouts_ref = &layouts;
        let push_ranges = [push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(layouts_ref)
            .push_constant_ranges(&push_ranges);

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() };

        // 3. Pipeline
        let swapchain = renderer.frame.swapchain.as_ref().unwrap();
        let mut builder = ash_renderer::vulkan::Pipeline::builder(Arc::clone(device))
            .with_layout(layout)
            .with_dynamic_rendering(&[swapchain.format], None, None)
            .with_extent(swapchain.extent)
            .with_cull_mode(vk::CullModeFlags::NONE);

        // Load debug shaders
        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/vsm_debug.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/vsm_debug.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

        let pipeline = builder.build()?;

        // 4. Descriptor Pool & Set
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 2,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

        // 5. Update Descriptor Set
        let vsm = &renderer.vsm_manager;
        let table_info = [vk::DescriptorImageInfo::default()
            .image_view(vsm.page_table_view()?)
            .image_layout(vk::ImageLayout::GENERAL)];
        let physical_info = [vk::DescriptorImageInfo::default()
            .image_view(vsm.physical_cache_view()?)
            .image_layout(vk::ImageLayout::GENERAL)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&table_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&physical_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]) };

        Ok(Self {
            pipeline,
            layout,
            _descriptor_set_layout: descriptor_set_layout,
            _descriptor_pool: descriptor_pool,
            descriptor_set,
        })
    }

    fn draw(&self, cmd: vk::CommandBuffer, device: &ash::Device, screen_extent: vk::Extent2D) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            // Draw Page Table (Bottom Left)
            let viewport_table = vk::Viewport {
                x: 10.0,
                y: screen_extent.height as f32 - 310.0,
                width: 300.0,
                height: 300.0,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport_table]);
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 10,
                        y: screen_extent.height as i32 - 310,
                    },
                    extent: vk::Extent2D {
                        width: 300,
                        height: 300,
                    },
                }],
            );

            let pc_table = [1u32, 0u32]; // Mode 1, Layer 0
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&pc_table),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);

            // Draw Physical Atlas (Bottom Right)
            let viewport_atlas = vk::Viewport {
                x: screen_extent.width as f32 - 310.0,
                y: screen_extent.height as f32 - 310.0,
                width: 300.0,
                height: 300.0,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport_atlas]);
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D {
                        x: screen_extent.width as i32 - 310,
                        y: screen_extent.height as i32 - 310,
                    },
                    extent: vk::Extent2D {
                        width: 300,
                        height: 300,
                    },
                }],
            );

            let pc_atlas = [0u32, 0u32]; // Mode 0, Layer 0
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&pc_atlas),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }
}

struct App {
    window: Option<Window>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    debug_pipeline: Option<DebugPipeline>,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: None,
            debug_pipeline: None,
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("ASH Renderer - VSM Debug Visualization")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

        let window = event_loop.create_window(window_attrs).unwrap();
        let surface_provider = ash_renderer::vulkan::WindowSurfaceProvider::new(&window);

        let renderer = Renderer::builder()
            .with_vsync(true)
            .build(&surface_provider);

        match renderer {
            Ok(mut renderer) => {
                let mut scene = Scene::new(
                    Arc::clone(&renderer.context.device.device),
                    Arc::clone(&renderer.context.alloc),
                    renderer.geometry_buffer(),
                )
                .expect("Failed to create scene");

                // Initialize Scene Buffers
                scene.global_cluster_buffer = renderer.resources.global_cluster_buffer.clone();
                scene.material_storage_buffer = renderer.resources.material_storage_buffer.clone();

                // Create debug pipeline
                let debug_pipeline = DebugPipeline::new(&renderer).unwrap();

                // Set up scene: Ground plane + cast cubes
                let mut plane =
                    ash_renderer::renderer::resources::Mesh::create_named_cube("Ground");
                for v in &mut plane.vertices {
                    v.color = [0.2, 0.2, 0.2];
                }

                let mut cube =
                    ash_renderer::renderer::resources::Mesh::create_named_cube("CastCube");
                for v in &mut cube.vertices {
                    v.color = [0.8, 0.1, 0.1];
                }

                // Upload logic... (omitted for brevity, assume standard upload)
                // Just for the sake of completeness in the actual file,
                // I'll add the upload logic and commands.

                let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
                let cmd_context = renderer.frame.cmds.context(upload_cmd);
                cmd_context
                    .begin(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .unwrap();

                let mut staging = Vec::new();
                let material = Material {
                    name: "Default".to_string(),
                    color: [1.0; 4],
                    ..Default::default()
                };
                let mat_handle = scene.register_material(&material).unwrap();

                let plane_handle = scene
                    .upload_mesh(MeshUploadInfo {
                        device: Arc::clone(&renderer.context.device.device),
                        allocator: Arc::clone(&renderer.context.alloc),
                        command_pool: renderer.frame.cmds.upload_command_pool_handle(),
                        command_buffer: upload_cmd,
                        queue: renderer.context.device.graphics_queue,
                        mesh: &mut plane,
                        asset_manager: &mut renderer.resources.assets,
                        staging_resources: &mut staging,
                        material_override: Some(mat_handle),
                    })
                    .unwrap();

                let cube_handle = scene
                    .upload_mesh(MeshUploadInfo {
                        device: Arc::clone(&renderer.context.device.device),
                        allocator: Arc::clone(&renderer.context.alloc),
                        command_pool: renderer.frame.cmds.upload_command_pool_handle(),
                        command_buffer: upload_cmd,
                        queue: renderer.context.device.graphics_queue,
                        mesh: &mut cube,
                        asset_manager: &mut renderer.resources.assets,
                        staging_resources: &mut staging,
                        material_override: Some(mat_handle),
                    })
                    .unwrap();

                cmd_context.end().unwrap();
                unsafe {
                    let cmd_bufs = [upload_cmd];
                    let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
                    renderer
                        .context
                        .device
                        .device
                        .queue_submit(
                            renderer.context.device.graphics_queue,
                            &[submit],
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

                // Add objects to scene
                let mut commands = Vec::new();
                // Ground
                commands.push(RenderCommand {
                    mesh_handle: plane_handle,
                    material_handle: mat_handle,
                    transform: Mat4::from_scale_rotation_translation(
                        Vec3::splat(50.0),
                        Quat::from_rotation_x(-1.57),
                        Vec3::new(0.0, -1.0, 0.0),
                    ),
                    ..Default::default()
                });
                // Floating Cubes
                for i in 0..5 {
                    commands.push(RenderCommand {
                        mesh_handle: cube_handle,
                        material_handle: mat_handle,
                        transform: Mat4::from_translation(Vec3::new(
                            i as f32 * 2.0 - 4.0,
                            1.0 + (i as f32 * 0.5),
                            0.0,
                        )),
                        ..Default::default()
                    });
                }

                // Lighting
                let lighting = LightingBuilder::new()
                    .with_ambient_preset(AmbientPreset::IndoorLit)
                    .with_directional(
                        Vec3::new(1.0, -2.0, -1.0).normalize(),
                        Vec3::splat(2.5),
                        1.0,
                    )
                    .build();
                scene.set_lighting(lighting);

                // Initialize post-processing
                renderer.enable_post_processing(&mut scene).unwrap();

                self.renderer = Some(renderer);
                self.scene = Some(scene);
                self.window = Some(window);
                self.debug_pipeline = Some(debug_pipeline);
                self.start_time = Instant::now();

                // Add commands to renderer persistent state if needed or just pass them in render loop
                // Actually 07_pbr_cube stores them in self.render_commands.
                // Let's just fix it in the RedrawRequested.
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
                if let (Some(renderer), Some(window), Some(scene), Some(debug_pipe)) = (
                    &mut self.renderer,
                    &self.window,
                    &mut self.scene,
                    &self.debug_pipeline,
                ) {
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    // Rebuild commands per frame for demo
                    // In this renderer, Scene stores MeshData which contains handles.
                    let mut render_commands = Vec::new();
                    if scene.mesh_data.len() >= 2 {
                        let plane_h = 0u32;
                        let cube_h = 1u32;
                        let mat_h = scene.mesh_data[0].material_handle;

                        render_commands.push(RenderCommand {
                            mesh_handle: plane_h,
                            material_handle: mat_h,
                            transform: Mat4::from_scale_rotation_translation(
                                Vec3::new(50.0, 1.0, 50.0),
                                Quat::from_rotation_x(-1.57),
                                Vec3::new(0.0, -1.0, 0.0),
                            ),
                            ..Default::default()
                        });

                        for i in 0..5 {
                            let t = elapsed + (i as f32);
                            render_commands.push(RenderCommand {
                                mesh_handle: cube_h,
                                material_handle: mat_h,
                                transform: Mat4::from_translation(Vec3::new(
                                    (t * 0.5).cos() * 3.0,
                                    1.0 + (i as f32 * 0.5),
                                    (t * 0.5).sin() * 3.0,
                                )),
                                ..Default::default()
                            });
                        }
                    }

                    // Camera
                    let camera_pos = Vec3::new(8.0, 8.0, 8.0);
                    let view = Mat4::look_at_rh(camera_pos, Vec3::ZERO, Vec3::Y);
                    let mut proj =
                        Mat4::perspective_infinite_reverse_rh(45.0_f32.to_radians(), aspect, 0.1);
                    proj.y_axis.y *= -1.0;

                    if let Err(e) = renderer.submit_render_commands(scene, &render_commands) {
                        log::error!("Submit error: {e}");
                    }
                    let screen_extent = vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    };

                    let device_clone = Arc::clone(&renderer.context.device.device);

                    // Execute frame with debug overlay
                    if let Err(e) = renderer.render_frame(
                        scene,
                        view,
                        proj,
                        camera_pos,
                        None,
                        Some(&|cmd| {
                            debug_pipe.draw(cmd, &device_clone, screen_extent);
                        }),
                    ) {
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
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Event loop error");
    Ok(())
}
