#![allow(deprecated)]
use ash_renderer::renderer::Scene;
use ash_renderer::{
    renderer::{Mesh, Renderer},
    vulkan::WindowSurfaceProvider,
    Result,
};
use std::sync::Arc;
use winit::{event_loop::EventLoop, window::Window};

fn main() -> Result<()> {
    // Initialize logger at DEBUG level to see VRAM stats
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let event_loop = EventLoop::builder().build().unwrap();
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes())
            .unwrap(),
    );

    let surface_provider = WindowSurfaceProvider::new(&window);

    log::info!("Initializing renderer for VRAM stress test...");
    let mut renderer = Renderer::new(&surface_provider)?;
    let mut scene = Scene::new(
        Arc::clone(&renderer.context.device.device),
        Arc::clone(&renderer.context.alloc),
        renderer.geometry_buffer(),
    )?;

    // Create a 2048x2048 synthetic texture (16MB)
    let texture_size = 2048 * 2048 * 4;
    let texture_data = vec![128u8; texture_size];

    log::info!("Starting stress test: Loading 100 high-res meshes...");

    for i in 0..100 {
        let descriptor = ash_renderer::renderer::resources::mesh::MeshDescriptor {
            key: format!("StressMesh_{i}").into(),
            vertices: vec![], // Empty mesh to focus on texture VRAM
            indices: None,
            texture: Some(
                ash_renderer::renderer::resources::texture::TextureData::new(
                    2048,
                    2048,
                    texture_data.clone(),
                )
                .unwrap(),
            ),
            normal_texture: None,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            material_properties: None,
        };

        log::info!("Registering mesh {i}...");
        let mut mesh = Mesh::from_descriptor(&descriptor);
        let upload_cmd = renderer.get_transfer_command_buffer().unwrap();
        let cmd_context = renderer.frame.cmds.context(upload_cmd);
        cmd_context
            .begin(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .unwrap();

        let mut staging_resources = Vec::new();
        let upload_res = scene.upload_mesh(
            Arc::clone(&renderer.context.device.device),
            Arc::clone(&renderer.context.alloc),
            renderer.frame.cmds.upload_command_pool_handle(),
            upload_cmd,
            &renderer.context.device.graphics_queue,
            &mut mesh,
            &mut renderer.resources.assets,
            &mut staging_resources,
            None,
        );

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

        if let Err(e) = upload_res {
            log::error!("Failed to register mesh {i}: {e}");
            break;
        }

        // Log stats after each load
    }

    log::info!("Stress test complete. Check logs for 'Using fallback' and VRAM percentages.");

    Ok(())
}
