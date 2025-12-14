use ash_renderer::{
    renderer::{Mesh, Renderer},
    vulkan::WindowSurfaceProvider,
    Result,
};
use std::sync::Arc;
use winit::{event_loop::EventLoop, window::Window};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::builder().build().unwrap();
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes())
            .unwrap(),
    );

    // Use the window surface provider wrapper
    let surface_provider = WindowSurfaceProvider::new(&window);

    log::info!("Initializing renderer to test crash...");
    let mut renderer = Renderer::new(&surface_provider)?;

    // Create a large mesh to simulate the user's scenario
    // 200k vertices -> ~200,000 * 48 bytes ~= 9.6 MB
    log::info!("Generating large synthetic mesh...");
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..200_000 {
        vertices.push(ash_renderer::renderer::Vertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        indices.push(i);
        indices.push(i); // Dummy indices
        indices.push(i);
    }

    let descriptor = ash_renderer::renderer::resources::mesh::MeshDescriptor {
        key: "CrashTestMesh".to_string(),
        vertices,
        indices: Some(indices),
        texture: None,
        normal_texture: None,
        metallic_roughness_texture: None,
        occlusion_texture: None,
        emissive_texture: None,
        material_properties: None,
    };
    let mesh = Mesh::from_descriptor(&descriptor);

    // This is where it reportedly crashes
    log::info!("Attempting to set mesh (uploads to GPU)...");
    renderer.set_mesh(mesh);

    log::info!("Success! No crash encountered.");
    Ok(())
}
