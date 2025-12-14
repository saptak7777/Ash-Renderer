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

    // User's mesh has 209668 vertices and 982380 indices
    let vertex_count = 209668;
    let index_count = 982380;

    // We push 3 indices per iteration. So we need 327460 iterations.
    for i in 0..327460 {
        vertices.push(ash_renderer::renderer::Vertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        });

        // This logic is just dummy data, indices don't need to be valid for buffer creation test
        indices.push(0);
        indices.push(1);
        indices.push(2);
    }

    // Ensure exact counts
    // We pushed 327460 * 1 vertex = 327460 vertices. User has 209668.
    // We pushed 327460 * 3 indices = 982380 indices. Correct.
    // Correct vertex buffer size isn't critical for alignment of *index* buffer,
    // unless they pack tightly.
    // User vertex buffer: 209668 * 48 = 10,064,064 bytes?
    // Wait, user log says "Creating buffer (12580080B)".
    // 12580080 / 48 = 262,085 vertices.
    // User log says "Vertices: 209668".
    // Maybe `Vertex` struct size is different?
    // Ash Renderer Vertex: pos(12) + normal(12) + uv(8) + color(12) + tangent(16)?
    // 12+12+8+12+16 = 60?
    // 12580080 / 60 = 209668. EXACTLY.
    // So Vertex size is 60 bytes.
    // My crash_repro uses `ash_renderer::renderer::Vertex`, so it should match.
    // I will adjust loop to match vertex count too.

    // Create dummy texture data (2048x2048 RGBA = 16MB)
    let texture_size = 2048 * 2048 * 4;
    let texture_data = vec![255u8; texture_size];

    let descriptor = ash_renderer::renderer::resources::mesh::MeshDescriptor {
        key: "CrashTestMesh".to_string(),
        vertices,
        indices: Some(indices),
        texture: Some(
            ash_renderer::renderer::resources::texture::TextureData::new(2048, 2048, texture_data)
                .unwrap(),
        ),
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
