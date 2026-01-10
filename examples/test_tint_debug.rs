use ash_renderer::prelude::*;
use glam::{Mat4, Vec3, Vec4};

fn main() -> Result<()> {
    env_logger::init();
    
    // Create a simple test renderer
    let surface_provider = ash_renderer::vulkan::HeadlessSurfaceProvider::new(800, 600);
    let mut renderer = Renderer::new(&surface_provider)?;
    
    // Create a cube
    let cube = Mesh::create_cube();
    renderer.set_mesh(cube)?;
    
    // Set up material with white base color (for tinting)
    let mut material = Material {
        color: [1.0, 1.0, 1.0, 1.0], // White base
        metallic: 0.0,
        roughness: 0.5,
        ..Default::default()
    };
    
    // Register bindless storage buffer with bright orange color
    let tint_colors = [Vec4::new(1.0, 0.5, 0.0, 1.0)]; // Bright orange
    let (_tint_buffer_gpu, tint_index) = renderer
        .register_bindless_storage_buffer(&tint_colors, "DebugTintBuffer")?;
    
    // Update the material to use the tint
    material.tint_index = tint_index as i32;
    
    // CRITICAL FIX: Register and upload material
    let material_handle = renderer.material_manager_mut().register_material(material.clone());
    renderer.upload_material_to_gpu(material_handle.index as u32, &material)?;
    
    // Refresh draw items to reflect the material change
    renderer.refresh_draw_items();
    
    // Test rendering with different camera positions
    let camera_positions = [
        Vec3::new(3.0, 3.0, 3.0),
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(2.0, 1.0, 2.0),
    ];
    
    for camera_pos in camera_positions.iter() {
        let view = Mat4::look_at_rh(*camera_pos, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 800.0 / 600.0, 0.1, 100.0);
        
        renderer.render_frame(view, proj, *camera_pos)?;
        
        // Small delay to ensure GPU operations complete
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    Ok(())
}