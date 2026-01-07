use ash_renderer::prelude::*;
use glam::{Mat4, Vec3, Vec4};

fn main() -> Result<()> {
    env_logger::init();
    
    // Create a simple test renderer
    let surface_provider = ash_renderer::vulkan::HeadlessSurfaceProvider::new(800, 600)?;
    let mut renderer = Renderer::new(&surface_provider)?;
    
    // Create a cube
    let cube = Mesh::create_cube();
    renderer.set_mesh(cube)?;
    
    // Set up material with white base color (for tinting)
    let material = Material {
        color: [1.0, 1.0, 1.0, 1.0], // White base
        metallic: 0.0,
        roughness: 0.5,
        ..Default::default()
    };
    *renderer.material_mut() = material;
    
    // Register bindless storage buffer with bright orange color
    let tint_colors = [Vec4::new(1.0, 0.5, 0.0, 1.0)]; // Bright orange
    let (tint_buffer_gpu, tint_index) = renderer
        .register_bindless_storage_buffer(&tint_colors, "DebugTintBuffer")?;
    
    log::info!("Registered bindless tint buffer at index {}", tint_index);
    
    // Update the material to use the tint
    renderer.material_mut().tint_index = tint_index as i32;
    log::info!("Set tint_index {} on renderer material", tint_index);
    
    // Verify material state
    log::info!("Material state: color={:?}, tint_index={}", 
               renderer.material().color, 
               renderer.material().tint_index);
    
    // Test rendering with different camera positions
    let camera_positions = [
        Vec3::new(3.0, 3.0, 3.0),
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(2.0, 1.0, 2.0),
    ];
    
    for (i, camera_pos) in camera_positions.iter().enumerate() {
        log::info!("Test render {}: camera at {:?}", i, camera_pos);
        
        let view = Mat4::look_at_rh(*camera_pos, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 800.0 / 600.0, 0.1, 100.0);
        
        renderer.render_frame(view, proj, *camera_pos)?;
        
        // Small delay to ensure GPU operations complete
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    log::info!("Tint buffer test completed successfully");
    Ok(())
}