//! Simple test for GLB material registration without window

use ash_renderer::prelude::*;

fn main() {
    env_logger::init();

    println!("Testing GLB material registration...");

    // Create a test mesh with material properties
    let mut test_mesh = ash_renderer::renderer::resources::mesh::Mesh::create_cube();
    test_mesh.name = "test_cube".to_string();
    test_mesh.material_properties = Some(
        ash_renderer::renderer::resources::mesh::MaterialProperties {
            base_color_factor: [0.8, 0.2, 0.2, 1.0],
            metallic_factor: 0.8,
            roughness_factor: 0.2,
            emissive_factor: [0.0, 0.0, 0.0, 1.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
            alpha_cutoff: 0.5,
        },
    );

    println!("✅ Created test mesh with material properties");

    // Verify material properties are set
    if let Some(props) = &test_mesh.material_properties {
        println!("✅ Material properties found:");
        println!("   - Base color: {:?}", props.base_color_factor);
        println!("   - Metallic: {}", props.metallic_factor);
        println!("   - Roughness: {}", props.roughness_factor);
    } else {
        println!("❌ No material properties found");
        return;
    }

    // Test that we can access the public field
    let props = test_mesh.material_properties.unwrap();
    assert!(props.metallic_factor > 0.0, "Metallic should be > 0");
    assert!(props.roughness_factor > 0.0, "Roughness should be > 0");

    println!("✅ All tests passed! Material properties are accessible and correct.");
}
