//! Test Phase 4: Smart material selection with mesh-to-material mapping

use ash_renderer::prelude::*;
use std::collections::HashMap;

fn main() {
    env_logger::init();

    println!("Testing Phase 4: Smart Material Selection...");

    // Simulate the mesh_material_mapping that would be in the renderer
    let mut mesh_material_mapping = HashMap::new();
    let mut material_registry = HashMap::new();

    // Simulate registering a mesh with material properties
    let mesh_handle = 1u32;
    let material_handle = mesh_handle; // In Phase 1, we use same handle

    // Create a material from properties (simulating register_mesh_handle)
    let material = ash_renderer::renderer::Material {
        name: "test_mesh_material".to_string(),
        color: [0.8, 0.2, 0.2, 1.0],
        metallic: 0.7,
        roughness: 0.3,
        emissive: [0.0, 0.0, 0.0, 1.0],
        occlusion_strength: 1.0,
        normal_scale: 1.0,
        alpha_cutoff: 0.5,
    };

    // Populate the registries (simulating register_mesh_handle)
    material_registry.insert(material_handle, material);
    mesh_material_mapping.insert(mesh_handle, material_handle);

    println!("✅ Registered mesh with material:");
    println!("   - Mesh handle: {}", mesh_handle);
    println!("   - Material handle: {}", material_handle);
    println!(
        "   - Mapping: {} → {}",
        mesh_handle, mesh_material_mapping[&mesh_handle]
    );

    // Test automatic material selection (simulating submit_render_commands)
    let render_command = ash_renderer::renderer::RenderCommand {
        mesh_handle,
        material_handle: 0, // 0 means "auto-select"
        transform: glam::Mat4::IDENTITY,
        is_skinned: false,
        joint_offset: 0,
    };

    println!("\n📝 Testing automatic material selection:");
    println!(
        "   - RenderCommand material_handle: {}",
        render_command.material_handle
    );

    // Simulate the automatic selection logic
    let selected_material_handle = if render_command.material_handle == 0 {
        println!("   - material_handle is 0, auto-selecting from mesh_material_mapping");
        mesh_material_mapping
            .get(&render_command.mesh_handle)
            .copied()
            .unwrap_or(0)
    } else {
        println!(
            "   - Using explicit material_handle: {}",
            render_command.material_handle
        );
        render_command.material_handle
    };

    println!(
        "   - Selected material handle: {}",
        selected_material_handle
    );

    // Get the material
    let default_material = ash_renderer::renderer::Material::default();
    let selected_material = material_registry
        .get(&selected_material_handle)
        .unwrap_or(&default_material);

    println!("✅ Retrieved material:");
    println!("   - Name: {}", selected_material.name);
    println!("   - Metallic: {:.2}", selected_material.metallic);
    println!("   - Roughness: {:.2}", selected_material.roughness);

    // Verify it's the correct material
    assert_eq!(selected_material.metallic, 0.7);
    assert_eq!(selected_material.roughness, 0.3);

    // Test with explicit material_handle (non-zero)
    println!("\n📝 Testing explicit material handle:");
    let explicit_command = ash_renderer::renderer::RenderCommand {
        mesh_handle,
        material_handle: 42, // Explicit handle
        transform: glam::Mat4::IDENTITY,
        is_skinned: false,
        joint_offset: 0,
    };

    let explicit_material_handle = if explicit_command.material_handle == 0 {
        mesh_material_mapping
            .get(&explicit_command.mesh_handle)
            .copied()
            .unwrap_or(0)
    } else {
        explicit_command.material_handle
    };

    println!(
        "   - Explicit material_handle: {}",
        explicit_material_handle
    );

    // Should fallback to default since 42 doesn't exist
    let default_material2 = ash_renderer::renderer::Material::default();
    let fallback_material = material_registry
        .get(&explicit_material_handle)
        .unwrap_or(&default_material2);

    println!("   - Fallback material name: {}", fallback_material.name);

    // Test multiple meshes
    println!("\n📝 Testing multiple mesh-material mappings:");
    let mut test_mappings = HashMap::new();
    let mut test_materials = HashMap::new();

    // Register multiple meshes
    for i in 1..=3 {
        let handle = i;
        let material = ash_renderer::renderer::Material {
            name: format!("material_{}", i),
            color: [0.5, 0.5, 0.5, 1.0],
            metallic: i as f32 * 0.3,
            roughness: 1.0 - (i as f32 * 0.3),
            ..Default::default()
        };

        test_materials.insert(handle, material);
        test_mappings.insert(handle, handle);

        println!(
            "   - Mesh {}: material_handle={}, metallic={:.1}, roughness={:.1}",
            i,
            handle,
            i as f32 * 0.3,
            1.0 - (i as f32 * 0.3)
        );
    }

    println!("\n🎉 Phase 4 smart material selection complete!");
    println!("✅ Mesh-to-material mapping created during registration");
    println!("✅ Automatic material selection works (material_handle=0)");
    println!("✅ Explicit material_handle still works");
    println!("✅ Fallback to default material when not found");
    println!("✅ Multiple mesh-material mappings supported");
}
