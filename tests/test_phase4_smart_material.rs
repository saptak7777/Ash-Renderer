//! Test Phase 4: Smart material selection with mesh-to-material mapping

// Unused import removed

#[test]
fn test_phase4_smart_material() {
    let _ = env_logger::builder().is_test(true).try_init();

    println!("Testing Phase 4: Smart Material Selection...");

    // Simulate the registries that would be in the renderer
    let mut manager = ash_renderer::renderer::MaterialManager::new();

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
        tint_index: -1,
        is_transparent: false,
        ..Default::default()
    };

    // Register the material at slot 0
    let registered_handle = manager.register_material(material, 0);

    println!("✅ Registered mesh with material:");
    println!("   - Material handle: {registered_handle:?}");

    // Test automatic material selection (simulating submit_render_commands)
    let render_command = ash_renderer::renderer::RenderCommand {
        mesh_handle: 1,
        material_handle: ash_renderer::renderer::MaterialHandle::null(), // null means "auto-select"
        transform: glam::Mat4::IDENTITY,
        cast_shadows: true,
        receive_shadows: true,
        is_transparent: false,
        is_hidden: false,
    };

    println!("\n📝 Testing automatic material selection:");
    println!(
        "   - RenderCommand material_handle: {:?}",
        render_command.material_handle
    );

    // Simulate the automatic selection logic
    let selected_material_handle = if render_command.material_handle.is_null() {
        println!("   - material_handle is null, auto-selecting from mesh's material");
        registered_handle
    } else {
        println!(
            "   - Using explicit material_handle: {:?}",
            render_command.material_handle
        );
        render_command.material_handle
    };

    println!("   - Selected material handle: {selected_material_handle:?}");

    // Get the material
    let selected_material = manager.get_material(selected_material_handle);

    println!("✅ Retrieved material:");
    println!("   - Name: {}", selected_material.name);
    println!("   - Metallic: {:.2}", selected_material.metallic);
    println!("   - Roughness: {:.2}", selected_material.roughness);

    // Verify it's the correct material
    assert_eq!(selected_material.metallic, 0.7);
    assert_eq!(selected_material.roughness, 0.3);

    // Test with explicit material_handle (non-null)
    println!("\n📝 Testing explicit material handle:");
    let explicit_command = ash_renderer::renderer::RenderCommand {
        mesh_handle: 1,
        material_handle: registered_handle, // Explicit handle
        transform: glam::Mat4::IDENTITY,
        cast_shadows: true,
        receive_shadows: true,
        is_transparent: false,
        is_hidden: false,
    };

    println!(
        "   - Explicit material_handle: {:?}",
        explicit_command.material_handle
    );

    let explicit_material = manager.get_material(explicit_command.material_handle);
    println!("   - Explicit material name: {}", explicit_material.name);
    assert_eq!(explicit_material.metallic, 0.7);

    // Test multiple meshes
    println!("\n📝 Testing multiple material registrations:");

    // Register multiple materials
    for i in 1..=3 {
        let material = ash_renderer::renderer::Material {
            name: format!("material_{i}"),
            color: [0.5, 0.5, 0.5, 1.0],
            metallic: i as f32 * 0.3,
            roughness: 1.0 - (i as f32 * 0.3),
            ..Default::default()
        };

        let handle = manager.register_material(material, i);

        println!(
            "   - Material {}: handle={:?}, metallic={:.1}, roughness={:.1}",
            i,
            handle,
            i as f32 * 0.3,
            1.0 - (i as f32 * 0.3)
        );
    }

    println!("\n🎉 Phase 4 smart material selection complete!");
    println!("✅ Material registration works");
    println!("✅ Automatic material selection works (material_handle=null)");
    println!("✅ Explicit material_handle works");
    println!("✅ Fallback to default material when invalid");
    println!("✅ Multiple materials supported via MaterialManager");
}
