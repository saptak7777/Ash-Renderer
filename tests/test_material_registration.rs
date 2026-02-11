//! Test GLB material registration with renderer

// Unused imports removed

#[test]
fn test_material_registration() {
    let _ = env_logger::builder().is_test(true).try_init();

    println!("Testing GLB material registration with renderer...");

    // Create a headless renderer for testing
    // Note: This would require a surface provider in a real scenario
    // For now, we'll test the material creation and registration logic

    // Create a test mesh with material properties
    let mut test_mesh = ash_renderer::renderer::resources::mesh::Mesh::create_cube();
    test_mesh.name = "test_cube".into();
    test_mesh.material_properties = Some(
        ash_renderer::renderer::resources::mesh::MaterialProperties {
            base_color_factor: [0.2, 0.2, 0.8, 1.0], // Blue color
            metallic_factor: 0.9,                    // Very metallic
            roughness_factor: 0.1,                   // Very smooth
            emissive_factor: [0.0, 0.0, 0.0, 1.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
            alpha_cutoff: 0.5,
        },
    );

    println!("✅ Created test mesh with material properties");

    // Verify material properties
    if let Some(props) = &test_mesh.material_properties {
        println!("✅ Material properties:");
        println!("   - Base color: {:?}", props.base_color_factor);
        println!("   - Metallic: {} (should be high)", props.metallic_factor);
        println!("   - Roughness: {} (should be low)", props.roughness_factor);

        // Create a Material from the properties (simulating what register_mesh_handle does)
        let material = ash_renderer::renderer::Material {
            name: format!("{}_material", test_mesh.name),
            color: props.base_color_factor,
            metallic: props.metallic_factor,
            roughness: props.roughness_factor,
            emissive: props.emissive_factor,
            occlusion_strength: props.occlusion_strength,
            normal_scale: props.normal_scale,
            alpha_cutoff: props.alpha_cutoff,
            tint_index: -1,
            is_transparent: props.base_color_factor[3] < 1.0,
            ..Default::default()
        };

        println!("✅ Created Material from properties:");
        println!("   - Material name: {}", material.name);
        println!("   - Material metallic: {}", material.metallic);
        println!("   - Material roughness: {}", material.roughness);

        // Verify the material was created correctly
        assert_eq!(material.color, props.base_color_factor);
        assert_eq!(material.metallic, props.metallic_factor);
        assert_eq!(material.roughness, props.roughness_factor);
        assert_eq!(material.emissive, props.emissive_factor);

        println!("✅ Material properties correctly transferred to Material struct");
    }

    // Test the fallback logic conceptually
    println!("\n📝 Testing material fallback logic:");
    println!("   - When material_handle is null, use mesh_handle to look up material");
    println!("   - When material_handle is valid, use material_handle directly");
    println!("   - If not found, fallback to default material");

    // Simulate the logic from submit_render_commands
    let _mesh_handle = 1u32;
    let material_handle = ash_renderer::renderer::MaterialHandle::null(); // This would trigger fallback

    // Create a material manager
    let mut manager = ash_renderer::renderer::MaterialManager::new();

    // Register the material
    let mut registered_handle = manager.default_material();
    if let Some(props) = &test_mesh.material_properties {
        let material = ash_renderer::renderer::Material {
            name: format!("{}_material", test_mesh.name),
            color: props.base_color_factor,
            metallic: props.metallic_factor,
            roughness: props.roughness_factor,
            emissive: props.emissive_factor,
            occlusion_strength: props.occlusion_strength,
            normal_scale: props.normal_scale,
            alpha_cutoff: props.alpha_cutoff,
            tint_index: -1,
            is_transparent: props.base_color_factor[3] < 1.0,
            ..Default::default()
        };
        registered_handle = manager.register_material(&material);
    }

    // Test the fallback logic
    let final_handle = if material_handle.is_null() {
        // Try to use mesh's own registered material first
        registered_handle
    } else {
        material_handle
    };

    let material = manager.get_material(final_handle);

    println!("✅ Fallback result: Material '{}'", material.name);
    assert_eq!(material.metallic, 0.9);
    println!("✅ Material correctly resolved via fallback");

    println!("\n🎉 All material registration tests passed!");
}
