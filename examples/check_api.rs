use ash_renderer::renderer::Renderer;
use glam::Mat4;

fn main() {
    // Mock renderer for compilation check - just checking API existence types
    // We don't actually run this, just compile it.

    // We can't easily instantiate Renderer without Vulkan context, so we just check methods on a hypothetical instance if we could.
    // However, to satisfy the compiler we need an instance.
    // Instead, let's just make a dummy function that takes a &mut Renderer.

    #[allow(dead_code)]
    fn check_api(renderer: &mut Renderer) {
        let matrices: Vec<Mat4> = vec![Mat4::IDENTITY; 10];
        unsafe {
            let _ = renderer.update_joint_ssbo(&matrices);
            // Also check offset method
            let _ = renderer.update_joint_ssbo_offset(&matrices, 0);
        }

        // The user report says this takes 3 args: mesh_handle, material_handle, transform
        // Now it takes 4: mesh_handle, material_handle, transform, joint_offset
        renderer.draw_skinned_mesh(0, 0, Mat4::IDENTITY, 0);

        // Check RenderCommand fields
        // Since fields are public, this struct init checks their existence.
        let _ = ash_renderer::renderer::RenderCommand {
            mesh_handle: 0,
            material_handle: 0,
            transform: Mat4::IDENTITY,
            is_skinned: false,
            joint_offset: 0,
        };
    }
}
