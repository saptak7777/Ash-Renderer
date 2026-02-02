use ash_renderer::renderer::{MaterialHandle, Renderer};
use glam::Mat4;

fn main() {
    // Mock renderer for compilation check - just checking API existence types
    // We don't actually run this, just compile it.

    // We can't easily instantiate Renderer without Vulkan context, so we just check methods on a hypothetical instance if we could.
    // However, to satisfy the compiler we need an instance.
    // Instead, let's just make a dummy function that takes a &mut Renderer.

    #[allow(dead_code)]
    fn check_api(renderer: &mut Renderer) {
        // Basic mesh registration check
        let mut cube = ash_renderer::renderer::resources::Mesh::create_cube();
        let _ = renderer.register_mesh_handle_single(0, &mut cube);

        // Check RenderCommand fields
        let commands = vec![ash_renderer::renderer::RenderCommand {
            mesh_handle: 0,
            material_handle: MaterialHandle::null(),
            transform: Mat4::IDENTITY,
            cast_shadows: true,
            receive_shadows: true,
            is_transparent: false,
            is_hidden: false,
        }];

        let _ = renderer.submit_render_commands(&commands);
    }
}
