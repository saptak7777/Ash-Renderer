use glam::{Mat4, Vec3};

/// Calculates a stable "snapped" orthographic view-projection matrix for a VSM clipmap level.
pub fn calculate_clipmap_view_proj(
    light_dir: Vec3,
    camera_center: Vec3,
    radius: f32,
    texture_size: u32,
) -> Mat4 {
    let light_dir_norm = light_dir.normalize();

    // 1. View: Create a view matrix looking in the light direction.
    // We use the camera center as the target and back up along the light direction.

    // 2. Project: Transform camera center into light space.
    // In this space, the camera center should be at X=0, Y=0 (since it's the target).
    // However, if we want stable shadows, we should snap the camera center's position
    // relative to the "world grid" in light space.

    // 3. Snap: Round position to nearest texel_size
    let texel_size = (2.0 * radius) / texture_size as f32;

    // We project the camera center into light space to find its coordinates.
    // But since eye = center - dir*radius, wait...
    // Let's use a fixed-point view to handle snapping properly across frames.
    let base_view = Mat4::look_at_rh(-light_dir_norm * radius, Vec3::ZERO, Vec3::Y);
    let light_space_center = base_view.transform_point3(camera_center);

    let snapped_x = (light_space_center.x / texel_size).floor() * texel_size;
    let snapped_y = (light_space_center.y / texel_size).floor() * texel_size;

    // 4. Ortho: Create orthographic projection centered on the snapped position.
    let left = snapped_x - radius;
    let right = snapped_x + radius;
    let bottom = snapped_y - radius;
    let top = snapped_y + radius;

    // Large depth range to ensure the scene is captured
    let near = -radius * 4.0;
    let far = radius * 4.0;

    let proj = Mat4::orthographic_rh(left, right, bottom, top, near, far);

    proj * base_view
}
