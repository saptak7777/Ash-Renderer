//! Overlay type definitions
//!
//! Contains vertex types and configuration for the diagnostics overlay.

/// Text vertex for overlay rendering
///
/// Used for both text glyphs and background quads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    /// Position in NDC (-1 to 1)
    pub pos: [f32; 2],
    /// UV coordinates (unused for solid color, used for future texture atlas)
    pub uv: [f32; 2],
    /// RGBA color
    pub color: [f32; 4],
}

impl TextVertex {
    /// Create a new text vertex
    pub const fn new(pos: [f32; 2], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self { pos, uv, color }
    }

    /// Vertex input binding description for Vulkan
    pub fn binding_description() -> ash::vk::VertexInputBindingDescription {
        ash::vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: ash::vk::VertexInputRate::VERTEX,
        }
    }

    /// Vertex attribute descriptions for Vulkan
    pub fn attribute_descriptions() -> [ash::vk::VertexInputAttributeDescription; 3] {
        [
            // Position
            ash::vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: ash::vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            // UV
            ash::vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: ash::vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            // Color
            ash::vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: ash::vk::Format::R32G32B32A32_SFLOAT,
                offset: 16,
            },
        ]
    }
}

/// Overlay configuration
#[derive(Clone, Debug)]
pub struct OverlayConfig {
    /// Font scale (2 = 16x16 pixels per glyph)
    pub scale: f32,
    /// Position offset from top-left in pixels
    pub offset: [f32; 2],
    /// Text color (RGBA)
    pub color: [f32; 4],
    /// Background color (RGBA, 0 alpha = transparent)
    pub bg_color: [f32; 4],
    /// Line spacing multiplier
    pub line_spacing: f32,
    /// Padding around text block
    pub padding: f32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            offset: [10.0, 10.0],
            color: [0.0, 1.0, 0.0, 1.0],    // Green
            bg_color: [0.0, 0.0, 0.0, 0.7], // Semi-transparent black
            line_spacing: 1.25,
            padding: 8.0,
        }
    }
}

impl OverlayConfig {
    /// Create a config optimized for minimal overhead
    pub fn minimal() -> Self {
        Self {
            scale: 1.5,
            offset: [5.0, 5.0],
            color: [1.0, 1.0, 1.0, 0.8],
            bg_color: [0.0, 0.0, 0.0, 0.5],
            line_spacing: 1.0,
            padding: 4.0,
        }
    }

    /// Create a config for high visibility
    pub fn high_visibility() -> Self {
        Self {
            scale: 2.5,
            offset: [15.0, 15.0],
            color: [1.0, 1.0, 0.0, 1.0], // Yellow
            bg_color: [0.0, 0.0, 0.2, 0.9],
            line_spacing: 1.5,
            padding: 12.0,
        }
    }
}

/// Helper for converting pixel coordinates to NDC
#[inline]
pub fn pixel_to_ndc(px: f32, py: f32, screen_w: f32, screen_h: f32) -> [f32; 2] {
    [(px / screen_w) * 2.0 - 1.0, (py / screen_h) * 2.0 - 1.0]
}

/// Generate a quad (2 triangles = 6 vertices) in NDC
pub fn generate_quad_ndc(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) -> [TextVertex; 6] {
    let tl = pixel_to_ndc(x, y, screen_w, screen_h);
    let tr = pixel_to_ndc(x + w, y, screen_w, screen_h);
    let bl = pixel_to_ndc(x, y + h, screen_w, screen_h);
    let br = pixel_to_ndc(x + w, y + h, screen_w, screen_h);

    [
        TextVertex::new(tl, [0.0, 0.0], color),
        TextVertex::new(tr, [1.0, 0.0], color),
        TextVertex::new(bl, [0.0, 1.0], color),
        TextVertex::new(bl, [0.0, 1.0], color),
        TextVertex::new(tr, [1.0, 0.0], color),
        TextVertex::new(br, [1.0, 1.0], color),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_vertex_size() {
        assert_eq!(std::mem::size_of::<TextVertex>(), 32); // 2+2+4 floats = 8 floats = 32 bytes
    }

    #[test]
    fn test_pixel_to_ndc() {
        let ndc = pixel_to_ndc(960.0, 540.0, 1920.0, 1080.0);
        assert!((ndc[0] - 0.0).abs() < 0.001);
        assert!((ndc[1] - 0.0).abs() < 0.001);
    }
}
