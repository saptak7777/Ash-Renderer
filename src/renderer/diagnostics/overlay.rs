//! Diagnostics text overlay renderer
//!
//! Generates vertices for debug text display using an embedded bitmap font.
//! The overlay is rendered in screen space with semi-transparent background.

use super::DiagnosticsState;
use super::font_data::{GLYPH_HEIGHT, GLYPH_WIDTH, get_glyph};
use super::overlay_types::{OverlayConfig, TextVertex, generate_quad_ndc, pixel_to_ndc};

/// Parameters for rasterizing a single glyph
struct GlyphRasterContext<'a> {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub glyph: &'a [u8; 8],
    pub color: [f32; 4],
    pub screen_w: f32,
    pub screen_h: f32,
}

/// Diagnostics overlay renderer
///
/// Generates vertex data for the debug overlay. Does not perform GPU rendering
/// directly - that's handled by `OverlayPipeline`.
pub struct DiagnosticsOverlay {
    config: OverlayConfig,
    /// Cached text vertices
    vertices: Vec<TextVertex>,
    /// Cached background vertices
    bg_vertices: Vec<TextVertex>,
}

impl DiagnosticsOverlay {
    /// Create a new diagnostics overlay with default config
    pub fn new() -> Self {
        Self {
            config: OverlayConfig::default(),
            vertices: Vec::with_capacity(2048),
            bg_vertices: Vec::with_capacity(6),
        }
    }

    /// Create with custom config
    pub fn with_config(config: OverlayConfig) -> Self {
        Self {
            config,
            vertices: Vec::with_capacity(2048),
            bg_vertices: Vec::with_capacity(6),
        }
    }

    /// Update overlay config
    pub fn set_config(&mut self, config: OverlayConfig) {
        self.config = config;
    }

    /// Get current config
    pub fn config(&self) -> &OverlayConfig {
        &self.config
    }

    /// Generate vertices for diagnostics text
    ///
    /// Returns (text_vertices, background_vertices)
    pub fn generate_vertices(
        &mut self,
        diagnostics: &DiagnosticsState,
        screen_width: f32,
        screen_height: f32,
    ) -> (&[TextVertex], &[TextVertex]) {
        self.vertices.clear();
        self.bg_vertices.clear();

        let lines = diagnostics.format_overlay();
        if lines.is_empty() {
            return (&self.vertices, &self.bg_vertices);
        }

        let glyph_w = GLYPH_WIDTH as f32 * self.config.scale;
        let glyph_h = GLYPH_HEIGHT as f32 * self.config.scale;
        let line_height = glyph_h * self.config.line_spacing;

        // Calculate background dimensions
        let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0) as f32;
        let padding = self.config.padding;
        let bg_width = max_chars * glyph_w + padding * 2.0;
        let bg_height = lines.len() as f32 * line_height + padding * 2.0;

        // Generate background quad
        let bg_x = self.config.offset[0] - padding;
        let bg_y = self.config.offset[1] - padding;
        let bg_quad = generate_quad_ndc(
            bg_x,
            bg_y,
            bg_width,
            bg_height,
            self.config.bg_color,
            screen_width,
            screen_height,
        );
        self.bg_vertices.extend_from_slice(&bg_quad);

        // Generate text vertices
        let mut y = self.config.offset[1];
        for line in &lines {
            let mut x = self.config.offset[0];
            for ch in line.chars() {
                if let Some(glyph) = get_glyph(ch) {
                    self.rasterize_glyph(&GlyphRasterContext {
                        x,
                        y,
                        w: glyph_w,
                        h: glyph_h,
                        glyph,
                        color: self.config.color,
                        screen_w: screen_width,
                        screen_h: screen_height,
                    });
                }
                x += glyph_w;
            }
            y += line_height;
        }

        (&self.vertices, &self.bg_vertices)
    }

    /// Rasterize a single glyph into vertices
    fn rasterize_glyph(&mut self, ctx: &GlyphRasterContext) {
        let pixel_w = ctx.w / 8.0;
        let pixel_h = ctx.h / 8.0;

        for (row, &row_bits) in ctx.glyph.iter().enumerate() {
            if row_bits == 0 {
                continue; // Skip empty rows for performance
            }

            for col in 0..8 {
                if (row_bits >> (7 - col)) & 1 == 1 {
                    let px = ctx.x + col as f32 * pixel_w;
                    let py = ctx.y + row as f32 * pixel_h;

                    let tl = pixel_to_ndc(px, py, ctx.screen_w, ctx.screen_h);
                    let tr = pixel_to_ndc(px + pixel_w, py, ctx.screen_w, ctx.screen_h);
                    let bl = pixel_to_ndc(px, py + pixel_h, ctx.screen_w, ctx.screen_h);
                    let br = pixel_to_ndc(px + pixel_w, py + pixel_h, ctx.screen_w, ctx.screen_h);

                    // Two triangles for quad
                    self.vertices
                        .push(TextVertex::new(tl, [0.0, 0.0], ctx.color));
                    self.vertices
                        .push(TextVertex::new(tr, [1.0, 0.0], ctx.color));
                    self.vertices
                        .push(TextVertex::new(bl, [0.0, 1.0], ctx.color));
                    self.vertices
                        .push(TextVertex::new(bl, [0.0, 1.0], ctx.color));
                    self.vertices
                        .push(TextVertex::new(tr, [1.0, 0.0], ctx.color));
                    self.vertices
                        .push(TextVertex::new(br, [1.0, 1.0], ctx.color));
                }
            }
        }
    }

    /// Get current text vertex count
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get background vertex count
    pub fn bg_vertex_count(&self) -> usize {
        self.bg_vertices.len()
    }

    /// Clear cached vertices
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.bg_vertices.clear();
    }
}

impl Default for DiagnosticsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_creation() {
        let overlay = DiagnosticsOverlay::new();
        assert_eq!(overlay.vertex_count(), 0);
        assert_eq!(overlay.bg_vertex_count(), 0);
    }

    #[test]
    fn test_overlay_vertices() {
        let mut overlay = DiagnosticsOverlay::new();
        let diagnostics = DiagnosticsState::default();

        let (text, bg) = overlay.generate_vertices(&diagnostics, 1920.0, 1080.0);

        // Should have background vertices
        assert!(!bg.is_empty(), "Background should have vertices");
        // Text vertices depend on content
        assert!(!text.is_empty() || !bg.is_empty());
    }
}
