use crate::renderer::vram_budget::VramStats;

/// High-level renderer telemetry and statistics.
#[derive(Debug, Clone)]
pub struct RendererStats {
    pub vram_usage: VramStats,
    pub draw_calls_per_frame: u32,
    pub triangles_rendered: u64,
}

impl RendererStats {
    /// Log frame statistics to the debug log.
    pub fn log_frame_stats(&self) {
        log::debug!(
            "Frame Stats | VRAM: {:.1}% ({:.1}MB / {:.1}MB) | Draws: {} | Tris: {:.1}M",
            self.vram_usage.utilization_percent,
            self.vram_usage.used_textures as f32 / 1024.0 / 1024.0,
            (self.vram_usage.used_textures + self.vram_usage.available_budget) as f32
                / 1024.0
                / 1024.0,
            self.draw_calls_per_frame,
            self.triangles_rendered as f32 / 1_000_000.0
        );

        if self.vram_usage.utilization_percent > 85.0 {
            log::warn!(
                "CRITICAL: VRAM utilization is high ({:.1}%)",
                self.vram_usage.utilization_percent
            );
        }
    }
}
