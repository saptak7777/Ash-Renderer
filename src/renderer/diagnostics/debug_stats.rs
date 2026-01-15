use crate::renderer::vram_budget::VramStats;

/// High-level renderer telemetry and statistics.
#[derive(Debug, Clone)]
pub struct RendererStats {
    pub vram_usage: VramStats,
    pub draw_calls_per_frame: u32,
    pub triangles_rendered: u64,
    /// Culling efficiency (0.0-1.0, higher is better)
    pub cull_efficiency: f32,
    /// Total GPU frame time (milliseconds)
    pub gpu_frame_ms: f32,
    /// Current Hi-Z quality mode
    pub hiz_quality: String,
    /// Frame counter for periodic reporting
    pub frame_count: u64,
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

    /// Log periodic performance report (every 60 frames = ~1 second at 60 FPS)
    pub fn log_perf_report(&mut self) {
        self.frame_count += 1;

        // Log every 60 frames
        if self.frame_count % 60 == 0 {
            let fps = if self.gpu_frame_ms > 0.0 {
                1000.0 / self.gpu_frame_ms
            } else {
                0.0
            };

            log::info!(
                "[PERF] FPS: {:.0} | GPU: {:.2}ms | Cull: {:.1}% | Hi-Z: {} | Draws: {}",
                fps,
                self.gpu_frame_ms,
                self.cull_efficiency * 100.0,
                self.hiz_quality,
                self.draw_calls_per_frame
            );
        }
    }

    /// Update frame counter
    pub fn increment_frame(&mut self) {
        self.frame_count += 1;
    }

    /// Get current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}
