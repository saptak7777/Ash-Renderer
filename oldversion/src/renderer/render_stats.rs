//! Unified Render Statistics Dashboard
//!
//! Aggregates performance metrics from all rendering subsystems.
//! Provides a single source of truth for performance monitoring.

use crate::renderer::diagnostics::ExtendedGpuTimings;
use crate::renderer::resources::{BufferPoolStats, PoolStats};

/// Frame timing statistics
#[derive(Debug, Clone, Default)]
pub struct FrameTimings {
    /// Total frame time (ms)
    pub frame_ms: f32,
    /// CPU time (ms)
    pub cpu_ms: f32,
    /// GPU time (ms)
    pub gpu_ms: f32,
    /// Present/vsync wait (ms)
    pub present_ms: f32,
}

/// Draw call statistics
#[derive(Debug, Clone, Default)]
pub struct DrawStats {
    /// Total draw calls
    pub draw_calls: u32,
    /// Instanced draw calls
    pub instanced_draws: u32,
    /// Indirect draw calls
    pub indirect_draws: u32,
    /// Total triangles rendered
    pub triangles: u64,
    /// Total vertices processed
    pub vertices: u64,
}

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total GPU memory allocated (bytes)
    pub gpu_allocated: u64,
    /// Buffers in use
    pub buffers_in_use: usize,
    /// Buffers available in pools
    pub buffers_available: usize,
    /// Texture memory (bytes)
    pub texture_memory: u64,
    /// Command buffer memory (bytes)
    pub command_memory: u64,
}

/// Culling statistics
#[derive(Debug, Clone, Default)]
pub struct CullingStats {
    /// Objects submitted
    pub objects_submitted: u32,
    /// Objects after frustum culling
    pub after_frustum: u32,
    /// Objects after occlusion culling
    pub after_occlusion: u32,
    /// Lights culled
    pub lights_culled: u32,
    /// Triangles culled by LOD
    pub lod_triangles_saved: u64,
}

/// Complete render statistics
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    /// Frame number
    pub frame: u64,
    /// Timing information
    pub timings: FrameTimings,
    /// Draw call information
    pub draws: DrawStats,
    /// Memory usage
    pub memory: MemoryStats,
    /// Culling efficiency
    pub culling: CullingStats,
    /// FPS (smoothed)
    pub fps: f32,
}

impl RenderStats {
    /// Calculate FPS from frame time
    pub fn calculate_fps(&mut self) {
        if self.timings.frame_ms > 0.0 {
            self.fps = 1000.0 / self.timings.frame_ms;
        }
    }

    /// Calculate overall efficiency score (0-100)
    pub fn efficiency_score(&self) -> f32 {
        let mut score = 100.0;

        // Penalize for high draw call count
        score -= (self.draws.draw_calls as f32 / 100.0).min(30.0);

        // Reward for instancing
        if self.draws.draw_calls > 0 {
            let instancing_ratio = self.draws.instanced_draws as f32 / self.draws.draw_calls as f32;
            score += instancing_ratio * 10.0;
        }

        // Reward for culling
        if self.culling.objects_submitted > 0 {
            let cull_ratio =
                1.0 - (self.culling.after_occlusion as f32 / self.culling.objects_submitted as f32);
            score += cull_ratio * 20.0;
        }

        score.clamp(0.0, 100.0)
    }

    /// Format as compact string
    pub fn format_compact(&self) -> String {
        format!(
            "{:.1}fps | {:.2}ms | {} draws | {:.1}M tris",
            self.fps,
            self.timings.frame_ms,
            self.draws.draw_calls,
            self.draws.triangles as f64 / 1_000_000.0
        )
    }

    /// Format as detailed multi-line string
    pub fn format_detailed(&self) -> String {
        format!(
            "Frame {:>6} | {:.1} FPS ({:.2}ms)\n\
             CPU: {:.2}ms | GPU: {:.2}ms | Present: {:.2}ms\n\
             Draws: {} (inst: {}, indirect: {})\n\
             Tris: {:.2}M | Verts: {:.2}M\n\
             Culled: {}/{} objects ({:.1}%)\n\
             Memory: {:.1}MB GPU | {} buffers\n\
             Score: {:.0}/100",
            self.frame,
            self.fps,
            self.timings.frame_ms,
            self.timings.cpu_ms,
            self.timings.gpu_ms,
            self.timings.present_ms,
            self.draws.draw_calls,
            self.draws.instanced_draws,
            self.draws.indirect_draws,
            self.draws.triangles as f64 / 1_000_000.0,
            self.draws.vertices as f64 / 1_000_000.0,
            self.culling.objects_submitted - self.culling.after_occlusion,
            self.culling.objects_submitted,
            if self.culling.objects_submitted > 0 {
                (1.0 - self.culling.after_occlusion as f32 / self.culling.objects_submitted as f32)
                    * 100.0
            } else {
                0.0
            },
            self.memory.gpu_allocated as f64 / (1024.0 * 1024.0),
            self.memory.buffers_in_use,
            self.efficiency_score()
        )
    }
}

/// Statistics collector (single frame accumulator)
#[derive(Debug, Default)]
pub struct StatsCollector {
    stats: RenderStats,
}

impl StatsCollector {
    /// Create new collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin new frame
    pub fn begin_frame(&mut self, frame: u64) {
        self.stats = RenderStats {
            frame,
            ..Default::default()
        };
    }

    /// Record frame timing
    pub fn record_timing(&mut self, frame_ms: f32, cpu_ms: f32, gpu_ms: f32) {
        self.stats.timings.frame_ms = frame_ms;
        self.stats.timings.cpu_ms = cpu_ms;
        self.stats.timings.gpu_ms = gpu_ms;
        self.stats.timings.present_ms = frame_ms - cpu_ms - gpu_ms;
        self.stats.calculate_fps();
    }

    /// Record GPU timings from profiler
    pub fn record_gpu_timings(&mut self, timings: &ExtendedGpuTimings) {
        self.stats.timings.gpu_ms = timings.total_ms;
    }

    /// Record a draw call
    pub fn record_draw(&mut self, triangles: u32, vertices: u32, instanced: bool) {
        self.stats.draws.draw_calls += 1;
        self.stats.draws.triangles += triangles as u64;
        self.stats.draws.vertices += vertices as u64;
        if instanced {
            self.stats.draws.instanced_draws += 1;
        }
    }

    /// Record indirect draw
    pub fn record_indirect_draw(&mut self, draw_count: u32) {
        self.stats.draws.indirect_draws += draw_count;
        self.stats.draws.draw_calls += draw_count;
    }

    /// Record culling results
    pub fn record_culling(&mut self, submitted: u32, after_frustum: u32, after_occlusion: u32) {
        self.stats.culling.objects_submitted = submitted;
        self.stats.culling.after_frustum = after_frustum;
        self.stats.culling.after_occlusion = after_occlusion;
    }

    /// Record buffer pool stats
    pub fn record_buffer_pool(&mut self, stats: &BufferPoolStats) {
        self.stats.memory.buffers_available = stats.current_available;
        self.stats.memory.buffers_in_use = stats.current_in_use;
        self.stats.memory.gpu_allocated = stats.total_allocated_bytes;
    }

    /// Record thread pool stats
    pub fn record_thread_pool(&mut self, _stats: &PoolStats) {
        // Could add thread pool stats if needed
    }

    /// Record LOD savings
    pub fn record_lod_savings(&mut self, triangles_saved: u64) {
        self.stats.culling.lod_triangles_saved = triangles_saved;
    }

    /// Get final statistics
    pub fn finish(&self) -> RenderStats {
        self.stats.clone()
    }

    /// Get current stats reference
    pub fn current(&self) -> &RenderStats {
        &self.stats
    }
}

/// Rolling average for stats smoothing
pub struct StatsHistory {
    history: Vec<RenderStats>,
    capacity: usize,
    index: usize,
}

impl StatsHistory {
    /// Create with capacity for N frames
    pub fn new(capacity: usize) -> Self {
        Self {
            history: Vec::with_capacity(capacity),
            capacity,
            index: 0,
        }
    }

    /// Add frame stats
    pub fn push(&mut self, stats: RenderStats) {
        if self.history.len() < self.capacity {
            self.history.push(stats);
        } else {
            self.history[self.index] = stats;
        }
        self.index = (self.index + 1) % self.capacity;
    }

    /// Get average FPS
    pub fn average_fps(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.fps).sum();
        sum / self.history.len() as f32
    }

    /// Get average frame time
    pub fn average_frame_ms(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.timings.frame_ms).sum();
        sum / self.history.len() as f32
    }

    /// Get min/max FPS
    pub fn fps_range(&self) -> (f32, f32) {
        if self.history.is_empty() {
            return (0.0, 0.0);
        }
        let min = self.history.iter().map(|s| s.fps).fold(f32::MAX, f32::min);
        let max = self.history.iter().map(|s| s.fps).fold(f32::MIN, f32::max);
        (min, max)
    }
}

impl Default for StatsHistory {
    fn default() -> Self {
        Self::new(120) // 2 seconds at 60fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_collection() {
        let mut collector = StatsCollector::new();
        collector.begin_frame(1);
        collector.record_timing(16.67, 5.0, 10.0);
        collector.record_draw(1000, 500, false);

        let stats = collector.finish();
        assert_eq!(stats.frame, 1);
        assert!(stats.fps > 59.0 && stats.fps < 61.0);
        assert_eq!(stats.draws.draw_calls, 1);
    }

    #[test]
    fn test_history_average() {
        let mut history = StatsHistory::new(3);

        for fps in [60.0, 62.0, 58.0] {
            let stats = RenderStats {
                fps,
                ..Default::default()
            };
            history.push(stats);
        }

        assert!((history.average_fps() - 60.0).abs() < 0.01);
    }
}
