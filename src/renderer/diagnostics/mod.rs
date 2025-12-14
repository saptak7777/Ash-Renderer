//! Diagnostics and profiling system for the Ash Renderer.
//!
//! Provides runtime stats collection, GPU timing queries, and display options
//! for performance monitoring and debugging.
//!
//! # Example
//! ```ignore
//! // Toggle diagnostics with F6
//! renderer.set_diagnostics_mode(DiagnosticsMode::ConsoleOnly);
//!
//! // Get current stats
//! let stats = renderer.diagnostics();
//! println!("FPS: {:.1}", stats.frame_stats.fps());
//! ```

mod font_data;
mod frame_profiler;
mod gpu_profiler;
mod overlay;
mod overlay_pipeline;
mod overlay_types;

pub use font_data::{get_glyph, FONT_8X8, GLYPH_HEIGHT, GLYPH_WIDTH};
pub use frame_profiler::FrameProfiler;
pub use gpu_profiler::{ExtendedGpuTimings, GpuProfiler, TimingScope};
pub use overlay::DiagnosticsOverlay;
pub use overlay_pipeline::OverlayPipeline;
pub use overlay_types::{generate_quad_ndc, pixel_to_ndc, OverlayConfig, TextVertex};

/// Controls how diagnostics are displayed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiagnosticsMode {
    /// No diagnostics output
    #[default]
    Off,
    /// Print stats to console only
    ConsoleOnly,
    /// Show in-engine overlay only
    OverlayOnly,
    /// Both console and overlay, F6 to toggle
    BothWithToggle,
}

impl DiagnosticsMode {
    /// Cycle to next mode (for F6 toggle)
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::ConsoleOnly,
            Self::ConsoleOnly => Self::OverlayOnly,
            Self::OverlayOnly => Self::BothWithToggle,
            Self::BothWithToggle => Self::Off,
        }
    }

    /// Whether console output is enabled
    pub fn console_enabled(&self) -> bool {
        matches!(self, Self::ConsoleOnly | Self::BothWithToggle)
    }

    /// Whether overlay is enabled
    pub fn overlay_enabled(&self) -> bool {
        matches!(self, Self::OverlayOnly | Self::BothWithToggle)
    }
}

/// Frame timing statistics with rolling averages
#[derive(Debug, Clone)]
pub struct FrameStats {
    /// Frames per second (rolling average)
    fps: f32,
    /// Frame time in milliseconds (rolling average)
    frame_time_ms: f32,
    /// Minimum frame time in current window
    frame_time_min_ms: f32,
    /// Maximum frame time in current window
    frame_time_max_ms: f32,
    /// Number of draw calls this frame
    pub draw_calls: u32,
    /// Number of triangles rendered
    pub triangles: u64,
    /// Total frames rendered
    pub total_frames: u64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            fps: 0.0,
            frame_time_ms: 0.0,
            frame_time_min_ms: f32::MAX,
            frame_time_max_ms: 0.0,
            draw_calls: 0,
            triangles: 0,
            total_frames: 0,
        }
    }
}

impl FrameStats {
    /// Get current FPS
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Get average frame time in milliseconds
    pub fn frame_time_ms(&self) -> f32 {
        self.frame_time_ms
    }

    /// Get min/max frame times
    pub fn frame_time_range_ms(&self) -> (f32, f32) {
        (self.frame_time_min_ms, self.frame_time_max_ms)
    }

    /// Format stats as a single line string
    pub fn format_line(&self) -> String {
        format!(
            "FPS: {:.1} | Frame: {:.2}ms (min: {:.2}, max: {:.2}) | Draws: {} | Tris: {}",
            self.fps,
            self.frame_time_ms,
            self.frame_time_min_ms,
            self.frame_time_max_ms,
            self.draw_calls,
            self.triangles
        )
    }
}

/// GPU timing results per pass
#[derive(Debug, Clone, Default)]
pub struct GpuTimings {
    /// Total GPU time for the frame (ms)
    pub total_ms: f32,
    /// Scene rendering pass (ms)
    pub scene_ms: f32,
    /// Post-processing pass (ms)
    pub post_process_ms: f32,
    /// UI/overlay pass (ms)
    pub ui_ms: f32,
}

impl GpuTimings {
    /// Format GPU timings as a string
    pub fn format_line(&self) -> String {
        format!(
            "GPU: {:.2}ms | Scene: {:.2}ms | Post: {:.2}ms | UI: {:.2}ms",
            self.total_ms, self.scene_ms, self.post_process_ms, self.ui_ms
        )
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// GPU memory used (bytes)
    pub gpu_used_bytes: u64,
    /// GPU memory budget (bytes)
    pub gpu_budget_bytes: u64,
    /// Number of allocations
    pub allocation_count: u32,
    /// Buffer pool stats (available, in_use, total_allocated)
    pub buffer_pool: (usize, usize, u64),
}

impl MemoryStats {
    /// Format memory stats as a string
    pub fn format_line(&self) -> String {
        let used_mb = self.gpu_used_bytes as f64 / (1024.0 * 1024.0);
        let budget_mb = self.gpu_budget_bytes as f64 / (1024.0 * 1024.0);
        let pool_mb = self.buffer_pool.2 as f64 / (1024.0 * 1024.0);
        format!(
            "VRAM: {:.1}/{:.1} MB | Allocs: {} | Pool: {:.1} MB ({} avail, {} used)",
            used_mb,
            budget_mb,
            self.allocation_count,
            pool_mb,
            self.buffer_pool.0,
            self.buffer_pool.1
        )
    }
}

/// Combined diagnostics state
#[derive(Debug, Clone)]
pub struct DiagnosticsState {
    /// Current display mode
    pub mode: DiagnosticsMode,
    /// Frame timing stats
    pub frame_stats: FrameStats,
    /// GPU timing per pass
    pub gpu_timings: GpuTimings,
    /// Memory usage
    pub memory_stats: MemoryStats,
    /// Frames since last console print
    console_print_counter: u32,
    /// Print to console every N frames
    console_print_interval: u32,
}

impl Default for DiagnosticsState {
    fn default() -> Self {
        Self {
            mode: DiagnosticsMode::Off,
            frame_stats: FrameStats::default(),
            gpu_timings: GpuTimings::default(),
            memory_stats: MemoryStats::default(),
            console_print_counter: 0,
            console_print_interval: 60, // Every 60 frames (~1 second at 60fps)
        }
    }
}

impl DiagnosticsState {
    /// Create with a specific mode
    pub fn with_mode(mode: DiagnosticsMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set console print interval (in frames)
    pub fn set_console_interval(&mut self, frames: u32) {
        self.console_print_interval = frames.max(1);
    }

    /// Toggle to next diagnostics mode
    pub fn toggle_mode(&mut self) {
        self.mode = self.mode.next();
        log::info!("Diagnostics mode: {:?}", self.mode);
    }

    /// Should print to console this frame?
    pub fn should_print_console(&mut self) -> bool {
        if !self.mode.console_enabled() {
            return false;
        }
        self.console_print_counter += 1;
        if self.console_print_counter >= self.console_print_interval {
            self.console_print_counter = 0;
            true
        } else {
            false
        }
    }

    /// Print stats to console
    pub fn print_console(&self) {
        println!("┌─ Ash Renderer Diagnostics ─────────────────────────────────────");
        println!("│ {}", self.frame_stats.format_line());
        println!("│ {}", self.gpu_timings.format_line());
        println!("│ {}", self.memory_stats.format_line());
        println!("└─────────────────────────────────────────────────────────");
    }

    /// Format all stats for overlay
    pub fn format_overlay(&self) -> Vec<String> {
        vec![
            format!("Ash Renderer v{}", env!("CARGO_PKG_VERSION")),
            self.frame_stats.format_line(),
            self.gpu_timings.format_line(),
            self.memory_stats.format_line(),
        ]
    }

    /// Reset per-frame counters (call at start of frame)
    pub fn begin_frame(&mut self) {
        self.frame_stats.draw_calls = 0;
        self.frame_stats.triangles = 0;
    }

    /// Record a draw call
    pub fn record_draw(&mut self, triangle_count: u64) {
        self.frame_stats.draw_calls += 1;
        self.frame_stats.triangles += triangle_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_mode_cycle() {
        let mut mode = DiagnosticsMode::Off;
        mode = mode.next();
        assert_eq!(mode, DiagnosticsMode::ConsoleOnly);
        mode = mode.next();
        assert_eq!(mode, DiagnosticsMode::OverlayOnly);
        mode = mode.next();
        assert_eq!(mode, DiagnosticsMode::BothWithToggle);
        mode = mode.next();
        assert_eq!(mode, DiagnosticsMode::Off);
    }

    #[test]
    fn test_frame_stats_format() {
        let stats = FrameStats {
            fps: 60.0,
            frame_time_ms: 16.67,
            frame_time_min_ms: 15.0,
            frame_time_max_ms: 18.0,
            draw_calls: 100,
            triangles: 50000,
            total_frames: 1000,
        };
        let line = stats.format_line();
        assert!(line.contains("60.0"));
        assert!(line.contains("100"));
    }
}
