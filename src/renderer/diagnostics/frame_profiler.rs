//! CPU-side frame profiling with rolling averages

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::FrameStats;

/// Window size for rolling averages
const FRAME_WINDOW: usize = 60;

/// CPU-side frame profiler with rolling statistics
#[derive(Debug)]
pub struct FrameProfiler {
    /// Frame time samples (in seconds)
    frame_times: VecDeque<f32>,
    /// Last frame start time
    last_frame_start: Instant,
    /// Current frame start time
    current_frame_start: Instant,
    /// Total frames profiled
    total_frames: u64,
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameProfiler {
    /// Create a new frame profiler
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            frame_times: VecDeque::with_capacity(FRAME_WINDOW),
            last_frame_start: now,
            current_frame_start: now,
            total_frames: 0,
        }
    }

    /// Mark the start of a new frame
    pub fn begin_frame(&mut self) {
        self.last_frame_start = self.current_frame_start;
        self.current_frame_start = Instant::now();

        // Calculate frame time from last frame
        let frame_time = self
            .current_frame_start
            .duration_since(self.last_frame_start)
            .as_secs_f32();

        // Add to rolling window
        if self.frame_times.len() >= FRAME_WINDOW {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(frame_time);
        self.total_frames += 1;
    }

    /// Get current frame stats
    pub fn stats(&self, draw_calls: u32, triangles: u64) -> FrameStats {
        let (fps, avg_ms, min_ms, max_ms) = self.compute_stats();

        FrameStats {
            fps,
            frame_time_ms: avg_ms,
            frame_time_min_ms: min_ms,
            frame_time_max_ms: max_ms,
            draw_calls,
            triangles,
            total_frames: self.total_frames,
        }
    }

    /// Compute rolling statistics
    fn compute_stats(&self) -> (f32, f32, f32, f32) {
        if self.frame_times.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let count = self.frame_times.len() as f32;
        let mut sum = 0.0f32;
        let mut min = f32::MAX;
        let mut max = 0.0f32;

        for &time in &self.frame_times {
            sum += time;
            min = min.min(time);
            max = max.max(time);
        }

        let avg = sum / count;
        let fps = if avg > 0.0 { 1.0 / avg } else { 0.0 };

        (fps, avg * 1000.0, min * 1000.0, max * 1000.0)
    }

    /// Get elapsed time since current frame started
    pub fn frame_elapsed(&self) -> Duration {
        Instant::now().duration_since(self.current_frame_start)
    }

    /// Get total frames profiled
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_frame_profiler_basic() {
        let mut profiler = FrameProfiler::new();

        // Simulate a few frames
        for _ in 0..5 {
            profiler.begin_frame();
            thread::sleep(Duration::from_millis(16));
        }

        let stats = profiler.stats(10, 1000);
        assert!(stats.fps() > 0.0);
        assert!(stats.frame_time_ms() > 0.0);
        assert_eq!(profiler.total_frames(), 5);
    }
}
