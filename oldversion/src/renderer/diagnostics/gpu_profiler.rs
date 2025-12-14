//! GPU timing profiler using Vulkan timestamp queries
//!
//! Features:
//! - Double-buffered query pools for async retrieval
//! - Non-blocking query result fetching
//! - Per-pass timing breakdown
//! - Automatic fallback when timestamps unavailable

use std::sync::Arc;

use ash::vk;
use ash::Device;

use super::GpuTimings;

/// Maximum number of timing scopes per frame
const MAX_TIMESTAMPS: u32 = 32;

/// Double-buffering for query pools (read from N-1 while writing to N)
const QUERY_POOL_COUNT: usize = 2;

/// Named timing scope indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TimingScope {
    /// Start of frame
    FrameStart = 0,
    /// End of shadow pass
    ShadowEnd = 1,
    /// End of main scene rendering
    SceneEnd = 2,
    /// End of bloom threshold
    BloomThresholdEnd = 3,
    /// End of bloom downsample
    BloomDownsampleEnd = 4,
    /// End of bloom upsample
    BloomUpsampleEnd = 5,
    /// End of post-processing (tonemapping)
    PostProcessEnd = 6,
    /// End of UI rendering
    UiEnd = 7,
    /// End of frame
    FrameEnd = 8,
}

impl TimingScope {
    fn index(self) -> u32 {
        self as u32
    }

    /// Get all scopes in order
    pub fn all() -> &'static [TimingScope] {
        &[
            TimingScope::FrameStart,
            TimingScope::ShadowEnd,
            TimingScope::SceneEnd,
            TimingScope::BloomThresholdEnd,
            TimingScope::BloomDownsampleEnd,
            TimingScope::BloomUpsampleEnd,
            TimingScope::PostProcessEnd,
            TimingScope::UiEnd,
            TimingScope::FrameEnd,
        ]
    }
}

/// Extended GPU timings with per-pass breakdown
#[derive(Debug, Clone, Default)]
pub struct ExtendedGpuTimings {
    /// Total frame GPU time (ms)
    pub total_ms: f32,
    /// Shadow pass (ms)
    pub shadow_ms: f32,
    /// Scene rendering (ms)
    pub scene_ms: f32,
    /// Bloom threshold (ms)
    pub bloom_threshold_ms: f32,
    /// Bloom downsample (ms)
    pub bloom_downsample_ms: f32,
    /// Bloom upsample (ms)
    pub bloom_upsample_ms: f32,
    /// Post-processing/tonemapping (ms)
    pub post_process_ms: f32,
    /// UI overlay (ms)
    pub ui_ms: f32,
    /// Whether data is valid (queries completed)
    pub valid: bool,
}

impl ExtendedGpuTimings {
    /// Convert to basic GpuTimings for backward compatibility
    pub fn to_basic(&self) -> GpuTimings {
        GpuTimings {
            total_ms: self.total_ms,
            scene_ms: self.shadow_ms + self.scene_ms,
            post_process_ms: self.bloom_threshold_ms
                + self.bloom_downsample_ms
                + self.bloom_upsample_ms
                + self.post_process_ms,
            ui_ms: self.ui_ms,
        }
    }

    /// Format detailed timing breakdown
    pub fn format_detailed(&self) -> String {
        if !self.valid {
            return "GPU: (waiting for data)".to_string();
        }
        format!(
            "GPU: {:.2}ms | Shadow: {:.2}ms | Scene: {:.2}ms | Bloom: {:.2}ms | Post: {:.2}ms | UI: {:.2}ms",
            self.total_ms,
            self.shadow_ms,
            self.scene_ms,
            self.bloom_threshold_ms + self.bloom_downsample_ms + self.bloom_upsample_ms,
            self.post_process_ms,
            self.ui_ms
        )
    }
}

/// GPU profiler using Vulkan timestamp queries
///
/// Uses double-buffered query pools with non-blocking result retrieval.
pub struct GpuProfiler {
    device: Arc<Device>,
    /// Query pools (double-buffered)
    query_pools: [vk::QueryPool; QUERY_POOL_COUNT],
    /// Current pool index (writing)
    current_pool: usize,
    /// Nanoseconds per timestamp tick
    timestamp_period_ns: f64,
    /// Whether timestamps are supported
    timestamps_supported: bool,
    /// Last successfully retrieved timing results
    last_results: ExtendedGpuTimings,
    /// Number of frames since last successful result
    frames_since_result: u32,
    /// Total frames profiled
    total_frames: u64,
}

impl GpuProfiler {
    /// Create a new GPU profiler
    ///
    /// # Safety
    /// Device must be valid and outlive this profiler
    pub unsafe fn new(
        device: Arc<Device>,
        timestamp_period_ns: f32,
        timestamps_supported: bool,
    ) -> crate::Result<Self> {
        let mut query_pools = [vk::QueryPool::null(); QUERY_POOL_COUNT];

        if timestamps_supported && timestamp_period_ns > 0.0 {
            let create_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(MAX_TIMESTAMPS);

            for pool in &mut query_pools {
                *pool = device.create_query_pool(&create_info, None)?;
            }

            log::info!(
                "GPU profiler initialized (period: {timestamp_period_ns:.3}ns, {QUERY_POOL_COUNT} pools)"
            );
        } else {
            log::warn!("GPU timestamps not supported on this device");
        }

        Ok(Self {
            device,
            query_pools,
            current_pool: 0,
            timestamp_period_ns: timestamp_period_ns as f64,
            timestamps_supported: timestamps_supported && timestamp_period_ns > 0.0,
            last_results: ExtendedGpuTimings::default(),
            frames_since_result: 0,
            total_frames: 0,
        })
    }

    /// Check if timestamps are supported
    pub fn is_supported(&self) -> bool {
        self.timestamps_supported
    }

    /// Begin a new frame - reset queries and swap pools
    ///
    /// # Safety
    /// Command buffer must be in recording state
    pub unsafe fn begin_frame(&mut self, cmd: vk::CommandBuffer) {
        if !self.timestamps_supported {
            return;
        }

        // Swap to next pool
        self.current_pool = (self.current_pool + 1) % QUERY_POOL_COUNT;
        let pool = self.query_pools[self.current_pool];

        // Reset queries for this frame
        self.device
            .cmd_reset_query_pool(cmd, pool, 0, MAX_TIMESTAMPS);

        // Write start timestamp
        self.write_timestamp(cmd, TimingScope::FrameStart);
        self.total_frames += 1;
    }

    /// Write a timestamp for a timing scope
    ///
    /// # Safety
    /// Command buffer must be in recording state
    pub unsafe fn write_timestamp(&self, cmd: vk::CommandBuffer, scope: TimingScope) {
        if !self.timestamps_supported {
            return;
        }

        let pool = self.query_pools[self.current_pool];
        self.device.cmd_write_timestamp(
            cmd,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            pool,
            scope.index(),
        );
    }

    /// End frame and collect results from previous frame's pool (non-blocking)
    ///
    /// Returns cached results if new data isn't available yet.
    pub fn end_frame(&mut self) -> GpuTimings {
        self.end_frame_extended().to_basic()
    }

    /// End frame with extended timing data (non-blocking)
    pub fn end_frame_extended(&mut self) -> ExtendedGpuTimings {
        if !self.timestamps_supported {
            return ExtendedGpuTimings::default();
        }

        // Read from the OTHER pool (previous frame)
        let read_pool_idx = (self.current_pool + 1) % QUERY_POOL_COUNT;
        let pool = self.query_pools[read_pool_idx];

        let mut timestamps = [0u64; MAX_TIMESTAMPS as usize];

        // Try non-blocking read first
        let result = unsafe {
            self.device.get_query_pool_results(
                pool,
                0,
                &mut timestamps[..=TimingScope::FrameEnd.index() as usize],
                vk::QueryResultFlags::TYPE_64, // No WAIT flag = non-blocking
            )
        };

        match result {
            Ok(_) => {
                // Check if we have valid data (FrameEnd should be non-zero)
                if timestamps[TimingScope::FrameEnd.index() as usize] > 0 {
                    self.last_results = self.compute_extended_timings(&timestamps);
                    self.last_results.valid = true;
                    self.frames_since_result = 0;
                } else {
                    self.frames_since_result += 1;
                }
            }
            Err(_) => {
                // Results not ready yet, keep cached values
                self.frames_since_result += 1;
                self.last_results.valid = self.frames_since_result < 10;
            }
        }

        self.last_results.clone()
    }

    /// Compute extended timing deltas from raw timestamps
    fn compute_extended_timings(&self, timestamps: &[u64]) -> ExtendedGpuTimings {
        let to_ms = |start: u64, end: u64| -> f32 {
            if end > start {
                ((end - start) as f64 * self.timestamp_period_ns / 1_000_000.0) as f32
            } else {
                0.0
            }
        };

        let get = |scope: TimingScope| timestamps.get(scope.index() as usize).copied().unwrap_or(0);

        let frame_start = get(TimingScope::FrameStart);
        let shadow_end = get(TimingScope::ShadowEnd);
        let scene_end = get(TimingScope::SceneEnd);
        let bloom_thresh_end = get(TimingScope::BloomThresholdEnd);
        let bloom_down_end = get(TimingScope::BloomDownsampleEnd);
        let bloom_up_end = get(TimingScope::BloomUpsampleEnd);
        let post_end = get(TimingScope::PostProcessEnd);
        let ui_end = get(TimingScope::UiEnd);
        let frame_end = get(TimingScope::FrameEnd);

        // Use previous valid timestamp if current is 0 (scope wasn't recorded)
        let shadow_start = frame_start;
        let scene_start = if shadow_end > 0 {
            shadow_end
        } else {
            frame_start
        };
        let bloom_thresh_start = if scene_end > 0 {
            scene_end
        } else {
            scene_start
        };
        let bloom_down_start = if bloom_thresh_end > 0 {
            bloom_thresh_end
        } else {
            bloom_thresh_start
        };
        let bloom_up_start = if bloom_down_end > 0 {
            bloom_down_end
        } else {
            bloom_down_start
        };
        let post_start = if bloom_up_end > 0 {
            bloom_up_end
        } else {
            bloom_up_start
        };
        let ui_start = if post_end > 0 { post_end } else { post_start };

        ExtendedGpuTimings {
            total_ms: to_ms(frame_start, frame_end),
            shadow_ms: to_ms(
                shadow_start,
                if shadow_end > 0 {
                    shadow_end
                } else {
                    shadow_start
                },
            ),
            scene_ms: to_ms(
                scene_start,
                if scene_end > 0 {
                    scene_end
                } else {
                    scene_start
                },
            ),
            bloom_threshold_ms: to_ms(
                bloom_thresh_start,
                if bloom_thresh_end > 0 {
                    bloom_thresh_end
                } else {
                    bloom_thresh_start
                },
            ),
            bloom_downsample_ms: to_ms(
                bloom_down_start,
                if bloom_down_end > 0 {
                    bloom_down_end
                } else {
                    bloom_down_start
                },
            ),
            bloom_upsample_ms: to_ms(
                bloom_up_start,
                if bloom_up_end > 0 {
                    bloom_up_end
                } else {
                    bloom_up_start
                },
            ),
            post_process_ms: to_ms(post_start, if post_end > 0 { post_end } else { post_start }),
            ui_ms: to_ms(ui_start, if ui_end > 0 { ui_end } else { ui_start }),
            valid: true,
        }
    }

    /// Get last frame's GPU timings (basic)
    pub fn last_timings(&self) -> GpuTimings {
        self.last_results.to_basic()
    }

    /// Get last frame's extended GPU timings
    pub fn last_extended_timings(&self) -> &ExtendedGpuTimings {
        &self.last_results
    }

    /// Get total frames profiled
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Get frames since last successful result retrieval
    pub fn frames_since_result(&self) -> u32 {
        self.frames_since_result
    }
}

impl Drop for GpuProfiler {
    fn drop(&mut self) {
        if self.timestamps_supported {
            unsafe {
                for pool in &self.query_pools {
                    if *pool != vk::QueryPool::null() {
                        self.device.destroy_query_pool(*pool, None);
                    }
                }
            }
            log::info!(
                "GPU profiler destroyed (profiled {} frames)",
                self.total_frames
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_scope_indices() {
        assert_eq!(TimingScope::FrameStart.index(), 0);
        assert_eq!(TimingScope::FrameEnd.index(), 8);
    }

    #[test]
    fn test_extended_timings_to_basic() {
        let ext = ExtendedGpuTimings {
            total_ms: 10.0,
            shadow_ms: 1.0,
            scene_ms: 2.0,
            bloom_threshold_ms: 0.5,
            bloom_downsample_ms: 0.5,
            bloom_upsample_ms: 0.5,
            post_process_ms: 1.0,
            ui_ms: 0.5,
            valid: true,
        };
        let basic = ext.to_basic();
        assert_eq!(basic.total_ms, 10.0);
        assert_eq!(basic.scene_ms, 3.0); // shadow + scene
        assert_eq!(basic.post_process_ms, 2.5); // bloom + post
        assert_eq!(basic.ui_ms, 0.5);
    }
}
