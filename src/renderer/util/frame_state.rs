//! Per-frame temporal data provider.
//!
//! `FrameState` is the single source of truth for all per-frame metadata that
//! multiple passes need:
//!
//! - Camera matrices (view / jittered-projection / previous VP)
//! - Halton jitter for TAA / VSR
//! - Elapsed time for animations and effects
//! - Frame index for ring-buffer indexing
//!
//! Previously this data was scattered across `Resources::update_global_data`,
//! `systems.vsr_pass`, and `renderer.rs`. Centralising it here ensures all
//! passes see the *same* values every frame.

use super::halton::HaltonSequence;
use glam::{Mat4, Vec2, Vec3};

/// Centralised, per-frame temporal state.
///
/// Call [`FrameState::begin_frame`] once at the start of every frame (after
/// acquiring the swapchain image index). All passes should read their
/// projection and jitter data from this struct rather than computing their own.
#[derive(Debug)]
pub struct FrameState {
    // ── Camera matrices ────────────────────────────────────────────────────
    /// Raw (unjittered) view matrix set by the caller.
    pub view: Mat4,
    /// Raw (unjittered) projection matrix set by the caller.
    pub projection: Mat4,
    /// Jitter-modified projection for use in the current frame.
    pub jittered_projection: Mat4,
    /// `view * jittered_projection` from the *previous* frame.
    /// Used by motion-vector / temporal passes to detect camera movement.
    pub prev_view_proj: Mat4,
    /// Camera world-space position (for lighting, SSAO, etc.).
    pub camera_pos: Vec3,

    // ── Jitter ─────────────────────────────────────────────────────────────
    /// NDC-space sub-pixel jitter offset for the current frame.
    /// Centred in `[-0.5, 0.5]` (in pixels).
    pub jitter: Vec2,
    /// Jitter from the previous frame (for motion-vector reconstruction).
    pub prev_jitter: Vec2,

    // ── Time ───────────────────────────────────────────────────────────────
    /// Seconds elapsed since the renderer was created.
    pub elapsed_time: f32,
    /// Frame counter (monotonically increasing).
    pub frame_index: u64,

    // ── Internal ───────────────────────────────────────────────────────────
    halton: HaltonSequence,
}

impl FrameState {
    /// Create a new `FrameState` with Halton bases `(base_x, base_y)`.
    ///
    /// Standard TAA uses bases `(2, 3)` — pass those unless you have a reason
    /// to use a different low-discrepancy sequence.
    pub fn new(base_x: u32, base_y: u32) -> Self {
        Self {
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            jittered_projection: Mat4::IDENTITY,
            prev_view_proj: Mat4::IDENTITY,
            camera_pos: Vec3::ZERO,
            jitter: Vec2::ZERO,
            prev_jitter: Vec2::ZERO,
            elapsed_time: 0.0,
            frame_index: 0,
            halton: HaltonSequence::new(base_x, base_y),
        }
    }

    /// Advance temporal state for a new frame.
    ///
    /// This must be called **once per frame**, after the swapchain image is
    /// acquired but before any pass reads from this struct.
    ///
    /// # Parameters
    /// - `delta_time`  – Seconds since the last frame (for animations).
    /// - `view`        – Updated camera view matrix.
    /// - `projection`  – Raw (unjittered) projection matrix.
    /// - `camera_pos`  – Camera world position.
    /// - `width`/`height` – Render target dimensions (for NDC → pixel mapping).
    pub fn begin_frame(
        &mut self,
        delta_time: f32,
        view: Mat4,
        projection: Mat4,
        camera_pos: Vec3,
        width: u32,
        height: u32,
    ) {
        // Clamp delta time to 0.1s (10 FPS) to prevent physics explosions and TAA history corruption
        // when the window is dragged or a breakpoint is hit.
        let delta_time = delta_time.min(0.1);
        // ── Archive previous frame state ───────────────────────────────────
        self.prev_view_proj = self.jittered_projection * self.view;
        self.prev_jitter = self.jitter;

        // ── Advance counters ───────────────────────────────────────────────
        self.frame_index += 1;
        self.elapsed_time += delta_time;

        // ── Store camera ───────────────────────────────────────────────────
        self.view = view;
        self.projection = projection;
        self.camera_pos = camera_pos;

        // ── Compute Halton jitter and apply to projection ──────────────────
        // Guard against zero-sized render targets (minimized window).
        if width == 0 || height == 0 {
            self.jitter = Vec2::ZERO;
            self.jittered_projection = projection;
            return;
        }

        let sample = self.halton.next_sample(); // in [-0.5, 0.5]
        self.jitter = sample;

        // Convert sub-pixel jitter into NDC translation and inject into the
        // projection's translation column.  This is the standard TAA / VSR
        // jitter application method used by UE5 and Unity HDRP.
        let jitter_ndc_x = (sample.x * 2.0) / width as f32;
        let jitter_ndc_y = (sample.y * 2.0) / height as f32;

        let mut jittered = projection;
        *jittered.col_mut(2) =
            projection.col(2) + glam::Vec4::new(jitter_ndc_x, jitter_ndc_y, 0.0, 0.0);
        self.jittered_projection = jittered;
    }

    /// `jitter` as a 2-element array for GPU push constants / uniforms.
    pub fn jitter_uv(&self) -> [f32; 2] {
        [self.jitter.x, self.jitter.y]
    }

    /// Current `view_proj` (using the jittered projection).
    pub fn view_proj(&self) -> Mat4 {
        self.jittered_projection * self.view
    }

    /// Reset the Halton sequence and jitter state (use on camera cuts).
    pub fn reset_temporal_history(&mut self) {
        self.halton.reset();
        self.jitter = Vec2::ZERO;
        self.prev_jitter = Vec2::ZERO;
        self.prev_view_proj = self.projection * self.view;
    }
}

impl Default for FrameState {
    fn default() -> Self {
        // Standard Halton bases used by Unreal / Unity HDRP
        Self::new(2, 3)
    }
}
