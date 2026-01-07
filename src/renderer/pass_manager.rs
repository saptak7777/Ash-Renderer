
/// Defines the strategy for rendering geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderingMode {
    /// GPU-driven rendering: uses instancing + indirect draw calls.
    /// Best performance for high object counts.
    #[default]
    GPUDriven,
    /// Legacy rendering: uses direct draw calls.
    /// Fallback for older hardware or small scenes.
    Legacy,
    /// Hybrid mode: enables both paths simultaneously for debugging.
    /// WARNING: This will cause duplicate rendering of objects.
    Hybrid,
}

/// Orchestrates different render passes (Shadow, Forward, etc.)
/// and ensures proper coordination between rendering paths.
pub struct RenderPassManager {
    mode: RenderingMode,
}

impl RenderPassManager {
    pub fn new(mode: RenderingMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> RenderingMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: RenderingMode) {
        self.mode = mode;
    }

    /// Determines if GPU-driven rendering should be used.
    pub fn use_gpu_driven(&self) -> bool {
        matches!(self.mode, RenderingMode::GPUDriven | RenderingMode::Hybrid)
    }

    /// Determines if legacy direct rendering should be used.
    pub fn use_legacy(&self) -> bool {
        matches!(self.mode, RenderingMode::Legacy | RenderingMode::Hybrid)
    }
}
