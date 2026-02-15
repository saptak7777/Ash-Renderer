use crate::renderer::passes::vsr::VsrConfig;
use crate::renderer::types::RendererConfig;
use crate::renderer::Renderer;
use crate::Result;
use ash::vk;

/// Fluent builder for the [`Renderer`].
pub struct RendererBuilder {
    config: RendererConfig,
}

impl RendererBuilder {
    pub fn new() -> Self {
        Self {
            config: RendererConfig::default(),
        }
    }

    /// Set whether VSync is enabled (FIFO) or disabled (IMMEDIATE/MAILBOX).
    pub fn with_vsync(mut self, enabled: bool) -> Self {
        self.config.present_mode = if enabled {
            vk::PresentModeKHR::FIFO
        } else {
            // Prefer Mailbox for low-latency non-tearing, fall back to Immediate
            vk::PresentModeKHR::MAILBOX
        };
        self
    }

    /// Set the VSM shadow map physical resolution.
    pub fn with_shadow_resolution(mut self, size: u32) -> Self {
        self.config.shadow_resolution = size;
        self
    }

    /// Set the VSR (Virtual Super Resolution) configuration.
    pub fn with_vsr(mut self, config: VsrConfig) -> Self {
        self.config.vsr_config = config;
        self
    }

    /// Set the desired resolution. If None, uses the window's physical size.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.config.resolution = Some((width, height));
        self
    }

    /// Enable or disable strict mode (errors on missing assets/materials).
    pub fn with_strict_mode(mut self, enabled: bool) -> Self {
        self.config.strict_mode = enabled;
        self
    }

    /// Finalize and build the [Renderer].
    pub fn build<S: crate::vulkan::SurfaceProvider>(
        self,
        surface_provider: &S,
    ) -> Result<Renderer> {
        Renderer::new_with_config(surface_provider, self.config)
    }
}

impl Default for RendererBuilder {
    fn default() -> Self {
        Self::new()
    }
}
