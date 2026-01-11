//! BRDF LUT Pass
//!
//! Generates a 2D BRDF lookup table for the split-sum approximation.
//! This is a one-time bake at engine startup.

use ash::vk;
use std::sync::Arc;

/// BRDF LUT configuration
#[derive(Debug, Clone, Copy)]
pub struct BrdfLutConfig {
    /// Resolution of the LUT (default 512)
    pub resolution: u32,
    /// Format for the LUT texture (default R16G16_SFLOAT)
    pub format: vk::Format,
}

impl Default for BrdfLutConfig {
    fn default() -> Self {
        Self {
            resolution: 512,
            format: vk::Format::R16G16_SFLOAT,
        }
    }
}

/// BRDF LUT Baker
///
/// Renders the BRDF integration lookup table once at startup.
/// The output is a 2D texture where:
/// - X axis = NdotV (0-1)
/// - Y axis = Roughness (0-1)
/// - Output = vec2(scale, bias) for F0 * scale + bias
pub struct BrdfLutPass {
    config: BrdfLutConfig,
    /// The baked LUT image (None until baked)
    lut_image: Option<crate::renderer::resources::ImageHandle>,
    /// Whether the LUT has been baked
    baked: bool,
}

impl BrdfLutPass {
    /// Create a new BRDF LUT pass with default config
    pub fn new() -> Self {
        Self {
            config: BrdfLutConfig::default(),
            lut_image: None,
            baked: false,
        }
    }

    /// Create with custom config
    pub fn with_config(config: BrdfLutConfig) -> Self {
        let mut pass = Self::new();
        pass.config = config;
        pass
    }

    /// Get config
    pub fn config(&self) -> &BrdfLutConfig {
        &self.config
    }

    /// Has the LUT been baked?
    pub fn is_baked(&self) -> bool {
        self.baked
    }

    /// Get the LUT image view for binding
    pub fn get_lut_view(&self) -> Option<vk::ImageView> {
        self.lut_image.as_ref().map(|img| img.view())
    }

    /// Get the LUT image for transitions
    pub fn get_lut_image(&self) -> Option<vk::Image> {
        self.lut_image.as_ref().map(|img| img.handle())
    }

    /// Get LUT resolution
    pub fn resolution(&self) -> u32 {
        self.config.resolution
    }

    // =========================================================================
    // GPU Resource Management
    // =========================================================================

    /// Create the BRDF LUT image handle.
    ///
    /// # Safety
    /// Valid Vulkan allocator and device required.
    pub unsafe fn create_image(
        &mut self,
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
    ) -> crate::Result<()> {
        let image = crate::renderer::resources::ImageHandle::create_brdf_lut(
            device,
            allocator,
            self.config.resolution,
        )?;

        self.lut_image = Some(image);
        Ok(())
    }

    /// Bake the BRDF LUT using a temporary render pass and pipeline.
    ///
    /// # Safety
    /// Valid Vulkan context required.
    pub unsafe fn bake(
        &mut self,
        device: &crate::vulkan::VulkanDevice,
        command_pool: vk::CommandPool,
        allocator: Arc<crate::vulkan::Allocator>,
    ) -> crate::Result<()> {
        if self.baked {
            return Ok(());
        }

        if self.lut_image.is_none() {
            self.create_image(Arc::clone(&allocator), Arc::clone(&device.device))?;
        }

        let lut_image = self.lut_image.as_ref().unwrap();
        let res = self.config.resolution;

        // 1. Create temporary Render Pass
        let render_pass = crate::vulkan::RenderPass::builder(Arc::clone(&device.device))
            .with_color_attachment(
                self.config.format,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
            .build()?;

        // 2. Create Framebuffer
        let framebuffer = crate::vulkan::framebuffer::Framebuffer::new(
            Arc::clone(&device.device),
            render_pass.handle(),
            &[lut_image.view()],
            vk::Extent2D {
                width: res,
                height: res,
            },
        )?;

        // 3. Create Pipeline Layout
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let layout = device.device.create_pipeline_layout(&layout_info, None)?;

        // 4. Create Pipeline
        // Use pre-compiled shaders if available, otherwise fallback to our generated ones
        // In a real project, we'd use the ones from the shaders directory.
        let pipeline = crate::vulkan::pipeline::Pipeline::builder(Arc::clone(&device.device))
            .with_layout(layout)
            .with_render_pass(render_pass.handle())
            .with_extent(vk::Extent2D {
                width: res,
                height: res,
            })
            .add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/postprocess.vert.spv")),
                vk::ShaderStageFlags::VERTEX,
                "main",
            )?
            .add_shader_from_bytes(
                include_bytes!(concat!(env!("OUT_DIR"), "/brdf_lut.frag.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                "main",
            )?
            .with_vertex_input(vec![], vec![]) // Procedural vertices
            .with_cull_mode(vk::CullModeFlags::NONE)
            .build()?;

        // 5. Render
        device.execute_single_use(command_pool, |cmd| {
            // Empty command buffer for testing
            // If this passes, the issue is inside the commands themselves.
            // log::info!("BrdfLutPass: Recording empty command buffer");

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass.handle())
                .framebuffer(framebuffer.handle())
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: res,
                        height: res,
                    },
                })
                .clear_values(&clear_values);

            device.device.cmd_begin_render_pass(
                cmd,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );

            // CRITICAL FIX: PipelineBuilder enables dynamic VIEWPORT and SCISSOR.
            // We must set them before drawing.
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: res as f32,
                height: res as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: res,
                    height: res,
                },
            }];

            device.device.cmd_set_viewport(cmd, 0, &viewports);
            device.device.cmd_set_scissor(cmd, 0, &scissors);

            device.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            device.device.cmd_draw(cmd, 3, 1, 0, 0);
            device.device.cmd_end_render_pass(cmd);
        })?;

        // 6. Cleanup temporary resources
        device.device.destroy_pipeline_layout(layout, None);
        // Pipeline, RenderPass, and Framebuffer are dropped automatically (RAII)

        self.baked = true;
        log::info!("BRDF LUT baked successfully ({res}x{res})");

        Ok(())
    }

    /// Mark as baked (called after render pass completes)
    pub fn mark_baked(&mut self) {
        self.baked = true;
        log::info!("BrdfLutPass: LUT baked successfully");
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use by GPU.
    pub unsafe fn destroy(&mut self, _allocator: &vk_mem::Allocator, _device: &ash::Device) {
        self.lut_image = None; // ImageHandle handles cleanup
        self.baked = false;
        log::info!("BrdfLutPass: Destroyed resources");
    }
}

impl Drop for BrdfLutPass {
    fn drop(&mut self) {
        // Cleanup requires device and allocator which may not be available during drop
        log::debug!("BrdfLutPass: Drop called (cleanup requires explicit destroy)");
    }
}

impl Default for BrdfLutPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let pass = BrdfLutPass::new();
        assert_eq!(pass.config().resolution, 512);
        assert!(!pass.is_baked());
        assert!(pass.get_lut_view().is_none());
    }

    #[test]
    fn test_custom_resolution() {
        let pass = BrdfLutPass::with_config(BrdfLutConfig {
            resolution: 256,
            ..Default::default()
        });
        assert_eq!(pass.resolution(), 256);
    }
}
