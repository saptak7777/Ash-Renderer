//! BRDF LUT Pass
//!
//! Generates a 2D BRDF lookup table for the split-sum approximation.
//! This is a one-time bake at engine startup.

use ash::vk;

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
    lut_image: Option<vk::Image>,
    /// Image view for the LUT
    lut_view: Option<vk::ImageView>,
    /// VMA allocation for the image
    allocation: Option<vk_mem::Allocation>,
    /// Whether the LUT has been baked
    baked: bool,
}

impl BrdfLutPass {
    /// Create a new BRDF LUT pass with default config
    pub fn new() -> Self {
        Self {
            config: BrdfLutConfig::default(),
            lut_image: None,
            lut_view: None,
            allocation: None,
            baked: false,
        }
    }

    /// Create with custom config
    pub fn with_config(config: BrdfLutConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
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
        self.lut_view
    }

    /// Get the LUT image for transitions
    pub fn get_lut_image(&self) -> Option<vk::Image> {
        self.lut_image
    }

    /// Get LUT resolution
    pub fn resolution(&self) -> u32 {
        self.config.resolution
    }

    // =========================================================================
    // GPU Resource Management
    // =========================================================================

    /// Create the LUT image
    ///
    /// # Safety
    /// Allocator and device must be valid.
    pub unsafe fn create_image(
        &mut self,
        allocator: &vk_mem::Allocator,
        device: &ash::Device,
    ) -> crate::Result<()> {
        use vk_mem::Alloc;

        let extent = vk::Extent3D {
            width: self.config.resolution,
            height: self.config.resolution,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(self.config.format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) =
            allocator
                .create_image(&image_info, &alloc_info)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("BRDF LUT image creation failed: {e:?}"))
                })?;

        // Create image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.config.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = device.create_image_view(&view_info, None)?;

        self.lut_image = Some(image);
        self.lut_view = Some(view);
        self.allocation = Some(allocation);

        log::info!(
            "BrdfLutPass: Created {}x{} LUT image",
            self.config.resolution,
            self.config.resolution
        );

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
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator, device: &ash::Device) {
        if let Some(view) = self.lut_view.take() {
            device.destroy_image_view(view, None);
        }
        if let (Some(image), Some(mut alloc)) = (self.lut_image.take(), self.allocation.take()) {
            allocator.destroy_image(image, &mut alloc);
        }
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
