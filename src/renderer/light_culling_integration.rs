//! Light Culling Integration for Renderer
//!
//! This module provides a clean integration point for GPU light culling.
//! Add this to your render loop for Forward+ lighting.
//!
//! # Usage
//! ```ignore
//! // In renderer initialization:
//! let light_culling = LightCullingIntegration::new(&vulkan_device, &allocator, extent)?;
//!
//! // Each frame, before main render pass:
//! light_culling.update_lights(&lighting_config);
//! light_culling.update_camera(&view, &projection, camera_pos);
//! light_culling.dispatch(command_buffer, extent);
//!
//! // The tile buffer is now ready for fragment shader binding
//! ```

use ash::vk;
use glam::{Mat4, Vec3};

use crate::renderer::features::light_culling::{
    CullingCameraData, GpuLight, LightCullingPass, LightCullingPushConstants,
};
use crate::renderer::features::{DirectionalLight, LightingConfig, PointLight};
use crate::vulkan::{ShaderModule, VulkanDevice};
use crate::Result;

/// Light culling integration for the main renderer
///
/// Provides a simplified API for GPU-based light culling.
pub struct LightCullingIntegration {
    /// Light culling state manager
    pass: LightCullingPass,
    /// Shader module (if loaded)
    shader_module: Option<vk::ShaderModule>,
}

impl LightCullingIntegration {
    /// Create a new light culling integration
    pub fn new() -> Self {
        Self {
            pass: LightCullingPass::new(),
            shader_module: None,
        }
    }

    /// Load the light culling compute shader
    ///
    /// Call this during renderer initialization.
    pub fn load_shader(&mut self, device: &VulkanDevice) -> Result<()> {
        let code = include_bytes!("../../shaders/light_culling.spv");
        let shader =
            ShaderModule::load_from_bytes(&device.device, code, vk::ShaderStageFlags::COMPUTE)?;

        self.shader_module = Some(shader.module);
        // Note: shader will be dropped, but module handle is Copy
        log::info!("Light culling shader loaded (embedded)");
        Ok(())
    }

    /// Check if light culling is available
    pub fn is_available(&self) -> bool {
        self.shader_module.is_some()
    }

    /// Update screen size (call on resize)
    pub fn resize(&mut self, width: u32, height: u32) {
        self.pass.calculate_tiles(width, height);
    }

    /// Update lights from lighting config
    pub fn update_lights(&mut self, config: &LightingConfig) {
        self.pass
            .update_lights(&config.point_lights, &config.directional_lights);
    }

    /// Update lights directly
    pub fn update_lights_direct(
        &mut self,
        point_lights: &[PointLight],
        directional_lights: &[DirectionalLight],
    ) {
        self.pass.update_lights(point_lights, directional_lights);
    }

    /// Get GPU light data for upload
    pub fn get_light_data(&self) -> &[GpuLight] {
        self.pass.get_light_buffer_data()
    }

    /// Get number of active lights
    pub fn light_count(&self) -> usize {
        self.pass.light_count()
    }

    /// Get push constants for current state
    pub fn get_push_constants(
        &self,
        screen_width: u32,
        screen_height: u32,
    ) -> LightCullingPushConstants {
        self.pass.get_push_constants(screen_width, screen_height)
    }

    /// Get dispatch dimensions
    pub fn get_dispatch_dimensions(&self) -> (u32, u32, u32) {
        self.pass.get_dispatch_dimensions()
    }

    /// Create camera data from matrices
    pub fn create_camera_data(
        view: &Mat4,
        projection: &Mat4,
        camera_pos: Vec3,
    ) -> CullingCameraData {
        CullingCameraData {
            view: view.to_cols_array_2d(),
            projection: projection.to_cols_array_2d(),
            inv_projection: projection.inverse().to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
        }
    }

    /// Check if culling should run this frame
    pub fn should_run(&self) -> bool {
        self.pass.is_enabled() && self.shader_module.is_some()
    }

    /// Get shader module handle
    pub fn shader_module(&self) -> Option<vk::ShaderModule> {
        self.shader_module
    }
}

impl Default for LightCullingIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Example of how to integrate into render loop
///
/// This is documentation showing the integration pattern.
///
/// ```ignore
/// // In Renderer struct, add:
/// light_culling: LightCullingIntegration,
/// light_culling_pipeline: Option<LightCullingPipeline>,
///
/// // In Renderer::new(), add:
/// let mut light_culling = LightCullingIntegration::new();
/// light_culling.load_shader(&vulkan_device)?;
/// light_culling.resize(swapchain.extent.width, swapchain.extent.height);
///
/// // Create pipeline if shader loaded:
/// let light_culling_pipeline = if light_culling.is_available() {
///     Some(LightCullingPipeline::new(
///         &vulkan_device,
///         light_culling.shader_module().unwrap(),
///         depth_sampler,
///         depth_image_view,
///         swapchain.extent.width,
///         swapchain.extent.height,
///     )?)
/// } else {
///     None
/// };
///
/// // In render_frame(), before main render pass:
/// if let Some(ref pipeline) = self.light_culling_pipeline {
///     if self.light_culling.should_run() {
///         // Update lights from lighting feature
///         self.light_culling.update_lights(lighting_config);
///         
///         // Upload to GPU
///         pipeline.upload_lights(self.light_culling.get_light_data())?;
///         
///         // Update camera
///         let camera_data = LightCullingIntegration::create_camera_data(
///             &view_matrix, &projection_matrix, camera_pos
///         );
///         pipeline.upload_camera(&camera_data)?;
///         
///         // Dispatch compute shader
///         let (tiles_x, tiles_y, _) = self.light_culling.get_dispatch_dimensions();
///         let push_constants = self.light_culling.get_push_constants(
///             swapchain.extent.width,
///             swapchain.extent.height
///         );
///         pipeline.dispatch(command_buffer, tiles_x, tiles_y, &push_constants);
///         
///         // Memory barrier before fragment shader reads
///         let barrier = vk::MemoryBarrier::builder()
///             .src_access_mask(vk::AccessFlags::SHADER_WRITE)
///             .dst_access_mask(vk::AccessFlags::SHADER_READ)
///             .build();
///         device.cmd_pipeline_barrier(
///             command_buffer,
///             vk::PipelineStageFlags::COMPUTE_SHADER,
///             vk::PipelineStageFlags::FRAGMENT_SHADER,
///             vk::DependencyFlags::empty(),
///             &[barrier],
///             &[],
///             &[],
///         );
///     }
/// }
/// ```
pub mod integration_example {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_creation() {
        let integration = LightCullingIntegration::new();
        assert!(!integration.is_available());
        assert_eq!(integration.light_count(), 0);
    }

    #[test]
    fn test_camera_data() {
        let view = Mat4::IDENTITY;
        let proj = Mat4::perspective_lh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let pos = Vec3::new(0.0, 5.0, 10.0);

        let data = LightCullingIntegration::create_camera_data(&view, &proj, pos);
        assert_eq!(data.camera_pos[1], 5.0);
    }
}
