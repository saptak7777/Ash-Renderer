//! Shadow mapping for directional lights
//!
//! Provides depth-based shadow mapping with PCF soft shadows.

use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

/// Shadow map configuration
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Shadow map resolution (width = height)
    pub resolution: u32,
    /// Depth bias to prevent shadow acne
    pub depth_bias: f32,
    /// Slope-scaled depth bias
    pub slope_bias: f32,
    /// PCF filter size (1 = hard shadows, 2+ = soft)
    pub pcf_size: u32,
    /// Enable/disable shadows
    pub enabled: bool,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            resolution: 2048,
            depth_bias: 0.005,
            slope_bias: 1.5,
            pcf_size: 3,
            enabled: true,
        }
    }
}

/// Shadow map for a directional light
pub struct ShadowMap {
    device: Arc<ash::Device>,
    /// Depth image for shadow map
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_memory: vk::DeviceMemory,
    /// Render pass for depth-only rendering
    pub render_pass: vk::RenderPass,
    /// Framebuffer for shadow pass
    pub framebuffer: vk::Framebuffer,
    /// Sampler for shadow sampling with comparison
    pub sampler: vk::Sampler,
    /// Resolution
    pub resolution: u32,
    /// Light-space matrix (view * projection from light's POV)
    pub light_space_matrix: glam::Mat4,
    /// Configuration
    pub config: ShadowConfig,
}

impl ShadowMap {
    /// Create a new shadow map
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of this shadow map.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        config: ShadowConfig,
    ) -> Result<Self> {
        let resolution = config.resolution;
        log::info!("[ShadowMap] Creating {resolution}x{resolution} shadow map");

        // Create depth image
        let depth_format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(depth_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let depth_image = device
            .create_image(&image_info, None)
            .map_err(|e| AshError::VulkanError(format!("Shadow depth image failed: {e}")))?;

        // Allocate memory
        let mem_requirements = device.get_image_memory_requirements(depth_image);
        let memory_type_index = find_memory_type(
            &memory_properties,
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| AshError::VulkanError("No suitable memory type".to_string()))?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let depth_memory = device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| AshError::VulkanError(format!("Shadow memory alloc failed: {e}")))?;

        device
            .bind_image_memory(depth_image, depth_memory, 0)
            .map_err(|e| AshError::VulkanError(format!("Bind shadow memory failed: {e}")))?;

        // Create depth image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let depth_image_view = device
            .create_image_view(&view_info, None)
            .map_err(|e| AshError::VulkanError(format!("Shadow image view failed: {e}")))?;

        // Create render pass (depth-only)
        let depth_attachment = vk::AttachmentDescription {
            format: depth_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ..Default::default()
        };

        let depth_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref);

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let attachments = [depth_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| AshError::VulkanError(format!("Shadow render pass failed: {e}")))?;

        // Create framebuffer
        let attachments = [depth_image_view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(resolution)
            .height(resolution)
            .layers(1);

        let framebuffer = device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| AshError::VulkanError(format!("Shadow framebuffer failed: {e}")))?;

        // Create sampler for shadow map sampling (manual PCF)
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .compare_enable(false) // Disable for manual PCF in shader
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = device
            .create_sampler(&sampler_info, None)
            .map_err(|e| AshError::VulkanError(format!("Shadow sampler failed: {e}")))?;

        log::info!("[ShadowMap] Shadow map created successfully");

        Ok(Self {
            device,
            depth_image,
            depth_image_view,
            depth_memory,
            render_pass,
            framebuffer,
            sampler,
            resolution,
            light_space_matrix: glam::Mat4::IDENTITY,
            config,
        })
    }

    /// Update light-space matrix for directional light
    pub fn update_light_matrix(
        &mut self,
        light_dir: glam::Vec3,
        scene_center: glam::Vec3,
        scene_radius: f32,
    ) {
        // Light position (far from scene, looking at center)
        let light_pos = scene_center - light_dir.normalize() * scene_radius * 2.0;

        // View matrix from light's perspective
        let light_view = glam::Mat4::look_at_rh(light_pos, scene_center, glam::Vec3::Y);

        // Orthographic projection to cover the scene
        let light_proj = glam::Mat4::orthographic_rh(
            -scene_radius,
            scene_radius,
            -scene_radius,
            scene_radius,
            0.1,
            scene_radius * 4.0,
        );

        self.light_space_matrix = light_proj * light_view;
    }

    /// Get the viewport for shadow rendering
    pub fn viewport(&self) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.resolution as f32,
            height: self.resolution as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Get the scissor for shadow rendering
    pub fn scissor(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.resolution,
                height: self.resolution,
            },
        }
    }
}

impl Drop for ShadowMap {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_memory, None);
            log::info!("[ShadowMap] Shadow map destroyed");
        }
    }
}

/// Find a suitable memory type
fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..properties.memory_type_count {
        let type_bits = 1 << i;
        let has_properties = properties.memory_types[i as usize]
            .property_flags
            .contains(required);

        if (type_filter & type_bits) != 0 && has_properties {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_config_default() {
        let config = ShadowConfig::default();
        assert_eq!(config.resolution, 2048);
        assert_eq!(config.pcf_size, 3);
        assert!(config.enabled);
    }

    #[test]
    fn test_light_matrix_calculation() {
        let mut shadow = MockShadowMap::new();
        shadow.update_light_matrix(glam::Vec3::new(-1.0, -1.0, -1.0), glam::Vec3::ZERO, 10.0);
        assert_ne!(shadow.light_space_matrix, glam::Mat4::IDENTITY);
    }

    struct MockShadowMap {
        light_space_matrix: glam::Mat4,
    }

    impl MockShadowMap {
        fn new() -> Self {
            Self {
                light_space_matrix: glam::Mat4::IDENTITY,
            }
        }

        fn update_light_matrix(
            &mut self,
            light_dir: glam::Vec3,
            scene_center: glam::Vec3,
            scene_radius: f32,
        ) {
            let light_pos = scene_center - light_dir.normalize() * scene_radius * 2.0;
            let light_view = glam::Mat4::look_at_rh(light_pos, scene_center, glam::Vec3::Y);
            let light_proj = glam::Mat4::orthographic_rh(
                -scene_radius,
                scene_radius,
                -scene_radius,
                scene_radius,
                0.1,
                scene_radius * 4.0,
            );
            self.light_space_matrix = light_proj * light_view;
        }
    }
}
