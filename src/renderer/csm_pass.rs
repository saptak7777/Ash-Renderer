//! Cascaded Shadow Maps (CSM) implementation for directional lights
//!
//! Provides industry-standard cascaded shadow mapping with:
//! - Multi-resolution cascades (4 levels)
//! - Texture array storage for efficient GPU access
//! - Per-cascade frustum culling
//! - Stable shadow projection (texel-snapped)

use ash::vk;
use std::sync::Arc;

use crate::renderer::resources::shadow::{CascadeData, CascadedShadowMap, CsmConfig, MAX_CASCADES};
use crate::vulkan::Allocator;
use crate::AshError;

/// CSM rendering pass - owns GPU resources for cascaded shadow mapping
pub struct CsmPass {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,

    /// Shadow map texture array (4 layers, D32_SFLOAT)
    pub depth_array: vk::Image,
    depth_array_alloc: Option<vk_mem::Allocation>,

    /// Full array view (for sampling in shaders)
    pub depth_array_view: vk::ImageView,

    /// Per-layer views (for framebuffer attachments)
    pub layer_views: Vec<vk::ImageView>,

    /// Framebuffers (one per cascade)
    pub framebuffers: Vec<vk::Framebuffer>,

    /// Render pass (depth-only)
    pub render_pass: vk::RenderPass,

    /// Sampler for shadow sampling
    pub sampler: vk::Sampler,

    /// Resolution per cascade
    pub resolution: u32,

    /// Cascade manager (CPU-side logic)
    pub csm: CascadedShadowMap,
}

impl CsmPass {
    /// Create a new CSM pass
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this pass.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        config: CsmConfig,
    ) -> Result<Self, AshError> {
        let resolution = config.resolution;

        log::info!("Creating CSM pass with {resolution}x{resolution} per cascade");

        // Create depth array image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width: resolution,
                height: resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(MAX_CASCADES as u32)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (depth_array, depth_array_alloc) = allocator
            .create_image(&image_info, vk_mem::MemoryUsage::AutoPreferDevice)
            .map_err(|e| {
                AshError::VulkanError(format!("CSM depth array creation failed: {e:?}"))
            })?;

        // Create full array view
        let array_view_info = vk::ImageViewCreateInfo::default()
            .image(depth_array)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: MAX_CASCADES as u32,
            });

        let depth_array_view = device
            .create_image_view(&array_view_info, None)
            .map_err(|e| AshError::VulkanError(format!("CSM array view creation failed: {e:?}")))?;

        // Create per-layer views
        let mut layer_views = Vec::with_capacity(MAX_CASCADES);
        for layer in 0..MAX_CASCADES {
            let layer_view_info = vk::ImageViewCreateInfo::default()
                .image(depth_array)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: layer as u32,
                    layer_count: 1,
                });

            let view = device
                .create_image_view(&layer_view_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("CSM layer view {layer} creation failed: {e:?}"))
                })?;

            layer_views.push(view);
        }

        // Create render pass (depth-only)
        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
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
            .map_err(|e| {
                AshError::VulkanError(format!("CSM render pass creation failed: {e:?}"))
            })?;

        // Create framebuffers (one per cascade)
        let mut framebuffers = Vec::with_capacity(MAX_CASCADES);
        for (i, layer_view) in layer_views.iter().enumerate() {
            let attachments = [*layer_view];
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(resolution)
                .height(resolution)
                .layers(1);

            let framebuffer = device
                .create_framebuffer(&framebuffer_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("CSM framebuffer {i} creation failed: {e:?}"))
                })?;

            framebuffers.push(framebuffer);
        }

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .compare_enable(false) // Manual PCF in shader
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = device
            .create_sampler(&sampler_info, None)
            .map_err(|e| AshError::VulkanError(format!("CSM sampler creation failed: {e:?}")))?;

        let csm = CascadedShadowMap::new(config);

        log::info!("CSM pass created successfully");

        Ok(Self {
            device,
            allocator,
            depth_array,
            depth_array_alloc: Some(depth_array_alloc),
            depth_array_view,
            layer_views,
            framebuffers,
            render_pass,
            sampler,
            resolution,
            csm,
        })
    }

    /// Update cascade matrices based on camera
    pub fn update_cascades(
        &mut self,
        camera_view: &glam::Mat4,
        camera_proj: &glam::Mat4,
        light_dir: glam::Vec3,
        shadow_distance: f32,
    ) {
        self.csm
            .update(camera_view, camera_proj, light_dir, shadow_distance);
    }

    /// Get cascade data for a specific cascade index
    pub fn cascade(&self, index: usize) -> Option<&CascadeData> {
        self.csm.cascade(index)
    }

    /// Render shadows for all cascades
    ///
    /// # Safety
    /// Command buffer must be in recording state. Pipeline and descriptor sets must be valid.
    pub unsafe fn render_shadows<F>(
        &self,
        cmd: vk::CommandBuffer,
        device: &ash::Device,
        shadow_pipeline: vk::Pipeline,
        _pipeline_layout: vk::PipelineLayout,
        mut draw_fn: F,
    ) where
        F: FnMut(vk::CommandBuffer, &glam::Mat4, usize),
    {
        let clear_values = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];

        // Render each cascade
        for cascade_idx in 0..self.csm.cascade_count() {
            if let Some(cascade) = self.csm.cascade(cascade_idx) {
                // Begin render pass for this cascade
                let render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[cascade_idx])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: self.resolution,
                            height: self.resolution,
                        },
                    })
                    .clear_values(&clear_values);

                device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);

                // Bind shadow pipeline
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);

                // Set viewport and scissor
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.resolution as f32,
                    height: self.resolution as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cmd, 0, &[viewport]);

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: self.resolution,
                        height: self.resolution,
                    },
                };
                device.cmd_set_scissor(cmd, 0, &[scissor]);

                // Call user-provided draw function with light space matrix
                draw_fn(cmd, &cascade.light_space_matrix, cascade_idx);

                device.cmd_end_render_pass(cmd);
            }
        }
    }

    /// Destroy CSM resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying CSM pass");

        // Destroy framebuffers
        for fb in self.framebuffers.drain(..) {
            self.device.destroy_framebuffer(fb, None);
        }

        // Destroy render pass
        if self.render_pass != vk::RenderPass::null() {
            self.device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }

        // Destroy sampler
        if self.sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }

        // Destroy layer views
        for view in self.layer_views.drain(..) {
            self.device.destroy_image_view(view, None);
        }

        // Destroy array view
        if self.depth_array_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.depth_array_view, None);
            self.depth_array_view = vk::ImageView::null();
        }

        // Destroy depth array
        if let Some(mut alloc) = self.depth_array_alloc.take() {
            self.allocator
                .vma
                .destroy_image(self.depth_array, &mut alloc);
            self.depth_array = vk::Image::null();
        }

        log::debug!("CSM pass destroyed");
    }
}

impl Drop for CsmPass {
    fn drop(&mut self) {
        // Safety check - resources should be explicitly destroyed
        if self.depth_array_alloc.is_some() {
            log::error!("CsmPass dropped without calling destroy()! Resources leaked.");
        }
    }
}
