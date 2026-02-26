//! VSM Shadow Pass - Renders geometry into physical cache pages

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::resources::VsmResources;

#[derive(Clone)]
pub struct ShadowPageRenderInfo<'a> {
    pub resources: &'a super::resources::VsmResources,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub vertex_addr: u64,
    pub index_addr: u64,
    pub object_addr: u64,
    pub pages: &'a [super::page_manager::PageToRender],
}

// ShadowPushConstants removed in favor of unified GpuPushConstants

/// VSM shadow rendering pass
pub struct VsmShadowPass {
    device: Arc<ash::Device>,

    /// Physical cache resolution (framebuffer size)
    physical_resolution: u32,

    /// Depth buffer for hardware depth testing
    depth_image: vk::Image,
    depth_image_alloc: Option<vk_mem::Allocation>,
    depth_view: vk::ImageView,

    /// Pipeline for shadow rendering
    shadow_pipeline_layout: Option<vk::PipelineLayout>,
    shadow_pipeline: Option<vk::Pipeline>,

    destroyed: bool,
}

impl VsmShadowPass {
    /// Create a new VSM shadow pass
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this pass.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        resources: &VsmResources,
    ) -> Result<Self> {
        log::info!("Creating VSM shadow pass");

        // Create depth buffer for hardware depth testing
        let physical_res = resources.config().physical_resolution;
        let depth_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width: physical_res,
                height: physical_res,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (depth_image, depth_image_alloc) =
            unsafe { allocator.create_image(&depth_info, vk_mem::MemoryUsage::AutoPreferDevice) }
                .map_err(|e| {
                AshError::VulkanError(format!("Failed to create VSM depth buffer: {e:?}"))
            })?;

        let depth_view_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let depth_view =
            unsafe { device.create_image_view(&depth_view_info, None) }.map_err(|e| {
                AshError::VulkanError(format!("Failed to create VSM depth view: {e:?}"))
            })?;

        log::info!("VSM shadow pass created successfully");

        Ok(Self {
            device,
            physical_resolution: 4096,
            depth_image,
            depth_image_alloc: Some(depth_image_alloc),
            depth_view,
            shadow_pipeline_layout: None,
            shadow_pipeline: None,
            destroyed: false,
        })
    }

    /// Create shadow rendering pipeline
    ///
    /// # Safety
    /// Device must remain valid. Descriptor set layouts must be valid.
    pub unsafe fn create_pipeline(
        &mut self,
        bindless_layout: vk::DescriptorSetLayout, // Set 0 (Combined Bindless)
    ) -> Result<()> {
        log::info!("Creating VSM shadow pipeline");

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            // Use the Rust struct size as the authoritative source so it can never drift.
            size: std::mem::size_of::<crate::renderer::types::GpuPushConstants>() as u32,
        };

        // Descriptor set layout: Set 0 (Combined Bindless)
        let layouts = [bindless_layout];

        // Create pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| {
                AshError::VulkanError(format!("Shadow pipeline layout creation failed: {e:?}"))
            })?;

        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv"));

        let pipeline = crate::vulkan::Pipeline::builder(Arc::clone(&self.device))
            .with_layout(pipeline_layout)
            .add_shader_from_bytes(vert_code, vk::ShaderStageFlags::VERTEX, "main")?
            .add_shader_from_bytes(frag_code, vk::ShaderStageFlags::FRAGMENT, "main")?
            .with_dynamic_rendering(
                &[vk::Format::R32G32_SFLOAT],
                Some(vk::Format::D32_SFLOAT),
                None,
            )
            .build()?;

        self.shadow_pipeline = Some(pipeline.pipeline);
        self.shadow_pipeline_layout = Some(pipeline_layout);

        log::info!("VSM shadow pipeline created successfully");
        Ok(())
    }

    /// Get shadow pipeline (if created)
    pub fn pipeline(&self) -> Option<vk::Pipeline> {
        self.shadow_pipeline
    }

    /// Get shadow pipeline layout (if created)
    pub fn pipeline_layout(&self) -> Option<vk::PipelineLayout> {
        self.shadow_pipeline_layout
    }

    /// Render shadows for specific pages using Dynamic Rendering
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn render_pages<F>(
        &self,
        cmd: vk::CommandBuffer,
        info: ShadowPageRenderInfo<'_>,
        mut scene_draw_fn: F,
    ) where
        F: FnMut(vk::CommandBuffer),
    {
        if info.pages.is_empty() {
            return;
        }

        log::debug!(
            "Rendering {} shadow pages via Dynamic Rendering",
            info.pages.len()
        );

        // 1. Transition layouts (Synchronization2)
        let cache_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(info.resources.physical_cache)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(info.resources.physical_depth_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let image_barriers = [cache_barrier, depth_barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);

        unsafe {
            self.device.cmd_pipeline_barrier2(cmd, &dependency_info);
        }

        // 2. Begin Dynamic Rendering
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(info.resources.physical_cache_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD) // Preserve atlas
            .store_op(vk::AttachmentStoreOp::STORE);

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(info.resources.physical_depth_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD) // Preserve atlas (we clear per-tile)
            .store_op(vk::AttachmentStoreOp::STORE);

        let color_attachments = [color_attachment];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.physical_resolution,
                    height: self.physical_resolution,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);
        }

        // 3. Bind Pipeline
        if let Some(pipeline) = self.shadow_pipeline {
            unsafe {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
        }

        // 4. Bind Bindless Texture Descriptor Set (Set 0)
        if let Some(layout) = self.shadow_pipeline_layout {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0, // Set 0
                    &[info.bindless_descriptor_set],
                    &[],
                );
            }
        }

        // Build the per-batch base push.
        let mut bda_push = crate::renderer::types::GpuPushConstants {
            frame_ptr: 0,
            vertex_ptr: info.vertex_addr,
            instance_ptr: info.object_addr,
            material_ptr: 0, // zero → frag skips alpha cutout branch
            index_ptr: info.index_addr,
            vsm_ptr: info.resources.metadata_address(),
            use_instancing: 1, // Enable instancing for the manual pull
            ..Default::default()
        };

        if let Some(layout) = self.shadow_pipeline_layout {
            // Initial push
            unsafe {
                self.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&bda_push),
                );
            }
        }

        // 6. Iterate over pages
        for page in info.pages {
            let page_size = info.resources.config().page_size;
            let x = (page.physical_coord.x * page_size as i32) as f32;
            let y = (page.physical_coord.y * page_size as i32) as f32;

            let viewport = vk::Viewport {
                x,
                y,
                width: page_size as f32,
                height: page_size as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };

            let scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D {
                    width: page_size,
                    height: page_size,
                },
            };

            unsafe {
                self.device.cmd_set_viewport(cmd, 0, &[viewport]);
                self.device.cmd_set_scissor(cmd, 0, &[scissor]);
            }

            // Clear just this tile's depth
            let clear_attachment = vk::ClearAttachment::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0, // Reverse-Z standard: 0.0 is far plane
                        stencil: 0,
                    },
                });

            let clear_rect = vk::ClearRect::default()
                .rect(scissor)
                .base_array_layer(0)
                .layer_count(1);

            unsafe {
                self.device
                    .cmd_clear_attachments(cmd, &[clear_attachment], &[clear_rect]);
            }

            // Per-page: update the entire structure (simplified for first pass)
            if let Some(layout) = self.shadow_pipeline_layout {
                bda_push.clipmap_level = page.clipmap_level;
                unsafe {
                    self.device.cmd_push_constants(
                        cmd,
                        layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&bda_push),
                    );
                }
            }

            // Draw call
            scene_draw_fn(cmd);
        }

        unsafe {
            self.device.cmd_end_rendering(cmd);
        }

        // 5. Transition Back (Synchronization2)
        let back_cache_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(info.resources.physical_cache)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&back_cache_barrier));

        unsafe {
            self.device.cmd_pipeline_barrier2(cmd, &dependency_info);
        }
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self, allocator: &Arc<Allocator>) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        log::debug!("Destroying VSM shadow pass");

        unsafe {
            if let Some(pipeline) = self.shadow_pipeline.take() {
                self.device.destroy_pipeline(pipeline, None);
            }

            if let Some(layout) = self.shadow_pipeline_layout.take() {
                self.device.destroy_pipeline_layout(layout, None);
            }

            // Destroy depth buffer
            if self.depth_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.depth_view, None);
                self.depth_view = vk::ImageView::null();
            }
            if let Some(mut alloc) = self.depth_image_alloc.take() {
                allocator.vma.destroy_image(self.depth_image, &mut alloc);
                self.depth_image = vk::Image::null();
            }
        }

        log::debug!("VSM shadow pass destroyed");
    }
}

impl Drop for VsmShadowPass {
    fn drop(&mut self) {
        if !self.destroyed {
            log::warn!("VsmShadowPass dropped without explicit destroy() call");
        }
    }
}
