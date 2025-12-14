use ash::vk;
use std::sync::Arc;

use super::utils;
use crate::{AshError, Result};

pub struct RenderPass {
    pub render_pass: vk::RenderPass,
    device: Arc<ash::Device>,
    managed_by_registry: bool,
}

impl RenderPass {
    pub fn builder(device: Arc<ash::Device>) -> RenderPassBuilder {
        RenderPassBuilder::new(device)
    }

    pub fn handle(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn mark_managed_by_registry(&mut self) {
        self.managed_by_registry = true;
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }
        unsafe {
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
            }
        }
        log::info!("Render pass destroyed");
    }
}

pub struct RenderPassBuilder {
    device: Arc<ash::Device>,
    color_attachments: Vec<vk::AttachmentDescription>,
    color_attachment_refs: Vec<vk::AttachmentReference>,
    resolve_attachments: Vec<vk::AttachmentDescription>,
    resolve_attachment_refs: Vec<vk::AttachmentReference>,
    depth_attachment: Option<vk::AttachmentDescription>,
    dependencies: Vec<vk::SubpassDependency>,
    sample_count: vk::SampleCountFlags,
}

impl RenderPassBuilder {
    fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            color_attachments: Vec::new(),
            color_attachment_refs: Vec::new(),
            resolve_attachments: Vec::new(),
            resolve_attachment_refs: Vec::new(),
            depth_attachment: None,
            dependencies: Vec::new(),
            sample_count: vk::SampleCountFlags::TYPE_1,
        }
    }

    /// Adds a color attachment matching swapchain usage with clear/load defaults.
    pub fn with_swapchain_color(mut self, format: vk::Format) -> Self {
        let attachment = vk::AttachmentDescription {
            format,
            samples: self.sample_count,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: if self.sample_count == vk::SampleCountFlags::TYPE_1 {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE // MSAA is resolved, not stored
            },
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: if self.sample_count == vk::SampleCountFlags::TYPE_1 {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            },
            ..Default::default()
        };

        self.push_color_attachment(attachment, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        // Add resolve attachment if MSAA is enabled
        if self.sample_count != vk::SampleCountFlags::TYPE_1 {
            let resolve = vk::AttachmentDescription {
                format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            };
            self.push_resolve_attachment(resolve, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }

        // Ensure we have at least the external->color dependency
        if self.dependencies.is_empty() {
            self.dependencies.push(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                src_access_mask: vk::AccessFlags::MEMORY_READ,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::BY_REGION,
            });
        }

        self
    }

    /// Sets the MSAA sample count for this render pass
    pub fn with_sample_count(mut self, sample_count: vk::SampleCountFlags) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// Adds a depth attachment configured for optimal depth/stencil usage.
    pub fn with_depth_attachment(mut self, format: vk::Format) -> Self {
        let (stencil_load_op, stencil_store_op) = if utils::has_stencil_component(format) {
            (
                vk::AttachmentLoadOp::CLEAR,
                vk::AttachmentStoreOp::DONT_CARE,
            )
        } else {
            (
                vk::AttachmentLoadOp::DONT_CARE,
                vk::AttachmentStoreOp::DONT_CARE,
            )
        };

        self.depth_attachment = Some(vk::AttachmentDescription {
            format,
            samples: self.sample_count,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op,
            stencil_store_op,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        });

        // Depth dependencies will be populated during build based on attachment presence.
        self
    }

    fn push_color_attachment(
        &mut self,
        attachment: vk::AttachmentDescription,
        layout: vk::ImageLayout,
    ) {
        let index = self.color_attachments.len() as u32;
        self.color_attachments.push(attachment);
        self.color_attachment_refs.push(vk::AttachmentReference {
            attachment: index,
            layout,
        });
    }

    fn push_resolve_attachment(
        &mut self,
        attachment: vk::AttachmentDescription,
        layout: vk::ImageLayout,
    ) {
        // Store with placeholder index; actual index computed in build()
        self.resolve_attachments.push(attachment);
        self.resolve_attachment_refs.push(vk::AttachmentReference {
            attachment: u32::MAX, // Placeholder - computed in build()
            layout,
        });
    }

    /// Builds the render pass.
    ///
    /// # Attachment Ordering (Internal Contract)
    /// Attachments are ordered as follows in the final `vk::RenderPass`:
    /// 1. **Color Attachments** (indices 0..color_count)
    /// 2. **Resolve Attachments** (indices color_count..color_count+resolve_count)
    /// 3. **Depth Attachment** (index at end, if present)
    ///
    /// Indices are computed centrally here to avoid fragile manual calculations.
    pub fn build(mut self) -> Result<RenderPass> {
        if self.color_attachments.is_empty() {
            return Err(AshError::VulkanError(
                "Render pass requires at least one color attachment".to_string(),
            ));
        }

        // Build attachment list: color attachments, then resolve attachments, then depth
        let mut attachments = self.color_attachments.clone();

        // Fix resolve attachment indices (they reference positions after color attachments)
        let resolve_base = attachments.len() as u32;
        let resolve_refs: Vec<vk::AttachmentReference> = self
            .resolve_attachment_refs
            .iter()
            .enumerate()
            .map(|(i, r)| vk::AttachmentReference {
                attachment: resolve_base + i as u32,
                layout: r.layout,
            })
            .collect();

        attachments.extend(self.resolve_attachments);

        let depth_ref = if let Some(depth_attachment) = self.depth_attachment.take() {
            let index = attachments.len() as u32;
            attachments.push(depth_attachment);
            Some(vk::AttachmentReference {
                attachment: index,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            })
        } else {
            None
        };

        let color_refs = self.color_attachment_refs;
        let has_resolve = !resolve_refs.is_empty();

        // Build subpass with or without resolve attachments
        let subpass = {
            let mut desc = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_refs);

            if has_resolve {
                desc = desc.resolve_attachments(&resolve_refs);
            }

            if let Some(ref depth_ref) = depth_ref {
                desc = desc.depth_stencil_attachment(depth_ref);
            }

            desc
        };

        let dependencies = if self.dependencies.is_empty() {
            let mut src_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let mut dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let mut dst_access =
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;

            if depth_ref.is_some() {
                src_stage |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                dst_stage |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
                dst_access |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            }

            vec![vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: src_stage,
                dst_stage_mask: dst_stage,
                dst_access_mask: dst_access,
                dependency_flags: if depth_ref.is_some() {
                    vk::DependencyFlags::BY_REGION
                } else {
                    vk::DependencyFlags::empty()
                },
                ..Default::default()
            }]
        } else {
            self.dependencies
        };

        let subpasses = [subpass];
        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = unsafe {
            self.device
                .create_render_pass(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("Render pass creation failed: {e}")))?
        };

        Ok(RenderPass {
            render_pass,
            device: Arc::clone(&self.device),
            managed_by_registry: false,
        })
    }
}
