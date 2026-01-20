//! VSM Shadow Pass - Renders geometry into physical cache pages

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::resources::{PageAllocation, VsmResources};

/// VSM shadow rendering pass
pub struct VsmShadowPass {
    device: Arc<ash::Device>,

    /// Render pass for depth-only rendering
    pub render_pass: vk::RenderPass,

    /// Framebuffer for physical cache
    pub framebuffer: vk::Framebuffer,

    /// Pipeline for shadow rendering
    shadow_pipeline: Option<vk::Pipeline>,
    shadow_pipeline_layout: Option<vk::PipelineLayout>,
}

impl VsmShadowPass {
    /// Create a new VSM shadow pass
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of this pass.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        _allocator: Arc<Allocator>,
        resources: &VsmResources,
    ) -> Result<Self> {
        log::info!("Creating VSM shadow pass");

        // Create render pass (depth-only)
        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::R32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::LOAD, // Load existing data
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::GENERAL,
            final_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        };

        let depth_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&depth_ref));

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
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
                AshError::VulkanError(format!("VSM render pass creation failed: {e:?}"))
            })?;

        // Create framebuffer
        let attachments = [resources.physical_cache_view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(resources.config().physical_resolution)
            .height(resources.config().physical_resolution)
            .layers(1);

        let framebuffer = device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("VSM framebuffer creation failed: {e:?}"))
            })?;

        log::info!("VSM shadow pass created successfully");

        Ok(Self {
            device,
            render_pass,
            framebuffer,
            shadow_pipeline: None,
            shadow_pipeline_layout: None,
        })
    }

    /// Create shadow rendering pipeline
    ///
    /// # Safety
    /// Device must remain valid. Descriptor set layouts must be valid.
    pub unsafe fn create_pipeline(
        &mut self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<()> {
        log::info!("Creating VSM shadow pipeline");

        // Push constant range (extended for lightSpaceMatrix at offset 160)
        const DRAW_PUSH_VERTEX_BYTES: u32 = 128;
        const DRAW_PUSH_FRAGMENT_BYTES: u32 = 32;
        const LIGHT_SPACE_MATRIX_BYTES: u32 = 64;

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: DRAW_PUSH_VERTEX_BYTES + DRAW_PUSH_FRAGMENT_BYTES + LIGHT_SPACE_MATRIX_BYTES,
        };

        // Create pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Shadow pipeline layout creation failed: {e:?}"))
            })?;

        // Load shaders
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv"));

        let vert_module = self
            .device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(vert_code)),
                None,
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Shadow vertex shader creation failed: {e:?}"))
            })?;

        let frag_module = self
            .device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(frag_code)),
                None,
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Shadow fragment shader creation failed: {e:?}"))
            })?;

        let entry_point = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_point),
        ];

        // Vertex input (matches main pass - Position, Normal, UV, etc.)
        let binding_desc = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<crate::renderer::resources::Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };

        let attribute_descs = [
            // Position
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            // Normal
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            // UV
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(1.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(2.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0);

        let pipelines = self
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| {
                AshError::VulkanError(format!("Shadow pipeline creation failed: {e:?}"))
            })?;

        // Cleanup shader modules
        self.device.destroy_shader_module(vert_module, None);
        self.device.destroy_shader_module(frag_module, None);

        self.shadow_pipeline = Some(pipelines[0]);
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

    /// Render shadows for allocated pages
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn render_shadows<F>(
        &self,
        cmd: vk::CommandBuffer,
        allocations: &[PageAllocation],
        page_size: u32,
        mut draw_fn: F,
    ) where
        F: FnMut(vk::CommandBuffer, &PageAllocation),
    {
        if allocations.is_empty() {
            return;
        }

        log::debug!("Rendering {} shadow pages", allocations.len());

        // For each allocated page, set viewport and render
        for allocation in allocations {
            // Calculate viewport for this physical page
            let x = (allocation.physical_x * page_size) as f32;
            let y = (allocation.physical_y * page_size) as f32;

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

            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Push lightSpaceMatrix (placeholder identity for MVP stabilization)
            // TODO: Calculate proper light-space matrix from PageAllocation and ClipmapManager
            let light_space_matrix = glam::Mat4::IDENTITY;
            let matrix_bytes = bytemuck::bytes_of(&light_space_matrix);

            if let Some(layout) = self.shadow_pipeline_layout {
                self.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    160, // Offset for lightSpaceMatrix
                    matrix_bytes,
                );
            }

            // Call user draw function for this page
            draw_fn(cmd, allocation);
        }
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying VSM shadow pass");

        if let Some(pipeline) = self.shadow_pipeline.take() {
            self.device.destroy_pipeline(pipeline, None);
        }

        if let Some(layout) = self.shadow_pipeline_layout.take() {
            self.device.destroy_pipeline_layout(layout, None);
        }

        if self.framebuffer != vk::Framebuffer::null() {
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.framebuffer = vk::Framebuffer::null();
        }

        if self.render_pass != vk::RenderPass::null() {
            self.device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }

        log::debug!("VSM shadow pass destroyed");
    }
}

impl Drop for VsmShadowPass {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
