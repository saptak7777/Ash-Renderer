//! VSM Shadow Pass - Renders geometry into physical cache pages

use ash::vk;
use std::sync::Arc;

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::resources::{PageAllocation, VsmResources};

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowPushConstants {
    // 0-31: BDA pointers (32 bytes)
    pub vertex_ptr_low: u32,
    pub vertex_ptr_high: u32,
    pub instance_ptr_low: u32,
    pub instance_ptr_high: u32,
    pub index_ptr_low: u32,
    pub index_ptr_high: u32,
    pub transform_ptr_low: u32,
    pub transform_ptr_high: u32,

    // 32-63: Indices and Flags (32 bytes)
    pub transform_index: u32,
    pub use_instancing: u32,
    pub _padding_lean: [u32; 6], // Padding to reach offset 64 for light_space_matrix

    // 64-127: Light Space Matrix (64 bytes)
    pub light_space_matrix: [[f32; 4]; 4], // Mat4 at offset 64
}

/// VSM shadow rendering pass
pub struct VsmShadowPass {
    device: Arc<ash::Device>,

    /// Render pass for depth-only rendering
    pub render_pass: vk::RenderPass,

    /// Framebuffer for physical cache
    pub framebuffer: vk::Framebuffer,

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

        // Create render pass (dual-attachment: color for variance + depth for testing)
        // Color attachment: R32G32_SFLOAT for storing (depth, depth^2) moments
        let color_attachment = vk::AttachmentDescription {
            format: vk::Format::R32G32_SFLOAT, // Store variance moments
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        };

        // Depth attachment: D32_SFLOAT for hardware depth testing
        let depth_attachment = vk::AttachmentDescription {
            format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE, // We don't need to keep depth after the pass
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let color_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        let dependency_in = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, // Wait for depth writes
            dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::BY_REGION,
        };

        let dependency_out = vk::SubpassDependency {
            src_subpass: 0,
            dst_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            dependency_flags: vk::DependencyFlags::BY_REGION,
        };

        let attachments = [color_attachment, depth_attachment];
        let subpasses = [subpass];
        let dependencies = [dependency_in, dependency_out];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("VSM render pass creation failed: {e:?}"))
            })?;

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

        let (depth_image, depth_image_alloc) = allocator
            .create_image(&depth_info, vk_mem::MemoryUsage::AutoPreferDevice)
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

        let depth_view = device
            .create_image_view(&depth_view_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create VSM depth view: {e:?}"))
            })?;

        // Create framebuffer with both color and depth attachments
        let attachments = [resources.physical_cache_view, depth_view];
        let physical_res = resources.config().physical_resolution;
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(physical_res)
            .height(physical_res)
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

        // BDA-only pipeline: No vertex input bindings/attributes
        // The shader pulls vertices from the global heap using the push constant pointer.
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

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
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL) // FIX: Reverse-Z support
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G)
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

    /// Render shadows using GPU-generated indirect commands
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn render_shadows_indirect(
        &self,
        cmd: vk::CommandBuffer,
        allocations: &[PageAllocation],
        page_size: u32,
        frame_set: vk::DescriptorSet,
        bindless_set: vk::DescriptorSet,
        indirect_buffer: vk::Buffer,
        count_buffer: vk::Buffer,
        max_commands_per_level: u32,
        level_matrices: &[glam::Mat4],
        _material_ptr: u64,
        index_ptr: u64,
        _light_ptr: u64,
        _tile_ptr: u64,
        _frame_ptr: u64,
        vertex_ptr: u64,
        instance_ptr: u64,
        transform_ptr: u64,
        transform_index: u32,
    ) {
        if allocations.is_empty() {
            return;
        }

        log::debug!("Rendering {} shadow pages (indirect)", allocations.len());

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 1.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0, // Reverse-Z
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.physical_resolution,
                    height: self.physical_resolution,
                },
            })
            .clear_values(&clear_values);

        self.device
            .cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

        // Bind Pipeline
        if let Some(pipeline) = self.shadow_pipeline {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        // Bind Descriptor Sets
        if let Some(layout) = self.shadow_pipeline_layout {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[frame_set, bindless_set],
                &[],
            );
        }

        // For each allocated page, set viewport and execute indirect draw
        for allocation in allocations {
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

            // Push constants for this level
            let level_matrix = level_matrices
                .get(allocation.layer as usize)
                .cloned()
                .unwrap_or(glam::Mat4::IDENTITY);

            let push_constants = ShadowPushConstants {
                vertex_ptr_low: vertex_ptr as u32,
                vertex_ptr_high: (vertex_ptr >> 32) as u32,
                instance_ptr_low: instance_ptr as u32,
                instance_ptr_high: (instance_ptr >> 32) as u32,
                index_ptr_low: index_ptr as u32,
                index_ptr_high: (index_ptr >> 32) as u32,
                transform_ptr_low: transform_ptr as u32,
                transform_ptr_high: (transform_ptr >> 32) as u32,

                transform_index: transform_index,
                use_instancing: 1, // DrawIndirect uses gl_InstanceIndex
                _padding_lean: [0; 6],
                light_space_matrix: level_matrix.to_cols_array_2d(),
            };

            if let Some(layout) = self.shadow_pipeline_layout {
                self.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
            }

            // Execute indirect draw for this page's clipmap level
            let level_offset = (allocation.layer * max_commands_per_level) as vk::DeviceSize
                * std::mem::size_of::<crate::renderer::vcgs::IndirectDrawCommand>()
                    as vk::DeviceSize;
            let count_offset = (allocation.layer * 4) as vk::DeviceSize;

            self.device.cmd_draw_indirect_count(
                cmd,
                indirect_buffer,
                level_offset,
                count_buffer,
                count_offset,
                max_commands_per_level,
                std::mem::size_of::<crate::renderer::vcgs::IndirectDrawCommand>() as u32,
            );
        }

        self.device.cmd_end_render_pass(cmd);
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
        light_space_matrix: &glam::Mat4,
        frame_set: vk::DescriptorSet,
        bindless_set: vk::DescriptorSet,
        mut draw_fn: F,
    ) where
        F: FnMut(vk::CommandBuffer, &PageAllocation),
    {
        if allocations.is_empty() {
            return;
        }

        log::debug!("Rendering {} shadow pages", allocations.len());

        // CRITICAL FIX: Begin Render Pass with proper clear values
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 1.0, 0.0, 0.0], // Clear variance to (1.0, 1.0) for far depth
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0, // Reverse-Z: 0.0 = far plane
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.physical_resolution,
                    height: self.physical_resolution,
                },
            })
            .clear_values(&clear_values);

        self.device
            .cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

        // Bind Pipeline
        if let Some(pipeline) = self.shadow_pipeline {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        // Bind Bindless Descriptor Set (Set 1 for BDA vertex pulling)
        if let Some(layout) = self.shadow_pipeline_layout {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,                          // Start at Set 0
                &[frame_set, bindless_set], // Bind Set 0 and Set 1
                &[],
            );
        }

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

            // Push lightSpaceMatrix from parameter (not hardcoded IDENTITY)
            let matrix_bytes = bytemuck::bytes_of(light_space_matrix);

            if let Some(layout) = self.shadow_pipeline_layout {
                self.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    160,
                    matrix_bytes,
                );
            }

            // Call user draw function for this page
            // Call user draw function for this page
            draw_fn(cmd, allocation);
        }

        // CRITICAL FIX: End Render Pass
        self.device.cmd_end_render_pass(cmd);
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

        // Destroy depth buffer
        if self.depth_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.depth_view, None);
            self.depth_view = vk::ImageView::null();
        }
        if let Some(mut alloc) = self.depth_image_alloc.take() {
            allocator.vma.destroy_image(self.depth_image, &mut alloc);
            self.depth_image = vk::Image::null();
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
        // Note: destroy() requires allocator, which we don't have in Drop
        // The allocator will clean up the depth buffer when it's destroyed
        log::warn!("VsmShadowPass dropped without explicit destroy() call");
    }
}
