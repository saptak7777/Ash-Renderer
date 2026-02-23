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

/// Push constants shared by `shadow.vert` and `shadow.frag`.
///
/// The layout is the union of what both stages need and MUST match the GLSL
/// blocks in those shaders exactly. All offsets are explicit.
///
///   BDA pointers (0–55):
///     frame_ptr     u64  @ 0
///     vertex_ptr    u64  @ 8
///     instance_ptr  u64  @ 16
///     material_ptr  u64  @ 24  ← fragment uses this for alpha-cutout
///     index_ptr     u64  @ 32
///     light_ptr     u64  @ 40
///     tile_ptr      u64  @ 48
///   Texture indices (56–63):
///     vsm_page_index   u32  @ 56
///     vsm_cache_index  u32  @ 60
///   Model matrix (64–127):
///     model  mat4  @ 64
///   Material & flags (128–159):
///     material_index              u32  @ 128  ← fragment uses for alpha-cutout
///     use_instancing              u32  @ 132
///     flags                       u32  @ 136
///     debug_path                  u32  @ 140
///     debug_visualization_enabled u32  @ 144
///     skybox_index                u32  @ 148
///     _pad_mat                    u32  @ 152 (×2 pad to 160)
///     _pad_mat2                   u32  @ 156
///   Light space matrix (160–223):
///     light_space_matrix  mat4  @ 160
///   Total: 224 bytes
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowPushConstants {
    // ── BDA pointers ─────────────────────────────────────────────────
    pub frame_ptr: u64,    // offset   0
    pub vertex_ptr: u64,   // offset   8
    pub instance_ptr: u64, // offset  16
    pub material_ptr: u64, // offset  24  ← frag: alpha-cutout BDA
    pub index_ptr: u64,    // offset  32
    pub light_ptr: u64,    // offset  40
    pub tile_ptr: u64,     // offset  48

    // ── Texture indices ───────────────────────────────────────────────
    pub vsm_page_index: u32,  // offset  56
    pub vsm_cache_index: u32, // offset  60

    // ── Model matrix (per-draw, used by vert) ────────────────────────
    pub model: [[f32; 4]; 4], // offset  64  (64 bytes)

    // ── Material / control flags (frag reads from 128) ───────────────
    pub material_index: u32,              // offset 128
    pub use_instancing: u32,              // offset 132
    pub flags: u32,                       // offset 136
    pub debug_path: u32,                  // offset 140
    pub debug_visualization_enabled: u32, // offset 144
    pub skybox_index: u32,                // offset 148
    pub _pad_mat: u32,                    // offset 152 — pad to align mat4
    pub _pad_mat2: u32,                   // offset 156

    // ── Light space matrix (160–223) ─────────────────────────────────
    pub light_space_matrix: [[f32; 4]; 4], // offset 160
}

/// Byte offset of the light-space-matrix field within ShadowPushConstants.
pub const LIGHT_SPACE_MATRIX_OFFSET: u32 = 160;
/// Byte offset of the model-matrix field within ShadowPushConstants.
pub const MODEL_MATRIX_OFFSET: u32 = 64;

const _: () = assert!(
    std::mem::size_of::<ShadowPushConstants>() == 224,
    "ShadowPushConstants must be 224 bytes to match shadow.vert / shadow.frag"
);

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

        let render_pass =
            unsafe { device.create_render_pass(&render_pass_info, None) }.map_err(|e| {
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

        // Create framebuffer with both color and depth attachments
        let attachments = [resources.physical_cache_view, depth_view];
        let physical_res = resources.config().physical_resolution;
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(physical_res)
            .height(physical_res)
            .layers(1);

        let framebuffer =
            unsafe { device.create_framebuffer(&framebuffer_info, None) }.map_err(|e| {
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
        _device_layout: vk::DescriptorSetLayout, // Set 0 (Reserved for future global bindless extension)
        bindless_layout: vk::DescriptorSetLayout, // Set 1 (Bindless Textures)
    ) -> Result<()> {
        log::info!("Creating VSM shadow pipeline");

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            // Use the Rust struct size as the authoritative source so it can never drift.
            size: std::mem::size_of::<ShadowPushConstants>() as u32,
        };

        // Descriptor set layouts: Set 0 (Reserved) and Set 1 (Bindless Textures)
        // We must have two layouts to match 'layout(set = 1, ...)' in the shader
        let layouts = [_device_layout, bindless_layout];

        // Create pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .map_err(|e| {
                AshError::VulkanError(format!("Shadow pipeline layout creation failed: {e:?}"))
            })?;

        // Load shaders
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv"));

        let vert_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(vert_code)),
                None,
            )
        }
        .map_err(|e| {
            AshError::VulkanError(format!("Shadow vertex shader creation failed: {e:?}"))
        })?;

        let frag_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(frag_code)),
                None,
            )
        }
        .map_err(|e| {
            AshError::VulkanError(format!("Shadow fragment shader creation failed: {e:?}"))
        })?;

        let entry_point = c"main";

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

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|e| AshError::VulkanError(format!("Shadow pipeline creation failed: {e:?}")))?;

        // Cleanup shader modules
        unsafe {
            self.device.destroy_shader_module(vert_module, None);
            self.device.destroy_shader_module(frag_module, None);
        }

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

        // 1. Transition layouts to attachment optimal
        let cache_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
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

        let depth_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) // Using DEPTH_STENCIL_ATTACHMENT_OPTIMAL for broad hardware compatibility (avoids requiring KHR_separate_depth_stencil_layouts).
            .image(info.resources.physical_depth_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[cache_barrier, depth_barrier],
            );
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

        // 4. Bind Bindless Texture Descriptor Set (Set 1)
        if let Some(layout) = self.shadow_pipeline_layout {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    1, // Set 1
                    &[info.bindless_descriptor_set],
                    &[],
                );
            }

            // Build the per-batch base push. Material pointer and index are left zero —
            // the shadow pass currently renders opaque objects. When alpha-tested geometry
            // is added, callers should fill material_ptr + material_index before issuing
            // the draw. light_space_matrix is overwritten per-page below.
            let bda_push = ShadowPushConstants {
                frame_ptr: 0,
                vertex_ptr: info.vertex_addr,
                instance_ptr: info.object_addr,
                material_ptr: 0, // zero → frag skips alpha cutout branch
                index_ptr: info.index_addr,
                light_ptr: 0,
                tile_ptr: 0,
                vsm_page_index: 0,
                vsm_cache_index: 0,
                model: [[0.0; 4]; 4],
                material_index: 0,
                use_instancing: 1, // Enable instancing for the manual pull
                flags: 0,
                debug_path: 0,
                debug_visualization_enabled: 0,
                skybox_index: 0,
                _pad_mat: 0,
                _pad_mat2: 0,
                light_space_matrix: [[0.0; 4]; 4], // overwritten per-page
            };

            unsafe {
                self.device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&bda_push), // push all 224 bytes
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

            // Per-page: update only the light-space-matrix slot (offset 160).
            let matrix = page.mvp.to_cols_array_2d();
            let matrix_bytes = bytemuck::bytes_of(&matrix);
            if let Some(layout) = self.shadow_pipeline_layout {
                unsafe {
                    self.device.cmd_push_constants(
                        cmd,
                        layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        LIGHT_SPACE_MATRIX_OFFSET, // 160 — matches ShadowPushConstants.light_space_matrix
                        matrix_bytes,
                    );
                }
            }

            // Draw call
            scene_draw_fn(cmd);
        }

        unsafe {
            self.device.cmd_end_rendering(cmd);
        }

        // 5. Transition Back
        let back_cache_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
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

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[back_cache_barrier],
            );
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
