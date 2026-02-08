//! Motion Vector Pass
//!
//! Renders per-pixel motion vectors to G-Buffer for use with VSR/TAA.
//! Motion vectors encode the screen-space velocity of each pixel between
//! the current and previous frame.

use ash::vk;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// Motion vector rendering pass
pub struct MotionVectorPass {
    device: Arc<ash::Device>,

    // Pipeline for motion vector rendering
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    // Render pass for motion output
    render_pass: vk::RenderPass,

    initialized: bool,
}

impl MotionVectorPass {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            pipeline: vk::Pipeline::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            initialized: false,
        }
    }

    /// Initialize motion vector pass
    ///
    /// # Safety
    /// Device and vulkan_device must be valid.
    pub unsafe fn init(
        &mut self,
        vulkan_device: &VulkanDevice,
        motion_format: vk::Format,
    ) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        self.create_render_pass(motion_format)?;
        self.create_pipeline(vulkan_device)?;

        self.initialized = true;
        log::info!("MotionVectorPass initialized");
        Ok(())
    }

    unsafe fn create_render_pass(&mut self, motion_format: vk::Format) -> Result<()> {
        let attachment = vk::AttachmentDescription::default()
            .format(motion_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        self.render_pass = self.device.create_render_pass(&create_info, None)?;
        Ok(())
    }

    unsafe fn create_pipeline(&mut self, _vulkan_device: &VulkanDevice) -> Result<()> {
        // Load shaders
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/motion.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/motion.frag.spv"));

        let vert_spv = ash::util::read_spv(&mut std::io::Cursor::new(vert_code))
            .map_err(|e| crate::AshError::VulkanError(format!("Failed to read vert spv: {e}")))?;
        let vert_module = self
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_spv), None)?;

        let frag_spv = ash::util::read_spv(&mut std::io::Cursor::new(frag_code))
            .map_err(|e| crate::AshError::VulkanError(format!("Failed to read frag spv: {e}")))?;
        let frag_module = self
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_spv), None)?;

        // Push constant range for ObjectMotionData
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<crate::renderer::resources::ObjectMotionData>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.pipeline_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

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
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"main"),
        ];

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0);

        let pipelines = self
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
            .map_err(|(_, e)| e)?;

        self.pipeline = pipelines[0];

        self.device.destroy_shader_module(vert_module, None);
        self.device.destroy_shader_module(frag_module, None);

        log::info!("MotionVectorPass: Pipeline created");
        Ok(())
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        if !self.initialized {
            return;
        }

        if self.pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.pipeline, None);
            self.pipeline = vk::Pipeline::null();
        }

        if self.pipeline_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.pipeline_layout = vk::PipelineLayout::null();
        }

        if self.render_pass != vk::RenderPass::null() {
            self.device.destroy_render_pass(self.render_pass, None);
            self.render_pass = vk::RenderPass::null();
        }

        self.initialized = false;
        log::info!("MotionVectorPass: Resources destroyed");
    }
}

impl Drop for MotionVectorPass {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
