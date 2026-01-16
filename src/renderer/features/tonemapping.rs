//! Tonemapping Feature
//!
//! Applies HDR to LDR tonemapping with configurable operators.

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};
use ash::{vk, Device};

/// Tonemapping operator selection
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TonemapOperator {
    /// ACES filmic curve - industry standard, cinematic look
    #[default]
    Aces,
    /// Reinhard - classic, softer highlights
    Reinhard,
    /// Uncharted 2 filmic - game-optimized, good contrast
    Uncharted2,
    /// No tonemapping - direct HDR output (will clamp)
    None,
}

/// Configuration for tonemapping
#[derive(Debug, Clone, Copy)]
pub struct TonemappingConfig {
    /// Which tonemapping operator to use
    pub operator: TonemapOperator,
    /// Exposure multiplier (1.0 = neutral)
    pub exposure: f32,
    /// Gamma correction value (2.2 = standard sRGB)
    pub gamma: f32,
    /// Whether tonemapping is enabled
    pub enabled: bool,
}

impl Default for TonemappingConfig {
    fn default() -> Self {
        Self {
            operator: TonemapOperator::Aces,
            exposure: 1.0,
            gamma: 2.2,
            enabled: true,
        }
    }
}

/// Tonemapping render feature
pub struct TonemappingFeature {
    config: TonemappingConfig,
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    render_pass: Option<vk::RenderPass>,
    device: Option<ash::Device>,
}

impl TonemappingFeature {
    /// Creates a new tonemapping feature with default config
    pub fn new() -> Self {
        Self {
            config: TonemappingConfig::default(),
            pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            render_pass: None,
            device: None,
        }
    }

    /// Creates a new tonemapping feature with the given config
    pub fn with_config(config: TonemappingConfig) -> Self {
        Self {
            config,
            pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            render_pass: None,
            device: None,
        }
    }

    /// Returns the current config
    pub fn config(&self) -> &TonemappingConfig {
        &self.config
    }

    /// Returns a mutable reference to the config
    pub fn config_mut(&mut self) -> &mut TonemappingConfig {
        &mut self.config
    }

    /// Sets the exposure value
    pub fn set_exposure(&mut self, exposure: f32) {
        self.config.exposure = exposure;
    }

    /// Sets the gamma value
    pub fn set_gamma(&mut self, gamma: f32) {
        self.config.gamma = gamma;
    }

    /// Sets the tonemapping operator
    pub fn set_operator(&mut self, operator: TonemapOperator) {
        self.config.operator = operator;
    }

    /// Toggles tonemapping on/off
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Create tonemapping pipeline
    ///
    /// # Safety
    /// Device must be valid and remain valid for pipeline lifetime.
    unsafe fn create_pipeline(&mut self, device: &Device) -> crate::Result<()> {
        // Load shaders
        let vert_code = include_bytes!("../../../shaders/postprocess.vert.spv");
        let frag_code = include_bytes!("../../../shaders/tonemapping.frag.spv");

        let vert_info = vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(vert_code));
        let vert_module = device.create_shader_module(&vert_info, None)?;

        let frag_info = vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(frag_code));
        let frag_module = device.create_shader_module(&frag_info, None)?;

        // Create descriptor set layout (3 sampled images: HDR, Bloom, SSGI)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            device.create_descriptor_set_layout(&descriptor_layout_info, None)?;
        self.descriptor_set_layout = Some(descriptor_set_layout);

        // Push constants for tonemapping parameters
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct TonemapPushConstants {
            exposure: f32,
            gamma: f32,
            bloom_intensity: f32,
            tonemap_enabled: f32,
        }

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<TonemapPushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;
        self.pipeline_layout = Some(pipeline_layout);

        // Create render pass (output to swapchain format)
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::B8G8R8A8_SRGB)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&attachment))
            .subpasses(std::slice::from_ref(&subpass));

        let render_pass = device.create_render_pass(&render_pass_info, None)?;
        self.render_pass = Some(render_pass);

        // Create pipeline
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
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
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
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipelines = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
            .map_err(|(_, e)| e)?;

        self.pipeline = Some(pipelines[0]);

        // Cleanup shader modules
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        log::info!(
            "Tonemapping pipeline created (operator: {:?})",
            self.config.operator
        );
        Ok(())
    }
}

impl Default for TonemappingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for TonemappingFeature {
    fn name(&self) -> &'static str {
        "TonemappingFeature"
    }

    fn on_added(&mut self, device: &Device) {
        log::info!(
            "Tonemapping feature added (operator: {:?})",
            self.config.operator
        );
        self.device = Some(device.clone());

        // Create pipeline
        if let Err(e) = unsafe { self.create_pipeline(device) } {
            log::error!("Failed to create tonemapping pipeline: {e}");
        }
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Could animate exposure here for auto-exposure
    }

    unsafe fn render(&self, ctx: &FeatureRenderContext<'_>) {
        if !self.config.enabled {
            return;
        }

        // Early return if pipeline not ready
        let Some(pipeline) = self.pipeline else {
            return;
        };
        let Some(pipeline_layout) = self.pipeline_layout else {
            return;
        };

        let device = ctx.device;
        let cmd = ctx.command_buffer;

        // Note: Full tonemapping implementation requires:
        // 1. HDR source texture (from renderer)
        // 2. Bloom texture (from BloomFeature)
        // 3. SSGI texture (from renderer)
        // 4. Descriptor set for binding all 3 textures
        // 5. Sampler for texture sampling
        // 6. Framebuffer for swapchain output
        //
        // Current limitation: FeatureRenderContext doesn't provide access to these resources.
        // Full integration requires renderer-level changes.
        //
        // Example command recording (when resources available):
        /*
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct TonemapPushConstants {
            exposure: f32,
            gamma: f32,
            bloom_intensity: f32,
            tonemap_enabled: f32,
        }

        let pc = TonemapPushConstants {
            exposure: self.config.exposure,
            gamma: self.config.gamma,
            bloom_intensity: 0.5, // From BloomFeature
            tonemap_enabled: if self.config.enabled { 1.0 } else { 0.0 },
        };

        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&pc),
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        // ... bind descriptor set, begin render pass, draw fullscreen triangle ...
        */

        // Placeholder: Log that tonemapping would execute
        let _ = (device, cmd, pipeline, pipeline_layout);
        log::trace!("Tonemapping render called (awaiting renderer integration)");
    }

    fn on_removed(&mut self, device: &Device) {
        unsafe {
            // Cleanup pipeline resources
            if let Some(pipeline) = self.pipeline.take() {
                device.destroy_pipeline(pipeline, None);
            }
            if let Some(layout) = self.pipeline_layout.take() {
                device.destroy_pipeline_layout(layout, None);
            }
            if let Some(layout) = self.descriptor_set_layout.take() {
                device.destroy_descriptor_set_layout(layout, None);
            }
            if let Some(render_pass) = self.render_pass.take() {
                device.destroy_render_pass(render_pass, None);
            }
        }
        log::info!("Tonemapping feature removed");
    }
}
