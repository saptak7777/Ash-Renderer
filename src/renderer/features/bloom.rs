//! Bloom Feature
//!
//! Multi-pass bloom effect with threshold, downsample, and upsample stages.
//! Implements industry-standard dual-filtering bloom with firefly suppression.

use super::{FeatureFrameContext, FeatureRenderContext, RenderFeature};
use ash::{vk, Device};

/// Push constants for bloom shaders, corresponding to the GLSL layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BloomPushConstants {
    /// Texel size (1.0 / texture_width, 1.0 / texture_height)
    pub texel_size: [f32; 2],
    /// Brightness threshold for prefilter
    pub threshold: f32,
    /// Soft knee for smooth threshold transition
    pub soft_knee: f32,
}

/// Configuration for bloom effect
#[derive(Debug, Clone, Copy)]
pub struct BloomConfig {
    /// Brightness threshold for bloom extraction (0.0 - 2.0, default 1.0)
    pub threshold: f32,
    /// Bloom intensity multiplier (0.0 - 1.0, default 0.5)
    pub intensity: f32,
    /// Number of mip levels for blur (3-8, default 5)
    pub mip_count: u32,
    /// Soft knee for threshold (0.0 - 1.0, default 0.5)
    pub soft_knee: f32,
    /// Whether bloom is enabled
    pub enabled: bool,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            intensity: 0.5,
            mip_count: 5,
            soft_knee: 0.5,
            enabled: true,
        }
    }
}

/// Mip level info for bloom chain
#[derive(Debug, Clone, Copy)]
pub struct MipInfo {
    pub width: u32,
    pub height: u32,
}

/// Bloom pass data manager (CPU-side logic, GPU-agnostic)
///
/// Handles mip chain calculations and push constant generation.
/// GPU resources are managed by BloomFeature.
pub struct BloomPass {
    config: BloomConfig,
    mip_chain: Vec<MipInfo>,
    base_width: u32,
    base_height: u32,
}

impl BloomPass {
    /// Create a new bloom pass with default config
    pub fn new() -> Self {
        Self {
            config: BloomConfig::default(),
            mip_chain: Vec::new(),
            base_width: 0,
            base_height: 0,
        }
    }

    /// Create with custom config
    pub fn with_config(config: BloomConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Get config reference
    pub fn config(&self) -> &BloomConfig {
        &self.config
    }

    /// Get mutable config reference
    pub fn config_mut(&mut self) -> &mut BloomConfig {
        &mut self.config
    }

    /// Calculate mip chain for given resolution
    ///
    /// Should be called on resize or first frame.
    pub fn calculate_mip_chain(&mut self, width: u32, height: u32) {
        if width == self.base_width && height == self.base_height {
            return; // Already calculated
        }

        self.base_width = width;
        self.base_height = height;
        self.mip_chain.clear();

        let mut mip_width = width;
        let mut mip_height = height;

        for _ in 0..self.config.mip_count {
            // Each subsequent mip level is half the dimensions of the previous, with a minimum size of 1x1.
            mip_width = (mip_width / 2).max(1);
            mip_height = (mip_height / 2).max(1);

            self.mip_chain.push(MipInfo {
                width: mip_width,
                height: mip_height,
            });

            // Termination condition reached if dimensions are 1x1.
            if mip_width == 1 && mip_height == 1 {
                break;
            }
        }

        log::debug!(
            "Bloom: calculated {} mip levels for {}x{} (smallest: {}x{})",
            self.mip_chain.len(),
            width,
            height,
            self.mip_chain.last().map(|m| m.width).unwrap_or(0),
            self.mip_chain.last().map(|m| m.height).unwrap_or(0),
        );
    }

    /// Get push constants for prefilter pass
    pub fn get_prefilter_push_constants(&self) -> BloomPushConstants {
        BloomPushConstants {
            texel_size: [1.0 / self.base_width as f32, 1.0 / self.base_height as f32],
            threshold: self.config.threshold,
            soft_knee: self.config.soft_knee,
        }
    }

    /// Get push constants for a specific downsample mip level
    pub fn get_downsample_push_constants(&self, mip_index: usize) -> Option<BloomPushConstants> {
        self.mip_chain.get(mip_index).map(|mip| BloomPushConstants {
            texel_size: [1.0 / mip.width as f32, 1.0 / mip.height as f32],
            threshold: 0.0, // Not used in downsample
            soft_knee: 0.0,
        })
    }

    /// Get push constants for a specific upsample mip level
    pub fn get_upsample_push_constants(&self, mip_index: usize) -> Option<BloomPushConstants> {
        // Upsample goes from smallest to largest
        let target_index = self.mip_chain.len().saturating_sub(1 + mip_index);
        self.mip_chain
            .get(target_index)
            .map(|mip| BloomPushConstants {
                texel_size: [1.0 / mip.width as f32, 1.0 / mip.height as f32],
                threshold: self.config.intensity, // Repurpose for blend factor
                soft_knee: 0.0,
            })
    }

    /// Get number of mip levels in the chain
    pub fn mip_count(&self) -> usize {
        self.mip_chain.len()
    }

    /// Get mip info for a specific level
    pub fn get_mip_info(&self, index: usize) -> Option<&MipInfo> {
        self.mip_chain.get(index)
    }

    /// Is bloom enabled and valid?
    pub fn is_enabled(&self) -> bool {
        self.config.enabled && !self.mip_chain.is_empty()
    }
}

impl Default for BloomPass {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU resources for bloom effect
///
/// Manages bloom image with mip chain and associated views.
/// Implements RAII cleanup pattern.
pub struct BloomFeature {
    pass: BloomPass,
    device: Option<ash::Device>,
    prefilter_pipeline: Option<vk::Pipeline>,
    downsample_pipeline: Option<vk::Pipeline>,
    upsample_pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    render_pass: Option<vk::RenderPass>,
}

impl BloomFeature {
    /// Creates a new bloom feature with default config
    pub fn new() -> Self {
        Self {
            pass: BloomPass::new(),
            device: None,
            prefilter_pipeline: None,
            downsample_pipeline: None,
            upsample_pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            render_pass: None,
        }
    }

    /// Creates a new bloom feature with the given config
    pub fn with_config(config: BloomConfig) -> Self {
        Self {
            pass: BloomPass::with_config(config),
            device: None,
            prefilter_pipeline: None,
            downsample_pipeline: None,
            upsample_pipeline: None,
            pipeline_layout: None,
            descriptor_set_layout: None,
            render_pass: None,
        }
    }

    /// Returns the current config
    pub fn config(&self) -> &BloomConfig {
        self.pass.config()
    }

    /// Returns a mutable reference to the config
    pub fn config_mut(&mut self) -> &mut BloomConfig {
        self.pass.config_mut()
    }

    /// Access the underlying BloomPass
    pub fn pass(&self) -> &BloomPass {
        &self.pass
    }

    /// Access the underlying BloomPass mutably
    pub fn pass_mut(&mut self) -> &mut BloomPass {
        &mut self.pass
    }

    /// Sets the threshold value
    pub fn set_threshold(&mut self, threshold: f32) {
        self.pass.config_mut().threshold = threshold.clamp(0.0, 2.0);
    }

    /// Sets the intensity value
    pub fn set_intensity(&mut self, intensity: f32) {
        self.pass.config_mut().intensity = intensity.clamp(0.0, 1.0);
    }

    /// Sets the mip count (requires re-calculation of chain)
    pub fn set_mip_count(&mut self, mip_count: u32) {
        self.pass.config_mut().mip_count = mip_count.clamp(3, 8);
        // Force recalculation on next frame
        self.pass.base_width = 0;
        self.pass.base_height = 0;
    }

    /// Toggles bloom on/off
    pub fn set_enabled(&mut self, enabled: bool) {
        self.pass.config_mut().enabled = enabled;
    }

    /// Create bloom pipelines
    ///
    /// # Safety
    /// Device must be valid and remain valid for pipeline lifetime.
    unsafe fn create_pipelines(&mut self, device: &Device) -> crate::Result<()> {
        // Load shaders
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/postprocess.vert.spv"));
        let prefilter_frag_code =
            include_bytes!(concat!(env!("OUT_DIR"), "/bloom_prefilter.frag.spv"));
        let downsample_frag_code =
            include_bytes!(concat!(env!("OUT_DIR"), "/bloom_downsample.frag.spv"));
        let upsample_frag_code =
            include_bytes!(concat!(env!("OUT_DIR"), "/bloom_upsample.frag.spv"));

        // Use ash::util::read_spv to ensure proper alignment
        let vert_spv = ash::util::read_spv(&mut std::io::Cursor::new(vert_code)).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to parse vertex shader: {e}"))
        })?;
        let vert_info = vk::ShaderModuleCreateInfo::default().code(&vert_spv);
        let vert_module = device.create_shader_module(&vert_info, None)?;

        let prefilter_spv = ash::util::read_spv(&mut std::io::Cursor::new(prefilter_frag_code))
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to parse bloom prefilter shader: {e}"))
            })?;
        let prefilter_info = vk::ShaderModuleCreateInfo::default().code(&prefilter_spv);
        let prefilter_frag_module = device.create_shader_module(&prefilter_info, None)?;

        let downsample_spv = ash::util::read_spv(&mut std::io::Cursor::new(downsample_frag_code))
            .map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to parse bloom downsample shader: {e}"))
        })?;
        let downsample_info = vk::ShaderModuleCreateInfo::default().code(&downsample_spv);
        let downsample_frag_module = device.create_shader_module(&downsample_info, None)?;

        let upsample_spv = ash::util::read_spv(&mut std::io::Cursor::new(upsample_frag_code))
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to parse bloom upsample shader: {e}"))
            })?;
        let upsample_info = vk::ShaderModuleCreateInfo::default().code(&upsample_spv);
        let upsample_frag_module = device.create_shader_module(&upsample_info, None)?;

        // Create descriptor set layout (1 sampled image)
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            device.create_descriptor_set_layout(&descriptor_layout_info, None)?;
        self.descriptor_set_layout = Some(descriptor_set_layout);

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<BloomPushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;
        self.pipeline_layout = Some(pipeline_layout);

        // Create render pass (single color attachment, no depth)
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

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

        // Common pipeline state (fullscreen triangle, no vertex input)
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

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Prefilter & Downsample: No blending
        let color_blend_attachment_opaque = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let color_blending_opaque = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment_opaque));

        // Upsample: Additive blending (ONE, ONE)
        let color_blend_attachment_additive = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blending_additive = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment_additive));

        // Create Prefilter pipeline
        let prefilter_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(prefilter_frag_module)
                .name(c"main"),
        ];

        let prefilter_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&prefilter_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending_opaque)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        // Create Downsample pipeline
        let downsample_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(downsample_frag_module)
                .name(c"main"),
        ];

        let downsample_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&downsample_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending_opaque)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        // Create Upsample pipeline (with additive blending)
        let upsample_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(upsample_frag_module)
                .name(c"main"),
        ];

        let upsample_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&upsample_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending_additive)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        // Create all pipelines
        let pipeline_infos = [prefilter_info, downsample_info, upsample_info];
        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
            .map_err(|(_, e)| e)?;

        self.prefilter_pipeline = Some(pipelines[0]);
        self.downsample_pipeline = Some(pipelines[1]);
        self.upsample_pipeline = Some(pipelines[2]);

        // Cleanup shader modules
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(prefilter_frag_module, None);
        device.destroy_shader_module(downsample_frag_module, None);
        device.destroy_shader_module(upsample_frag_module, None);

        log::info!("Bloom pipelines created");
        Ok(())
    }
}

impl Default for BloomFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderFeature for BloomFeature {
    fn name(&self) -> &'static str {
        "BloomFeature"
    }

    fn on_added(&mut self, device: &Device) {
        let cfg = self.pass.config();
        log::info!(
            "Bloom feature added (threshold: {:.2}, intensity: {:.2}, mips: {})",
            cfg.threshold,
            cfg.intensity,
            cfg.mip_count
        );
        self.device = Some(device.clone());

        // Create pipelines
        if let Err(e) = unsafe { self.create_pipelines(device) } {
            log::error!("Failed to create bloom pipelines: {e}");
        }
    }

    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {
        // Note: Resource creation deferred to render() where we have access to actual dimensions
        // This is a limitation of the current FeatureFrameContext API
    }

    unsafe fn render(&self, ctx: &FeatureRenderContext<'_>) {
        if !self.pass.is_enabled() {
            return;
        }

        // Early return if pipelines not ready
        let Some(prefilter_pipeline) = self.prefilter_pipeline else {
            return;
        };
        let Some(downsample_pipeline) = self.downsample_pipeline else {
            return;
        };
        let Some(upsample_pipeline) = self.upsample_pipeline else {
            return;
        };
        let Some(pipeline_layout) = self.pipeline_layout else {
            return;
        };
        let Some(render_pass) = self.render_pass else {
            return;
        };

        let device = ctx.device;
        let cmd = ctx.command_buffer;

        // Note: Full bloom implementation requires:
        // 1. Source HDR image (from renderer)
        // 2. Framebuffers for each mip level
        // 3. Descriptor sets for binding textures
        // 4. Sampler for texture sampling
        //
        // Current limitation: FeatureRenderContext doesn't provide access to:
        // - Source HDR texture
        // - Descriptor pool for dynamic allocation
        // - Sampler
        //
        // This is a placeholder that sets up the structure.
        // Full integration requires renderer-level changes to pass these resources.

        let mip_count = self.pass.mip_count();
        if mip_count == 0 {
            return;
        }

        // Bloom pipeline structure (for future implementation):
        //
        // 1. Prefilter Pass:
        //    - Input: HDR source texture
        //    - Output: Mip 0
        //    - Push constants: threshold, soft_knee
        //
        // 2. Downsample Chain (i = 0 to mip_count-2):
        //    - Input: Mip i
        //    - Output: Mip i+1
        //    - Push constants: texel_size for mip i+1
        //
        // 3. Upsample Chain (i = mip_count-1 down to 1):
        //    - Input: Mip i
        //    - Output: Mip i-1 (additive blend)
        //    - Push constants: texel_size for mip i-1, intensity
        //
        // Example command recording (when resources available):
        /*
        // Prefilter
        let pc = self.pass.get_prefilter_push_constants();
        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&pc),
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, prefilter_pipeline);
        // ... bind descriptor set, begin render pass, draw ...

        // Downsample loop
        for i in 0..mip_count-1 {
            if let Some(pc) = self.pass.get_downsample_push_constants(i) {
                device.cmd_push_constants(...);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, downsample_pipeline);
                // ... render to mip i+1 ...
            }
        }

        // Upsample loop (reverse order, with additive blending)
        for i in (1..mip_count).rev() {
            if let Some(pc) = self.pass.get_upsample_push_constants(mip_count - 1 - i) {
                device.cmd_push_constants(...);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, upsample_pipeline);
                // ... render to mip i-1 with additive blend ...
            }
        }
        */

        // Placeholder: Log that bloom would execute
        let _ = (
            device,
            cmd,
            prefilter_pipeline,
            downsample_pipeline,
            upsample_pipeline,
            pipeline_layout,
            render_pass,
        );
        log::trace!("Bloom render called (awaiting renderer integration)");
    }

    fn on_removed(&mut self, device: &Device) {
        unsafe {
            // Cleanup pipelines
            if let Some(pipeline) = self.prefilter_pipeline.take() {
                device.destroy_pipeline(pipeline, None);
            }
            if let Some(pipeline) = self.downsample_pipeline.take() {
                device.destroy_pipeline(pipeline, None);
            }
            if let Some(pipeline) = self.upsample_pipeline.take() {
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

        log::info!("Bloom feature removed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mip_chain_calculation() {
        let mut pass = BloomPass::new();
        pass.calculate_mip_chain(1920, 1080);

        // Default is 5 mips
        assert_eq!(pass.mip_count(), 5);

        // First mip should be half of 1920x1080
        let mip0 = pass.get_mip_info(0).unwrap();
        assert_eq!(mip0.width, 960);
        assert_eq!(mip0.height, 540);

        // Second mip
        let mip1 = pass.get_mip_info(1).unwrap();
        assert_eq!(mip1.width, 480);
        assert_eq!(mip1.height, 270);
    }

    #[test]
    fn test_push_constants() {
        let mut pass = BloomPass::with_config(BloomConfig {
            threshold: 1.2,
            soft_knee: 0.3,
            ..Default::default()
        });
        pass.calculate_mip_chain(1920, 1080);

        let pc = pass.get_prefilter_push_constants();
        assert!((pc.threshold - 1.2).abs() < 0.001);
        assert!((pc.soft_knee - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_small_resolution() {
        let mut pass = BloomPass::with_config(BloomConfig {
            mip_count: 10, // More than possible
            ..Default::default()
        });
        pass.calculate_mip_chain(32, 32);

        // Stop before reaching 10 mips if resolution reaches 1x1.
        assert!(pass.mip_count() < 10);

        // Last mip should be 1x1 or close
        let last = pass.get_mip_info(pass.mip_count() - 1).unwrap();
        assert!(last.width <= 2 && last.height <= 2);
    }
}
