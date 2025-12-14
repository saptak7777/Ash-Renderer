use ash::vk;
use bytemuck::Pod;
use std::io::Cursor;
use std::{collections::HashMap, ffi::CString, fs, path::PathBuf, sync::Arc, time::SystemTime};

use crate::renderer::resources::mesh::Vertex;
use crate::{AshError, Result};

use super::pipeline_state::PipelineState;

#[derive(Default, Clone)]
struct SpecializationData {
    entries: Vec<vk::SpecializationMapEntry>,
    data: Vec<u8>,
}

struct ShaderStage {
    module: vk::ShaderModule,
    stage: vk::ShaderStageFlags,
    entry_point: CString,
}

struct ShaderWatchRegistration {
    stage: vk::ShaderStageFlags,
    path: PathBuf,
    last_modified: Option<SystemTime>,
}

#[derive(Clone, Debug)]
pub struct ShaderWatchInfo {
    stage: vk::ShaderStageFlags,
    path: PathBuf,
    last_modified: Option<SystemTime>,
}

impl ShaderWatchInfo {
    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }
}

/// Graphics pipeline wrapper managed by the resource registry.
pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    device: Arc<ash::Device>,
    state: PipelineState,
    shader_watch: Vec<ShaderWatchInfo>,
    managed_by_registry: bool,
}

impl Pipeline {
    pub fn builder(device: Arc<ash::Device>) -> PipelineBuilder {
        PipelineBuilder::new(device)
    }

    pub fn state(&self) -> &PipelineState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut PipelineState {
        &mut self.state
    }

    pub fn shader_watch_info(&self) -> &[ShaderWatchInfo] {
        &self.shader_watch
    }

    pub fn detect_shader_changes(&mut self) -> Result<bool> {
        let mut changed = false;
        for entry in &mut self.shader_watch {
            if let Ok(metadata) = fs::metadata(&entry.path) {
                if let Ok(modified) = metadata.modified() {
                    let is_newer = entry
                        .last_modified
                        .map(|prev| modified > prev)
                        .unwrap_or(true);
                    if is_newer {
                        log::info!(
                            "Shader change detected ({:?}): {}",
                            entry.stage(),
                            entry.path.display()
                        );
                        entry.last_modified = Some(modified);
                        changed = true;
                    }
                }
            }
        }
        Ok(changed)
    }

    pub fn mark_managed_by_registry(&mut self) {
        self.managed_by_registry = true;
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.managed_by_registry {
            return;
        }

        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.pipeline, None);
            }
        }
    }
}

/// Declarative pipeline builder mirroring the reference abstraction.
pub struct PipelineBuilder {
    device: Arc<ash::Device>,
    layout: Option<vk::PipelineLayout>,
    render_pass: Option<vk::RenderPass>,
    extent: Option<vk::Extent2D>,
    pipeline_cache: Option<vk::PipelineCache>,
    subpass: u32,
    shader_stages: Vec<ShaderStage>,
    shader_watch: Vec<ShaderWatchRegistration>,
    specialization: HashMap<vk::ShaderStageFlags, SpecializationData>,
    vertex_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
    vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo<'static>,
    rasterization: vk::PipelineRasterizationStateCreateInfo<'static>,
    multisample_cfg: MultisampleConfig,
    depth_stencil: Option<vk::PipelineDepthStencilStateCreateInfo<'static>>,
    color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    dynamic_states: Vec<vk::DynamicState>,
}

impl PipelineBuilder {
    fn new(device: Arc<ash::Device>) -> Self {
        let binding = Vertex::binding_description();
        let attributes = Vertex::attribute_descriptions();

        Self {
            device,
            layout: None,
            render_pass: None,
            extent: None,
            pipeline_cache: None,
            subpass: 0,
            shader_stages: Vec::new(),
            shader_watch: Vec::new(),
            specialization: HashMap::new(),
            vertex_binding_descriptions: vec![binding],
            vertex_attribute_descriptions: attributes.to_vec(),
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false),
            rasterization: vk::PipelineRasterizationStateCreateInfo::default()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0)
                .depth_bias_enable(false),
            multisample_cfg: MultisampleConfig::default(),
            depth_stencil: None,
            color_blend_attachments: vec![vk::PipelineColorBlendAttachmentState {
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            }],
            dynamic_states: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
        }
    }

    pub fn with_layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_render_pass(mut self, render_pass: vk::RenderPass) -> Self {
        self.render_pass = Some(render_pass);
        self
    }

    pub fn with_extent(mut self, extent: vk::Extent2D) -> Self {
        self.extent = Some(extent);
        self
    }

    pub fn with_pipeline_cache(mut self, cache: vk::PipelineCache) -> Self {
        if cache != vk::PipelineCache::null() {
            self.pipeline_cache = Some(cache);
        }
        self
    }

    pub fn with_subpass(mut self, subpass: u32) -> Self {
        self.subpass = subpass;
        self
    }

    pub fn with_depth_format(mut self, format: vk::Format) -> Self {
        self.depth_stencil = Some(
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0),
        );

        // If the format includes stencil, enable read/write masks.
        if matches!(
            format,
            vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
                | vk::Format::D16_UNORM_S8_UINT
        ) {
            if let Some(ref mut state) = self.depth_stencil {
                state.stencil_test_enable = vk::TRUE;
                let stencil = vk::StencilOpState {
                    fail_op: vk::StencilOp::KEEP,
                    pass_op: vk::StencilOp::KEEP,
                    depth_fail_op: vk::StencilOp::KEEP,
                    compare_op: vk::CompareOp::ALWAYS,
                    compare_mask: 0xff,
                    write_mask: 0xff,
                    reference: 0,
                };
                state.front = stencil;
                state.back = stencil;
            }
        }

        self
    }

    pub fn add_shader_from_path(self, path: &str, stage: vk::ShaderStageFlags) -> Result<Self> {
        self.add_shader_with_options(path, stage, false)
    }

    pub fn add_shader_from_bytes(
        mut self,
        code: &[u8],
        stage: vk::ShaderStageFlags,
        entry_point: &str,
    ) -> Result<Self> {
        if code.len() % 4 != 0 {
            return Err(AshError::VulkanError(
                "Shader code size must be multiple of 4".to_string(),
            ));
        }

        let code_u32 = ash::util::read_spv(&mut Cursor::new(code))
            .map_err(|e| AshError::VulkanError(format!("Failed to parse SPIR-V: {e}")))?;

        let module = unsafe {
            let create_info = vk::ShaderModuleCreateInfo::default().code(&code_u32);
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create shader module: {e}"))
                })?
        };

        self.shader_stages.push(ShaderStage {
            module,
            stage,
            entry_point: CString::new(entry_point).unwrap(),
        });

        Ok(self)
    }

    pub fn add_shader_with_options(
        mut self,
        path: &str,
        stage: vk::ShaderStageFlags,
        watch: bool,
    ) -> Result<Self> {
        let code = fs::read(path)
            .map_err(|e| AshError::VulkanError(format!("Failed to read shader {path}: {e}")))?;

        if code.len() % 4 != 0 {
            return Err(AshError::VulkanError(format!(
                "Shader {path} size must be multiple of 4"
            )));
        }

        let code_u32 = ash::util::read_spv(&mut Cursor::new(&code))
            .map_err(|e| AshError::VulkanError(format!("Failed to parse SPIR-V: {e}")))?;

        let module = unsafe {
            let create_info = vk::ShaderModuleCreateInfo::default().code(&code_u32);
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create shader module: {e}"))
                })?
        };

        self.shader_stages.push(ShaderStage {
            module,
            stage,
            entry_point: CString::new("main").unwrap(),
        });

        if watch {
            let metadata = fs::metadata(path).ok();
            let last_modified = metadata.and_then(|m| m.modified().ok());
            self.shader_watch.push(ShaderWatchRegistration {
                stage,
                path: PathBuf::from(path),
                last_modified,
            });
        }

        Ok(self)
    }

    pub fn with_vertex_input(
        mut self,
        bindings: Vec<vk::VertexInputBindingDescription>,
        attributes: Vec<vk::VertexInputAttributeDescription>,
    ) -> Self {
        self.vertex_binding_descriptions = bindings;
        self.vertex_attribute_descriptions = attributes;
        self
    }

    pub fn with_dynamic_states(mut self, states: Vec<vk::DynamicState>) -> Self {
        self.dynamic_states = states;
        self
    }

    pub fn with_cull_mode(mut self, cull_mode: vk::CullModeFlags) -> Self {
        self.rasterization.cull_mode = cull_mode;
        self
    }

    pub fn with_multisampling(mut self, config: MultisampleConfig) -> Self {
        self.multisample_cfg = config;
        self
    }

    pub fn with_specialization_constant<T: Pod>(
        self,
        stage: vk::ShaderStageFlags,
        constant_id: u32,
        value: &T,
    ) -> Self {
        self.with_specialization_bytes(stage, constant_id, bytemuck::bytes_of(value))
    }

    pub fn with_specialization_bytes(
        mut self,
        stage: vk::ShaderStageFlags,
        constant_id: u32,
        data: &[u8],
    ) -> Self {
        let entry = self.specialization.entry(stage).or_default();
        let offset = entry.data.len() as u32;
        entry.data.extend_from_slice(data);
        entry.entries.push(vk::SpecializationMapEntry {
            constant_id,
            offset,
            size: data.len(),
        });
        self
    }

    pub fn build(mut self) -> Result<Pipeline> {
        let layout = self
            .layout
            .ok_or_else(|| AshError::VulkanError("Pipeline layout not specified".to_string()))?;
        let render_pass = self
            .render_pass
            .ok_or_else(|| AshError::VulkanError("Render pass not specified".to_string()))?;
        let extent = self
            .extent
            .ok_or_else(|| AshError::VulkanError("Viewport extent not specified".to_string()))?;

        if self.shader_stages.is_empty() {
            return Err(AshError::VulkanError(
                "At least one shader stage must be provided".to_string(),
            ));
        }

        // Pre-create specialization infos to ensure they have a stable address
        // when referenced by the pipeline stage create infos.
        let specialization_infos: Vec<Option<vk::SpecializationInfo>> = self
            .shader_stages
            .iter()
            .map(|stage| {
                self.specialization.get(&stage.stage).map(|spec| {
                    vk::SpecializationInfo::default()
                        .map_entries(&spec.entries)
                        .data(&spec.data)
                })
            })
            .collect();

        let stage_infos: Vec<_> = self
            .shader_stages
            .iter()
            .zip(specialization_infos.iter())
            .map(|(stage, spec_opt)| {
                let mut info = vk::PipelineShaderStageCreateInfo::default()
                    .module(stage.module)
                    .stage(stage.stage)
                    .name(&stage.entry_point);

                if let Some(spec) = spec_opt {
                    info = info.specialization_info(spec);
                }

                info
            })
            .collect();

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.vertex_binding_descriptions)
            .vertex_attribute_descriptions(&self.vertex_attribute_descriptions);

        let dynamic_state_info = if self.dynamic_states.is_empty() {
            None
        } else {
            Some(vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic_states))
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&self.color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(self.multisample_cfg.sample_count)
            .sample_shading_enable(self.multisample_cfg.enable_sample_shading)
            .min_sample_shading(self.multisample_cfg.min_sample_shading);

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stage_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&self.rasterization)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(self.subpass)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        if let Some(ref depth_stencil) = self.depth_stencil {
            pipeline_info = pipeline_info.depth_stencil_state(depth_stencil);
        }

        if let Some(ref dynamic_state) = dynamic_state_info {
            pipeline_info = pipeline_info.dynamic_state(dynamic_state);
        }

        let cache = self.pipeline_cache.unwrap_or(vk::PipelineCache::null());
        let pipeline = unsafe {
            match self
                .device
                .create_graphics_pipelines(cache, &[pipeline_info], None)
            {
                Ok(pipelines) => pipelines[0],
                Err((_, e)) => {
                    self.destroy_shader_modules();
                    return Err(AshError::VulkanError(format!(
                        "Failed to create graphics pipeline: {e}"
                    )));
                }
            }
        };

        self.destroy_shader_modules();

        let state = PipelineState::new()
            .with_viewport(viewport)
            .with_scissor(scissor);

        let shader_watch = self
            .shader_watch
            .iter()
            .map(|entry| ShaderWatchInfo {
                stage: entry.stage,
                path: entry.path.clone(),
                last_modified: entry.last_modified,
            })
            .collect();

        Ok(Pipeline {
            pipeline,
            device: Arc::clone(&self.device),
            state,
            shader_watch,
            managed_by_registry: false,
        })
    }
}

impl PipelineBuilder {
    fn destroy_shader_modules(&mut self) {
        for stage in self.shader_stages.drain(..) {
            unsafe {
                self.device.destroy_shader_module(stage.module, None);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MultisampleConfig {
    pub sample_count: vk::SampleCountFlags,
    pub enable_sample_shading: bool,
    pub min_sample_shading: f32,
}

impl Default for MultisampleConfig {
    fn default() -> Self {
        Self {
            sample_count: vk::SampleCountFlags::TYPE_1,
            enable_sample_shading: false,
            min_sample_shading: 0.0,
        }
    }
}
