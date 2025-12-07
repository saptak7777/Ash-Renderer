use std::collections::HashMap;

use ash::vk;

use crate::vulkan::PipelineState;

/// Minimal frame graph for tracking named passes and their pipelines.
#[derive(Default)]
pub struct FrameGraph {
    passes: HashMap<String, FrameGraphPass>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            passes: HashMap::new(),
        }
    }

    pub fn register_pass(&mut self, pass: FrameGraphPass) {
        self.passes.insert(pass.name.clone(), pass);
    }

    pub fn pass(&self, name: &str) -> Option<&FrameGraphPass> {
        self.passes.get(name)
    }

    pub fn pass_mut(&mut self, name: &str) -> Option<&mut FrameGraphPass> {
        self.passes.get_mut(name)
    }
}

/// A single render pass node within the frame graph.
#[derive(Clone)]
pub struct FrameGraphPass {
    name: String,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    dynamic_state: PipelineState,
}

impl FrameGraphPass {
    pub fn new(
        name: impl Into<String>,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        dynamic_state: PipelineState,
    ) -> Self {
        Self {
            name: name.into(),
            pipeline,
            pipeline_layout,
            render_pass,
            dynamic_state,
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn dynamic_state(&self) -> &PipelineState {
        &self.dynamic_state
    }

    pub fn dynamic_state_mut(&mut self) -> &mut PipelineState {
        &mut self.dynamic_state
    }
}
