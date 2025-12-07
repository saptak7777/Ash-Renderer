use ash::vk;

/// Encapsulates the dynamic state applied alongside a graphics pipeline.
#[derive(Debug, Clone)]
pub struct PipelineState {
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    blend_constants: [f32; 4],
    line_width: f32,
    depth_bias_constant: f32,
    depth_bias_clamp: f32,
    depth_bias_slope: f32,
    stencil_reference: u32,
    stencil_compare_mask: u32,
    stencil_write_mask: u32,
    dynamic_viewport: bool,
    dynamic_scissor: bool,
    dynamic_blend_constants: bool,
    dynamic_line_width: bool,
    dynamic_depth_bias: bool,
    dynamic_stencil_reference: bool,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            viewport: vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: 0, height: 0 },
            },
            blend_constants: [0.0; 4],
            line_width: 1.0,
            depth_bias_constant: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope: 0.0,
            stencil_reference: 0,
            stencil_compare_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            dynamic_viewport: false,
            dynamic_scissor: false,
            dynamic_blend_constants: false,
            dynamic_line_width: false,
            dynamic_depth_bias: false,
            dynamic_stencil_reference: false,
        }
    }
}

impl PipelineState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_viewport(mut self, viewport: vk::Viewport) -> Self {
        self.viewport = viewport;
        self.dynamic_viewport = true;
        self
    }

    pub fn with_scissor(mut self, scissor: vk::Rect2D) -> Self {
        self.scissor = scissor;
        self.dynamic_scissor = true;
        self
    }

    pub fn with_blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend_constants = constants;
        self.dynamic_blend_constants = true;
        self
    }

    pub fn with_line_width(mut self, line_width: f32) -> Self {
        self.line_width = line_width;
        self.dynamic_line_width = true;
        self
    }

    pub fn with_depth_bias(mut self, constant: f32, clamp: f32, slope: f32) -> Self {
        self.depth_bias_constant = constant;
        self.depth_bias_clamp = clamp;
        self.depth_bias_slope = slope;
        self.dynamic_depth_bias = true;
        self
    }

    pub fn with_stencil_reference(
        mut self,
        reference: u32,
        compare_mask: u32,
        write_mask: u32,
    ) -> Self {
        self.stencil_reference = reference;
        self.stencil_compare_mask = compare_mask;
        self.stencil_write_mask = write_mask;
        self.dynamic_stencil_reference = true;
        self
    }

    pub fn viewport(&self) -> vk::Viewport {
        self.viewport
    }

    pub fn scissor(&self) -> vk::Rect2D {
        self.scissor
    }

    pub fn blend_constants(&self) -> [f32; 4] {
        self.blend_constants
    }

    pub fn line_width(&self) -> f32 {
        self.line_width
    }

    pub fn depth_bias(&self) -> (f32, f32, f32) {
        (
            self.depth_bias_constant,
            self.depth_bias_clamp,
            self.depth_bias_slope,
        )
    }

    pub fn stencil_reference(&self) -> u32 {
        self.stencil_reference
    }

    pub fn stencil_compare_mask(&self) -> u32 {
        self.stencil_compare_mask
    }

    pub fn stencil_write_mask(&self) -> u32 {
        self.stencil_write_mask
    }

    pub fn dynamic_viewport(&self) -> bool {
        self.dynamic_viewport
    }

    pub fn dynamic_scissor(&self) -> bool {
        self.dynamic_scissor
    }

    pub fn dynamic_blend_constants(&self) -> bool {
        self.dynamic_blend_constants
    }

    pub fn dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }

    pub fn dynamic_depth_bias(&self) -> bool {
        self.dynamic_depth_bias
    }

    pub fn dynamic_stencil_reference(&self) -> bool {
        self.dynamic_stencil_reference
    }

    /// Applies the configured non-viewport dynamic state to the provided command buffer.
    ///
    /// # Safety
    /// Caller must ensure the command buffer is recording for a compatible pipeline and remains
    /// valid for the duration of the call.
    pub unsafe fn apply(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if self.dynamic_blend_constants {
            device.cmd_set_blend_constants(command_buffer, &self.blend_constants);
        }

        if self.dynamic_line_width {
            device.cmd_set_line_width(command_buffer, self.line_width);
        }

        if self.dynamic_depth_bias {
            device.cmd_set_depth_bias(
                command_buffer,
                self.depth_bias_constant,
                self.depth_bias_clamp,
                self.depth_bias_slope,
            );
        }

        if self.dynamic_stencil_reference {
            device.cmd_set_stencil_reference(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                self.stencil_reference,
            );
            device.cmd_set_stencil_compare_mask(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                self.stencil_compare_mask,
            );
            device.cmd_set_stencil_write_mask(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                self.stencil_write_mask,
            );
        }
    }
}
