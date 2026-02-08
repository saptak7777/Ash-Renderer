use crate::renderer::passes::fullscreen::FullscreenPass;
use crate::renderer::passes::fullscreen::PostProcessPushConstants;
use crate::vulkan;
use crate::Result;
use ash::vk;
use bytemuck;
use std::sync::Arc;

/// Configuration for post-processing effects.
#[derive(Clone, Copy, Debug)]
pub struct PostProcessConfig {
    pub tonemapping_enabled: bool,
    pub exposure: f32,
    pub gamma: f32,
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            tonemapping_enabled: true,
            exposure: 1.2,
            gamma: 2.2,
            bloom_enabled: true,
            bloom_intensity: 0.1,
        }
    }
}

/// High-level system for managing post-processing effects and resources.
/// Wraps FullscreenPass and manages its own pipelines and descriptors.
pub struct PostProcessSystem {
    device: Arc<ash::Device>,
    fullscreen_pass: FullscreenPass,
    pub config: PostProcessConfig,

    // Resource Management
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline: Option<vulkan::Pipeline>,
    framebuffers: Vec<vulkan::Framebuffer>,
}

impl PostProcessSystem {
    pub fn new(
        device: Arc<ash::Device>,
        image_count: usize,
        _swapchain_extent: vk::Extent2D,
        output_format: vk::Format,
    ) -> Result<Self> {
        log::info!("Initializing PostProcessSystem");

        let final_layout = vk::ImageLayout::PRESENT_SRC_KHR; // Standard for swapchain output

        let fullscreen_pass =
            unsafe { FullscreenPass::new(Arc::clone(&device), output_format, final_layout)? };

        // Initialize with default pool, sets, and pipeline (Step 1 placeholder)
        Ok(Self {
            device,
            fullscreen_pass,
            config: PostProcessConfig::default(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_sets: Vec::with_capacity(image_count),
            pipeline: None,
            framebuffers: Vec::new(),
        })
    }

    pub fn resize(&mut self, output_views: &[vk::ImageView], extent: vk::Extent2D) -> Result<()> {
        let image_count = output_views.len();

        // Pipeline cleanup handled by RAII in vulkan::Pipeline
        self.pipeline = None;

        // Framebuffers cleaned up by RAII
        self.framebuffers.clear();

        let pipeline = match self
            .fullscreen_pass
            .create_pipeline(&self.device, extent, "main")
        {
            Ok(p) => p,
            Err(e) => {
                log::error!("Failed to create post-process pipeline: {}", e);
                return Err(e);
            }
        };
        self.pipeline = Some(pipeline);

        // Create framebuffers
        for &view in output_views {
            let framebuffer = vulkan::Framebuffer::new(
                Arc::clone(&self.device),
                self.fullscreen_pass.render_pass(),
                &[view],
                extent,
            )?;
            self.framebuffers.push(framebuffer);
        }

        if self.descriptor_sets.len() != image_count {
            self.reallocate_descriptors(image_count)?;
        }

        Ok(())
    }

    fn reallocate_descriptors(&mut self, count: usize) -> Result<()> {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }

            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (count * 3) as u32,
            }];

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(count as u32);

            self.descriptor_pool = self.device.create_descriptor_pool(&pool_info, None)?;

            let layouts = vec![self.fullscreen_pass.descriptor_set_layout(); count];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);

            self.descriptor_sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        }
        Ok(())
    }

    pub fn update_descriptor_set(
        &self,
        image_index: usize,
        input_view: vk::ImageView,
        bloom_view: vk::ImageView,
        ssgi_view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        if image_index >= self.descriptor_sets.len() {
            return;
        }

        let input_info = vk::DescriptorImageInfo::default()
            .image_view(input_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler);

        let bloom_info = vk::DescriptorImageInfo::default()
            .image_view(bloom_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler);

        let ssgi_info = vk::DescriptorImageInfo::default()
            .image_view(ssgi_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .sampler(sampler);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[image_index])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&input_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[image_index])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&bloom_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[image_index])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&ssgi_info)),
        ];

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn render(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        extent: vk::Extent2D,
    ) -> Result<()> {
        if self.pipeline.is_none()
            || self.descriptor_sets.is_empty()
            || self.framebuffers.is_empty()
        {
            return Ok(());
        }

        let pipeline_wrapper = self.pipeline.as_ref().unwrap();
        let pipeline = pipeline_wrapper.pipeline; // helper from wrapper

        // Handle case where image_index is out of bounds
        if image_index >= self.descriptor_sets.len() || image_index >= self.framebuffers.len() {
            return Ok(());
        }
        let descriptor_set = self.descriptor_sets[image_index];
        let framebuffer = &self.framebuffers[image_index];

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.fullscreen_pass.render_pass())
            .framebuffer(framebuffer.handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.fullscreen_pass.pipeline_layout(),
                0,
                &[descriptor_set],
                &[],
            );

            let push_constants = PostProcessPushConstants {
                exposure: self.config.exposure,
                gamma: self.config.gamma,
                bloom_intensity: if self.config.bloom_enabled {
                    self.config.bloom_intensity
                } else {
                    0.0
                },
                tonemapping_enabled: if self.config.tonemapping_enabled {
                    1.0
                } else {
                    0.0
                },
            };

            self.device.cmd_push_constants(
                command_buffer,
                self.fullscreen_pass.pipeline_layout(),
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );

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
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            self.device.cmd_end_render_pass(command_buffer);
        }

        Ok(())
    }
}

impl Drop for PostProcessSystem {
    fn drop(&mut self) {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            // Pipeline is dropped automatically by RAII
        }
    }
}
