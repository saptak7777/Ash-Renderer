use crate::{
    renderer::{
        features::ShadowSystem,
        passes::hiz::{AdaptiveHiZManager, HiZPass},
        passes::vsr::VsrPass,
        systems::post_process::PostProcessSystem,
        HdrSystem, Scene,
    },
    Result,
};
use ash::vk;
use glam::Vec3;
use std::sync::{Arc, RwLock};

/// RenderPipeline orchestrates the high-level rendering flow.
/// It owns the major rendering subsystems and manages their execution order.
pub struct RenderPipeline {
    shadow_system: Option<ShadowSystem>,
    post_process: PostProcessSystem,
    pub(crate) hiz_pass: Option<Arc<RwLock<HiZPass>>>, // Keep crate-public for Renderer access for now
    adaptive_hiz_manager: AdaptiveHiZManager,
}

impl RenderPipeline {
    pub fn new(
        shadow_system: Option<ShadowSystem>,
        post_process: PostProcessSystem,
        hiz_pass: Option<Arc<RwLock<HiZPass>>>,
        adaptive_hiz_manager: AdaptiveHiZManager,
    ) -> Self {
        Self {
            shadow_system,
            post_process,
            hiz_pass,
            adaptive_hiz_manager,
        }
    }

    /// Execute the Hi-Z depth pyramid construction pass.
    /// This is typically done at the start of the frame after the previous frame's depth is available.
    ///
    /// # Lock Ordering
    /// This method acquires locks in the following order:
    /// 1. `hiz_pass` (write lock)
    /// 2. `indirect_draw_pass` (write lock, if provided)
    ///
    /// Any other code that acquires both locks must follow this same ordering
    /// to avoid deadlocks.
    pub fn execute_hiz_pass(
        &mut self,
        command_buffer: vk::CommandBuffer,
        depth_image: vk::Image,
        gpu_profiler: Option<&crate::renderer::diagnostics::GpuProfiler>,
        indirect_draw_pass: Option<&Arc<RwLock<crate::renderer::vcgs::IndirectDrawPass>>>,
        black_texture_view: vk::ImageView,
        black_texture_sampler: vk::Sampler,
    ) -> Result<()> {
        if let Some(ref hiz_arc) = self.hiz_pass {
            let mut hiz = hiz_arc.write().map_err(|e| {
                log::error!("Hi-Z pass RwLock poisoned: {}", e);
                crate::AshError::VulkanError("Hi-Z pass RwLock poisoned".into())
            })?;

            // Update adaptive quality based on previous frame's metrics
            if let Some(profiler) = gpu_profiler {
                let timings = profiler.last_extended_timings();
                if timings.valid {
                    let hiz_time_ms = timings.hiz_generate_ms as f64;
                    if let Some(new_quality) =
                        self.adaptive_hiz_manager.update(hiz.quality(), hiz_time_ms)
                    {
                        hiz.set_quality(new_quality);
                    }
                }
            }

            // Build Hi-Z pyramid from depth buffer
            unsafe {
                hiz.build_pyramid(command_buffer, depth_image)?;
            }

            // Update descriptors for systems that depend on Hi-Z (like occlusion culling)
            let hiz_view = hiz.hiz_view().unwrap_or(black_texture_view);
            let hiz_sampler = if hiz.is_initialized() {
                hiz.hiz_sampler()
            } else {
                black_texture_sampler
            };

            if let Some(indirect_arc) = indirect_draw_pass {
                let indirect = indirect_arc.write().map_err(|e| {
                    log::error!("Indirect draw pass RwLock poisoned: {}", e);
                    crate::AshError::VulkanError("Indirect draw pass RwLock poisoned".into())
                })?;
                unsafe {
                    indirect.update_hiz_descriptor(hiz_view, hiz_sampler);
                }
            }

            if let Some(profiler) = gpu_profiler {
                unsafe {
                    profiler.write_timestamp(
                        command_buffer,
                        crate::renderer::diagnostics::TimingScope::HiZGenerateEnd,
                    );
                }
            }
        }
        Ok(())
    }

    /// Render shadow maps for the current frame.
    /// This implementation currently supports Virtual Shadow Maps (VSM).
    pub fn render_shadows(
        &mut self,
        command_buffer: vk::CommandBuffer,
        scene: &Scene,
        frame_index: usize,
        device: &ash::Device,
        bindless_descriptor_set: vk::DescriptorSet,
        uniform_buffer_address: u64,
        instance_buffer_address: u64,
        material_heap_address: u64,
        light_ptr: u64,
        tile_ptr: u64,
        all_instances_len: usize,
    ) -> Result<()> {
        if let Some(shadow_system) = &mut self.shadow_system {
            if shadow_system
                .vsm_feature()
                .shadow_pipeline_layout()
                .is_some()
            {
                let light_dir = Vec3::from_slice(&scene.scene_lighting.directional.direction[0..3]);

                let vertex_ptr = scene.model_renderer.geometry_buffer.vertex_heap_address();
                let index_ptr = scene.model_renderer.geometry_buffer.index_heap_address();

                if vertex_ptr == 0 || index_ptr == 0 {
                    log::warn!("Shadow pass: Invalid BDA pointers. Skipping.");
                    return Ok(());
                }
                let vsm_feature = shadow_system.vsm_feature();
                unsafe {
                    vsm_feature.render_shadows(
                        command_buffer,
                        light_dir,
                        all_instances_len as u32,
                        frame_index,
                        vk::DescriptorSet::null(),
                        bindless_descriptor_set,
                        uniform_buffer_address,
                        vertex_ptr,
                        instance_buffer_address,
                        material_heap_address,
                        index_ptr,
                        light_ptr,
                        tile_ptr,
                        scene.transform_system.arena_addr,
                        0,
                    );
                }

                // Synchronization barrier for VSM results
                let vsm_barrier = vk::ImageMemoryBarrier::default()
                    .image(shadow_system.vsm_feature().resources.physical_cache)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[vsm_barrier],
                    );
                }
            }
        }
        Ok(())
    }

    /// Execute post-processing effects including blooming, tonemapping, and upscaling resolve.
    pub fn render_post_process(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        swapchain_extent: vk::Extent2D,
        hdr_system: Option<&HdrSystem>,
        vsr_pass: Option<&VsrPass>,
        black_texture_view: vk::ImageView,
    ) -> Result<()> {
        // Update post-processing descriptors with current frame outputs
        if let Some(hdr) = hdr_system {
            let input_view = vsr_pass
                .map(|vsr| vsr.active_view())
                .unwrap_or_else(|| hdr.view());

            self.post_process.update_descriptor_set(
                image_index,
                input_view,
                black_texture_view, // bloom_view placeholder
                black_texture_view, // ssgi_view placeholder
                hdr.sampler(),
            );
        }

        // Final tonemapping and swapchain resolve
        self.post_process
            .render(command_buffer, image_index, swapchain_extent)?;
        Ok(())
    }

    // Accessors
    pub fn shadow_system(&self) -> Option<&ShadowSystem> {
        self.shadow_system.as_ref()
    }

    pub fn shadow_system_mut(&mut self) -> Option<&mut ShadowSystem> {
        self.shadow_system.as_mut()
    }

    pub fn take_shadow_system(&mut self) -> Option<ShadowSystem> {
        self.shadow_system.take()
    }

    pub fn post_process(&self) -> &PostProcessSystem {
        &self.post_process
    }

    pub fn post_process_mut(&mut self) -> &mut PostProcessSystem {
        &mut self.post_process
    }

    pub fn hiz_pass(&self) -> Option<&Arc<RwLock<HiZPass>>> {
        self.hiz_pass.as_ref()
    }
}
