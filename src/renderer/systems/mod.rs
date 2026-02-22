pub mod culling;
pub mod lighting_system;
pub mod post_process;

use crate::renderer::features::AutoRotateFeature;
use crate::renderer::systems::culling::CullingSystem;
use crate::renderer::{
    HdrSystem, PipelineCache,
    context::Context,
    diagnostics::{DiagnosticsOverlay, DiagnosticsState, FrameProfiler, GpuProfiler},
    features::FeatureManager,
    frame::Frame,
    passes::{
        SkyboxPass,
        motion::MotionVectorPass,
        temporal_aa::{ConfigMetrics, TaaConfig},
    },
    render_pipeline::RenderPipeline,
    resource_registry::ResourceId,
    resources::Resources,
    types::{DebugMode, RendererConfig},
};
use crate::{AshError, Result, vulkan};
use ash::vk;
use std::sync::{Arc, RwLock};

/// Systems manages high-level rendering logic and pipeline state.
pub struct Systems {
    pub pipeline: RenderPipeline,
    pub culling: CullingSystem,
    pub skybox_pass: Option<SkyboxPass>,

    // Features & Config
    pub features: FeatureManager,
    pub motion_pass: Option<MotionVectorPass>,
    pub hdr_system: Option<HdrSystem>,

    // Cache & IDs
    pub pipeline_cache: PipelineCache,
    pub pipeline_id: Option<ResourceId>,
    pub pipeline_layout_id: Option<ResourceId>,

    // Diagnostics & Performance
    pub diagnostics: DiagnosticsState,
    pub frame_profiler: FrameProfiler,
    pub gpu_profiler: Option<GpuProfiler>,
    pub diagnostics_overlay: DiagnosticsOverlay,

    // Configuration
    pub debug_mode: DebugMode,
    pub strict_mode: bool,
    pub sample_shading: crate::renderer::types::SampleShadingQuality,
    pub taa_config: TaaConfig,
    pub taa_config_metrics: ConfigMetrics,
    /// Previous frame's jitter offset (UV space) for TAA reprojection.
    pub prev_jitter_uv: [f32; 2],

    // Owned Systems
    /// Authoritative owner of scene lighting state and the Forward+ culling pipeline.
    pub lighting: lighting_system::LightingSystem,
}

impl Systems {
    /// Initializes all high-level rendering systems.
    pub fn new(
        context: &Context,
        resources: &mut Resources,
        frame: &mut Frame,
        pipeline_cache: PipelineCache,
        _width: u32,
        _height: u32,
        config: &RendererConfig,
    ) -> Result<Self> {
        log::info!("Initializing Systems");

        let mut features = FeatureManager::new();
        features.set_device(Arc::clone(&context.device.device));
        features.add_feature(AutoRotateFeature::new());

        let lighting = resources.lighting.take().ok_or_else(|| {
            AshError::VulkanError("Lighting system not found in ResourceRegistry".to_string())
        })?;
        let post = resources.post.take().ok_or_else(|| {
            AshError::VulkanError(
                "Post-processing system not found in ResourceRegistry".to_string(),
            )
        })?;
        let mut passes = resources.passes.take().ok_or_else(|| {
            AshError::VulkanError("PassRegistry not found in ResourceRegistry".to_string())
        })?;
        let pipelines = resources.pipelines.take().ok_or_else(|| {
            AshError::VulkanError("PipelineRegistry not found in ResourceRegistry".to_string())
        })?;

        let indirect_draw_pass = Some(Arc::new(RwLock::new(
            passes
                .indirect_draw_pass
                .take()
                .ok_or_else(|| AshError::VulkanError("Indirect Draw Pass not found".to_string()))?,
        )));

        // Promote ForwardPlusIntegration into a shared Arc so both RenderPipeline
        // (which needs it for descriptor binding) and LightingSystem (which owns updates)
        // can refer to the same GPU state without copying.
        let forward_plus_arc = Arc::new(RwLock::new(lighting.forward_plus));

        let systems = Self {
            pipeline: RenderPipeline::new(
                post.post_process,
                Some(Arc::new(RwLock::new(passes.hiz_pass.take().ok_or_else(
                    || AshError::VulkanError("HiZ Pass not found".to_string()),
                )?))),
                Some(Arc::clone(&forward_plus_arc)),
                indirect_draw_pass.clone(),
                Some(pipelines.pipeline),
                Some(pipelines.layout),
            ),

            culling: CullingSystem::new(indirect_draw_pass),
            skybox_pass: passes.skybox_pass.take(),
            features,
            motion_pass: None,
            hdr_system: None,
            pipeline_cache,
            pipeline_id: Some(pipelines.pipeline_id),
            pipeline_layout_id: Some(pipelines.layout_id),
            diagnostics: DiagnosticsState::default(),
            frame_profiler: FrameProfiler::new(),
            gpu_profiler: None,
            diagnostics_overlay: DiagnosticsOverlay::new(),
            debug_mode: DebugMode::default(),
            strict_mode: config.strict_mode,
            sample_shading: config.pipeline.sample_shading,
            taa_config: TaaConfig::default(),
            taa_config_metrics: ConfigMetrics::default(),
            prev_jitter_uv: [0.0, 0.0],
            lighting: lighting_system::LightingSystem::new(forward_plus_arc),
        };

        frame.gbuffer_indices = Some(passes.gbuffer_indices);

        Ok(systems)
    }

    /// Access the post-processing system.
    pub fn post_process(&self) -> &post_process::PostProcessSystem {
        &self.pipeline.post_process
    }

    /// Access the post-processing system mutably.
    pub fn post_process_mut(&mut self) -> &mut post_process::PostProcessSystem {
        &mut self.pipeline.post_process
    }

    /// Access the lighting system.
    pub fn lighting(&self) -> &lighting_system::LightingSystem {
        &self.lighting
    }

    /// Access the lighting system mutably.
    pub fn lighting_mut(&mut self) -> &mut lighting_system::LightingSystem {
        &mut self.lighting
    }

    /// Initialize motion vector pass for VSR/TAA
    ///
    /// # Safety
    /// Must be called after GBuffer is initialized
    pub unsafe fn init_motion_pass(
        &mut self,
        context: &Context,
        resources: &Resources,
    ) -> Result<()> {
        if self.motion_pass.is_some() {
            return Ok(()); // Already initialized
        }

        let _gbuffer = resources.gbuffer.as_ref().ok_or(AshError::VulkanError(
            "GBuffer must be initialized before motion pass".to_string(),
        ))?;

        let mut motion_pass = MotionVectorPass::new(Arc::clone(&context.device.device));

        // Initialize with G-Buffer motion format
        let motion_format = vk::Format::R16G16_SFLOAT;
        unsafe { motion_pass.init(&context.device, motion_format)? };

        self.motion_pass = Some(motion_pass);

        log::info!("Motion vector pass initialized");

        Ok(())
    }

    /// Extracted resize logic for Pipelines and Passes.
    pub fn resize(
        &mut self,
        context: &Context,
        resources: &mut Resources,
        width: u32,
        height: u32,
        swapchain_format: vk::Format,
        image_count: usize,
    ) -> Result<()> {
        let extent = vk::Extent2D { width, height };

        // --- 2. Recreate Main Graphics Pipeline ---
        log::info!("Recompiling pipeline due to resize/shader change...");
        let layout = self
            .pipeline
            .pipeline_layout
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Pipeline layout missing".to_string()))?
            .handle();

        // Determine color format (HDR or swapchain)
        let color_format = if let Some(hdr) = &self.hdr_system {
            hdr.format()
        } else {
            swapchain_format
        };

        let cache = self.pipeline_cache.handle();
        let depth_format = resources
            .depth_buffer
            .as_ref()
            .ok_or(AshError::VulkanError("Depth buffer missing".into()))?
            .format();

        let multisample_config = vulkan::MultisampleConfig {
            sample_count: vk::SampleCountFlags::TYPE_1,
            enable_sample_shading: self.sample_shading.enabled(),
            min_sample_shading: self.sample_shading.min_sample_shading(),
        };

        // Build color attachment formats based on GBuffer configuration
        let color_formats = if resources.gbuffer.is_some() {
            vec![
                color_format,                    // Index 0: Main color (HDR or swapchain)
                vk::Format::R16G16B16A16_SFLOAT, // Index 1: Normals
                vk::Format::R8G8B8A8_UNORM,      // Index 2: Albedo
                vk::Format::R16G16_SFLOAT,       // Index 3: Motion Vectors
            ]
        } else {
            vec![color_format] // ONLY Main Color
        };

        let mut builder = vulkan::Pipeline::builder(Arc::clone(&context.device.device))
            .with_layout(layout)
            .with_dynamic_rendering(&color_formats, Some(depth_format), None)
            .with_extent(extent)
            .with_pipeline_cache(cache)
            .with_depth_format(depth_format)
            .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, true)
            .with_cull_mode(vk::CullModeFlags::NONE)
            .with_front_face(vk::FrontFace::CLOCKWISE)
            .with_multisampling(multisample_config);

        if resources.gbuffer.is_some() {
            let blend_attachments = vec![
                vk::PipelineColorBlendAttachmentState {
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
                },
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
                vk::PipelineColorBlendAttachmentState {
                    color_write_mask: vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                    blend_enable: vk::FALSE,
                    ..Default::default()
                },
            ];
            builder = builder.with_color_blend_attachments(blend_attachments);
        }

        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/forward.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?;
        builder = builder.add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/forward.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

        let mut new_pipeline = builder.build()?;
        let pipeline_layout_id = self.pipeline_layout_id.ok_or_else(|| {
            AshError::VulkanError("Pipeline layout ID missing during recreation".into())
        })?;

        let pipeline_id = context
            .resources
            .register_pipeline(new_pipeline.pipeline, &[pipeline_layout_id])
            .map_err(|e| AshError::VulkanError(format!("Failed to register pipeline: {e}")))?;

        new_pipeline.mark_managed_by_registry();
        self.pipeline.main_graphics_pipeline = Some(new_pipeline);
        self.pipeline_id = Some(pipeline_id);

        // --- 3. Skybox Pass (Check status/log) ---
        if self.skybox_pass.is_some() {
            log::info!("Skybox pass adapts via dynamic state; no explicit recreation needed.");
        }

        // --- 4. Post Process Resize ---
        self.pipeline
            .post_process_mut()
            .resize(image_count, extent)?;

        // --- 5. TAA Pass Re-initialization ---
        // Re-create ping-pong history buffers at the new render resolution.
        self.pipeline
            .post_process_mut()
            .init(&context.alloc.vma, width, height)?;

        log::info!("Systems recreation complete");
        Ok(())
    }

    pub fn destroy(&mut self, context: &Context) {
        // 1. Cleanup Post Process Systems (including TAA)
        self.pipeline
            .post_process_mut()
            .destroy_resources(&context.alloc.vma);

        log::info!("Systems shutdown complete.");
    }
}
