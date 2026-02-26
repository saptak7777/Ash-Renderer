//! CPU-to-GPU Scene Reconciliation.
//!
//! `SceneSynchronizer` is the dedicated home for all logic that bridges the
//! CPU-side `Scene` with the GPU's buffer state each frame.
//!
//! Previously this responsibility was split across:
//! - `Renderer::sync_frame_resources` – material/shader hot-reload
//! - `Resources::update_global_data`  – uniform updates, VSM prep, jitter
//!
//! By centralising both here, `Renderer` becomes a pure orchestrator that
//! never directly reads or writes GPU memory.

use crate::renderer::util::frame_state::FrameState;
use crate::{AshError, Result};

/// Encapsulates all CPU-to-GPU state reconciliation for a single frame.
///
/// The synchronizer is stateless by design: it borrows everything it needs
/// per call, making it trivially testable in isolation.
pub struct SceneSynchronizer;

pub struct FramePreparationInfo<'a> {
    pub context: &'a crate::renderer::context::Context,
    pub frame: &'a mut crate::renderer::frame::Frame,
    pub scene: &'a mut crate::renderer::Scene,
    pub systems: &'a mut crate::renderer::systems::Systems,
    pub resources: &'a mut crate::renderer::resources::Resources,
    pub vsm_manager: &'a mut crate::renderer::features::vsm::VsmManager,
    pub frame_state: &'a FrameState,
    pub frame_index: usize,
    pub model_matrix: Option<glam::Mat4>,
}

impl SceneSynchronizer {
    pub fn new() -> Self {
        Self
    }

    // --- Resource Synchronization ---
    ///
    /// Uploads dirty materials, recycles descriptor pools, and checks for
    /// hot-reloaded shaders. Must run before `prepare_frame` every frame.
    ///
    /// Returns `true` if shaders have changed and a swapchain recreation is required.
    pub fn sync_resources(
        &self,
        frame: &mut crate::renderer::frame::Frame,
        scene: &mut crate::renderer::Scene,
        systems: &mut crate::renderer::systems::Systems,
        resources: &mut crate::renderer::resources::Resources,
    ) -> Result<bool> {
        // ── Material Sync ──────────────────────────────────────────────────
        let mut newly_uploaded = Vec::new();

        // Extract just the uniform representation we need.
        // This drops the immutable borrow of `scene` immediately.
        let sync_list: Vec<(u32, crate::renderer::resources::uniform::MaterialUniform)> = {
            scene
                .material_manager
                .iter_unsynced(&scene.uploaded_material_indices)
                .map(|(i, mat)| (i, mat.to_uniform()))
                .collect()
        };

        for (handle_index, material_uniform) in sync_list {
            if !scene.uploaded_material_indices.contains(&handle_index) {
                // Register uniform data to GPU. This calls into the Buffer map which requires mutable scene.
                if let Err(e) = scene.register_material_uniform(handle_index, material_uniform) {
                    log::error!("Failed to sync material {handle_index} to GPU: {e}");
                } else {
                    newly_uploaded.push(handle_index);
                }
            }
        }

        // Update the set after the loop to avoid borrow checker issues during iteration
        for index in newly_uploaded {
            scene.uploaded_material_indices.insert(index);
        }

        // ── Descriptor Pool Recycling ──────────────────────────────────────
        if let Some(dm) = resources.descriptors.as_mut() {
            dm.next_frame();
        }

        // ── Shader Hot-Reload Detection ───────────────────────────────────
        const SHADER_CHECK_INTERVAL: usize = 60;
        let current_frame = frame.frame_manager.get_current_frame_index();

        let shaders_changed = if current_frame % SHADER_CHECK_INTERVAL == 0 {
            systems
                .pipeline
                .main_graphics_pipeline
                .detect_shader_changes()
                .unwrap_or_else(|e| {
                    log::warn!("Failed to check shader changes: {e}");
                    false
                })
        } else {
            false
        };

        Ok(shaders_changed)
    }

    // --- Frame Data Preparation ---
    ///
    /// Uploads the uniform buffer for this frame's camera matrices, advances
    /// the VSM shadow manager, and updates Forward+ light clusters.
    ///
    /// Call **after** `sync_frame_resources` and **before** command recording.
    pub fn prepare_frame(&self, info: FramePreparationInfo<'_>) -> Result<()> {
        let FramePreparationInfo {
            context,
            frame: _frame,
            scene,
            systems,
            resources,
            vsm_manager,
            frame_state,
            frame_index,
            model_matrix,
        } = info;
        // ── Uniform buffer upload ──────────────────────────────────────
        {
            let uniform_buffer =
                resources
                    .uniform_buffers
                    .get_mut(frame_index)
                    .ok_or_else(|| {
                        AshError::VulkanError(format!(
                            "Uniform buffer not found for frame index {frame_index}"
                        ))
                    })?;
            let mut ub = uniform_buffer
                .write()
                .map_err(|_| AshError::LockPoisoned("UniformBuffer".to_string()))?;
            let matrices = ub.matrices_mut();

            let model = model_matrix.unwrap_or(glam::Mat4::IDENTITY);
            let mut transform = crate::renderer::resources::transform::Transform::identity();
            transform.set_model(model);

            let extent = resources.gbuffer.extent();
            let width = extent.width as f32;
            let height = extent.height as f32;

            matrices.screen_params = glam::Vec4::new(width, height, 1.0 / width, 1.0 / height);
            matrices.hiz_levels = systems
                .pipeline
                .hiz_pass
                .read()
                .map(|h| h.mip_count())
                .unwrap_or(0);

            matrices.model = model;
            matrices.normal_matrix = glam::Mat4::from_mat3(transform.normal_matrix());
            matrices.view = frame_state.view;
            matrices.projection = frame_state.jittered_projection;
            matrices.view_proj = frame_state.view_proj();
            matrices.prev_view_proj = frame_state.prev_view_proj;
            matrices.view_proj_no_jitter = frame_state.view_proj_no_jitter;
            matrices.prev_view_proj_no_jitter = frame_state.prev_view_proj_no_jitter;
            matrices.camera_pos = frame_state.camera_pos.extend(1.0);

            // ── Light cluster metadata ─────────────────────────────────
            scene.scene_lighting.point_light_count = scene.point_lights.len() as u32;
            {
                let fp = &systems.pipeline.forward_plus;
                let (num_tiles, tile_size) = fp
                    .read()
                    .map_err(|_| AshError::LockPoisoned("ForwardPlus".to_string()))?
                    .get_lights()
                    .get_tile_info();
                scene.scene_lighting.num_tiles_x = num_tiles[0];
                scene.scene_lighting.num_tiles_y = num_tiles[1];
                scene.scene_lighting.tile_size = tile_size;
            }
            matrices.set_lighting(&scene.scene_lighting);
            matrices.set_light_space_matrix(glam::Mat4::IDENTITY);

            unsafe {
                ub.update()?;
            }
        }

        // ── Feature system tick ────────────────────────────────────────
        {
            let mut dummy_transform = crate::renderer::resources::transform::Transform::identity();
            let mut feature_ctx = crate::renderer::features::FeatureFrameContext {
                device: context.device.device.as_ref(),
                descriptor_allocator: resources.descriptors.as_ref(),
                transform: &mut dummy_transform,
                auto_rotate: false,
                elapsed_seconds: frame_state.elapsed_time,
            };
            systems.features.before_frame(&mut feature_ctx);
        }

        // ── VSM shadow manager ─────────────────────────────────────────
        let light_dir = scene
            .directional_lights
            .first()
            .map(|l| l.direction)
            .unwrap_or(glam::Vec3::new(0.0, -1.0, 0.0));

        vsm_manager.prepare(
            scene,
            frame_state.camera_pos,
            frame_state.view_proj(),
            light_dir,
            frame_index as u32,
        )?;

        // ── Forward+ GPU upload ────────────────────────────────────────
        {
            let mut fp = systems
                .pipeline
                .forward_plus
                .write()
                .map_err(|_| AshError::LockPoisoned("ForwardPlusIntegration".to_string()))?;
            fp.update_lights(
                &scene.point_lights,
                &scene.directional_lights,
                &scene.spot_lights,
            );
            unsafe {
                fp.upload_to_gpu(&context.alloc, &context.device.device, frame_index)?;
            }
        }

        Ok(())
    }
}

impl Default for SceneSynchronizer {
    fn default() -> Self {
        Self::new()
    }
}
