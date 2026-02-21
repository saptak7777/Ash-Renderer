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

use crate::renderer::resources::Resources;
use crate::renderer::util::frame_state::FrameState;
use crate::{AshError, Result};

/// Drop guard to ensure instancing_manager is restored to resources even on panic.
///
/// SAFETY: This guard takes EXCLUSIVE ownership of the `InstancingManager` from the `Resources`
/// struct to allow safe parallel/exclusive access during frame synchronization. The `Drop`
/// implementation is guaranteed to return the manager, maintaining structural integrity.
///
/// SAFETY INVARIANT: This guard uses std::mem::take to temporarily extract the InstancingManager
/// from the main Resources struct to bypass borrow checker conflicts during updates.
/// The strict requirement is that this guard MUST be dropped (or explicitly consumed via update_and_restore)
/// to return the manager to the Resources struct. Failure to do so will leave the engine with an empty, invalid instancing state.
///
/// PANIC SAFETY: If a thread panics while this guard is held, the `Resources` struct will be left
/// with a default/empty manager until the stack unwinds and `drop()` is called. Because the
/// restoration occurs in the 'Drop' implementation, the InstancingManager is guaranteed to be
/// restored to the Resources struct even during thread unwinding (panics). This strictly
/// prevents memory leaks or dangling pointers during catastrophic engine failures.
pub struct InstancingRestoreGuard<'a> {
    resources: &'a mut Resources,
    manager: Option<crate::renderer::instancing::InstancingManager>,
}

impl<'a> InstancingRestoreGuard<'a> {
    fn new(resources: &'a mut Resources) -> Self {
        #[cfg(debug_assertions)]
        {
            resources.instancing_guard_active = true;
        }
        let manager = std::mem::take(&mut resources._instancing_manager);
        Self {
            resources,
            manager: Some(manager),
        }
    }

    // manager() was unused, so it was removed to satisfy Clippy.

    fn update_instance_buffers(&mut self, frame_index: usize) -> Result<()> {
        self.resources
            .update_instance_buffers(self.manager.as_ref().unwrap(), frame_index)
    }

    pub fn update_and_restore(mut self, frame_index: usize) -> Result<()> {
        // Self is consumed here, Drop handles the move back to resources.
        self.update_instance_buffers(frame_index)
    }
}

impl<'a> Drop for InstancingRestoreGuard<'a> {
    fn drop(&mut self) {
        if let Some(manager) = self.manager.take() {
            self.resources._instancing_manager = manager;
            #[cfg(debug_assertions)]
            {
                self.resources.instancing_guard_active = false;
            }
        } else {
            log::error!(
                "InstancingRestoreGuard dropped without restoration - this indicates a logic bug or a botched move!"
            );
        }
    }
}

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
        let sync_list: Vec<(u32, crate::renderer::resources::Material)> = {
            scene
                .material_manager
                .iter_unsynced(&scene.uploaded_material_indices)
                .map(|(id, mat)| (id, mat.clone()))
                .collect()
        };

        for (handle_index, material) in sync_list {
            if !scene.uploaded_material_indices.contains(&handle_index) {
                if let Err(e) = scene.register_material(&material) {
                    log::error!("Failed to sync material {handle_index} to GPU: {e}");
                }
            }
        }

        // ── Descriptor Pool Recycling ──────────────────────────────────────
        if let Some(dm) = resources.descriptors.as_mut() {
            dm.next_frame();
        }

        // ── Shader Hot-Reload Detection ───────────────────────────────────
        const SHADER_CHECK_INTERVAL: usize = 60;
        let current_frame = frame.frame_manager.get_current_frame_index();

        let shaders_changed = if current_frame % SHADER_CHECK_INTERVAL == 0 {
            if let Some(pipeline) = &mut systems.pipeline.main_graphics_pipeline {
                pipeline.detect_shader_changes().unwrap_or_else(|e| {
                    log::warn!("Failed to check shader changes: {e}");
                    false
                })
            } else {
                false
            }
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
            frame,
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

            matrices.model = model;
            matrices.normal_matrix = glam::Mat4::from_mat3(transform.normal_matrix());
            matrices.view = frame_state.view;
            matrices.projection = frame_state.jittered_projection;
            matrices.view_proj = frame_state.view_proj();
            matrices.prev_view_proj = frame_state.prev_view_proj;
            matrices.camera_pos = frame_state.camera_pos.extend(1.0);

            // ── Light cluster metadata ─────────────────────────────────
            scene.scene_lighting.point_light_count = scene.point_lights.len() as u32;
            if let Some(fp) = &systems.pipeline.forward_plus {
                let info = fp
                    .read()
                    .map_err(|_| AshError::LockPoisoned("ForwardPlus".to_string()))?
                    .get_lights()
                    .get_forward_plus_info();
                scene.scene_lighting.num_tiles_x = info.num_tiles[0];
                scene.scene_lighting.num_tiles_y = info.num_tiles[1];
                scene.scene_lighting.tile_size = info.tile_size;
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
        if let Some(ref fp_integration) = systems.pipeline.forward_plus {
            let mut fp = fp_integration
                .write()
                .map_err(|_| AshError::LockPoisoned("ForwardPlusIntegration".to_string()))?;
            fp.update_lights(
                &scene.point_lights,
                &scene.directional_lights,
                &scene.spot_lights,
            );
            unsafe {
                fp.upload_to_gpu(&context.alloc, &context.device.device, frame_index)?;
                fp.update_camera(
                    &context.alloc,
                    frame_index,
                    &frame_state.view.to_cols_array_2d(),
                    &frame_state.projection.to_cols_array_2d(),
                    &frame_state.camera_pos.extend(1.0).to_array(),
                )?;
            }
        }

        // ── Instance buffers ───────────────────────────────────────────
        {
            InstancingRestoreGuard::new(resources).update_and_restore(frame_index)?;
        }

        // Archive this frame's view_proj so the next frame has a valid prev_view_proj.
        frame.prev_view_proj = frame_state.view_proj();
        resources.current_view_proj = frame_state.view_proj();

        Ok(())
    }
}

impl Default for SceneSynchronizer {
    fn default() -> Self {
        Self::new()
    }
}
