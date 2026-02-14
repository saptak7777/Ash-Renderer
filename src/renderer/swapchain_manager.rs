use crate::renderer::*;
use crate::AshError;
use crate::Result;

pub fn recreate_swapchain_resources(renderer: &mut Renderer, scene: &mut Scene) -> Result<()> {
    log::info!("Starting swapchain recreation...");

    // Delegate creation to the Queue
    {
        let queue = &mut renderer.queue;
        let swapchain = renderer
            .swapchain
            .as_mut()
            .ok_or(AshError::VulkanError("Swapchain not available".into()))?;
        queue.recreate_swapchain(swapchain, &renderer.device)?;
    }

    let (swapchain_extent, image_views, image_count) = {
        let swapchain = renderer.swapchain.as_ref().ok_or_else(|| {
            AshError::VulkanError("Swapchain unavailable after recreation".into())
        })?;
        (
            swapchain.extent,
            swapchain.image_views.clone(),
            swapchain.images.len(),
        )
    };

    // Cleanup resources
    cleanup_pipeline(renderer);

    renderer.update_image_views(&image_views)?;

    renderer.recreate_depth_buffer(swapchain_extent)?;

    // Update Forward+ depth descriptor if the system is active
    if let (Some(ref db), Some(ref fp_lock)) =
        (&renderer.depth_buffer, &renderer.pipeline.forward_plus)
    {
        let mut fp = fp_lock.write().unwrap();
        unsafe {
            fp.update_depth_descriptor(&renderer.device.device, db.view(), db.sampler());
        }
        log::info!("Forward+ depth descriptors updated after resize.");
    }
    renderer.recreate_gbuffer(swapchain_extent)?;

    if renderer.hdr_system.is_some() {
        renderer.initialize_hdr(swapchain_extent.width, swapchain_extent.height)?;
    }

    renderer.recreate_vsr_pass(swapchain_extent)?;

    renderer.recreate_frame_syncs(image_count)?;
    renderer.recreate_command_buffers()?;
    renderer.recreate_uniform_buffers(image_count)?;

    if let Some(ref forward_plus_arc) = renderer.pipeline.forward_plus {
        let mut forward_plus = forward_plus_arc.write().unwrap();
        forward_plus.on_resize(swapchain_extent.width, swapchain_extent.height);
        let fp_info = forward_plus.get_lights().get_forward_plus_info();
        scene.scene_lighting.num_tiles_x = fp_info.num_tiles[0];
        scene.scene_lighting.num_tiles_y = fp_info.num_tiles[1];
        scene.scene_lighting.tile_size = fp_info.tile_size;
    }

    renderer.recreate_descriptor_sets()?;
    renderer.recreate_pipeline()?;
    renderer.recreate_skybox_pipeline()?;
    renderer
        .pipeline
        .post_process_mut()
        .resize(image_count, swapchain_extent)?;

    log::info!("Swapchain recreation complete ({image_count} images)");
    Ok(())
}

pub(crate) fn cleanup_pipeline(renderer: &mut Renderer) {
    if let Some(pipeline_id) = renderer.pipeline_id.take() {
        if let Err(e) = renderer.resources.cleanup_resource(pipeline_id) {
            log::warn!("Failed to cleanup pipeline: {e}");
        }
    }
    renderer.pipeline.main_graphics_pipeline = None;
}
