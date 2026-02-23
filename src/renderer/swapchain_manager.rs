use crate::AshError;
use crate::Result;
use crate::renderer::*;

pub fn recreate_swapchain_resources(renderer: &mut Renderer, scene: &mut Scene) -> Result<()> {
    log::info!("Starting swapchain recreation...");

    // Delegate creation to the Queue
    {
        let queue = &mut renderer.context.queue;
        let swapchain = renderer
            .frame
            .swapchain
            .as_mut()
            .ok_or(AshError::VulkanError("Swapchain not available".into()))?;
        queue.recreate_swapchain(swapchain, &renderer.context.device)?;
    }

    let (swapchain_extent, image_views, image_count) = {
        let swapchain = renderer.frame.swapchain.as_ref().ok_or_else(|| {
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

    {
        let gbuffer_indices = renderer.frame.gbuffer_indices.as_mut().unwrap();
        renderer.resources.resize(
            &renderer.context,
            gbuffer_indices,
            swapchain_extent,
            image_count,
        )?;
    }

    // Update Forward+ depth descriptor if the system is active
    if let (Some(db), Some(fp_lock)) = (
        &renderer.resources.depth_buffer,
        &renderer.systems.pipeline.forward_plus,
    ) {
        let mut fp = fp_lock
            .write()
            .map_err(|e| AshError::VulkanError(format!("Forward+ lock poisoned: {e}")))?;
        unsafe {
            fp.update_depth_descriptor(&renderer.context.device.device, db.view(), db.sampler());
        }
        log::info!("Forward+ depth descriptors updated after resize.");
    }

    if renderer.systems.hdr_system.is_some() {
        renderer.initialize_hdr(swapchain_extent.width, swapchain_extent.height)?;
    }

    renderer.recreate_frame_syncs(image_count)?;
    renderer.recreate_command_buffers()?;

    if let Some(ref forward_plus_arc) = renderer.systems.pipeline.forward_plus {
        let mut forward_plus = forward_plus_arc
            .write()
            .map_err(|e| AshError::VulkanError(format!("Forward+ lock poisoned: {e}")))?;
        forward_plus.on_resize(swapchain_extent.width, swapchain_extent.height);
        let fp_info = forward_plus.get_lights().get_forward_plus_info();
        scene.scene_lighting.num_tiles_x = fp_info.num_tiles[0];
        scene.scene_lighting.num_tiles_y = fp_info.num_tiles[1];
        scene.scene_lighting.tile_size = fp_info.tile_size;
    }

    renderer.recreate_descriptor_sets()?;

    // Push systems logic down
    renderer.systems.resize(
        &renderer.context,
        &mut renderer.resources,
        swapchain_extent.width,
        swapchain_extent.height,
        renderer.frame.swapchain.as_ref().unwrap().format,
        image_count,
    )?;

    log::info!("Swapchain recreation complete ({image_count} images)");
    Ok(())
}

pub(crate) fn cleanup_pipeline(renderer: &mut Renderer) {
    if let Some(pipeline_id) = renderer.systems.pipeline_id.take() {
        if let Err(e) = renderer.context.resources.cleanup_resource(pipeline_id) {
            log::warn!("Failed to cleanup pipeline: {e}");
        }
    }
    renderer.systems.pipeline.main_graphics_pipeline = None;
}
