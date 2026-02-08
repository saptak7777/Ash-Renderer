use crate::renderer::*;
use crate::AshError;
use crate::Result;

pub fn recreate_swapchain_resources(renderer: &mut Renderer) -> Result<()> {
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

    let (swapchain_extent, swapchain_format, image_views, image_count) = {
        let swapchain = renderer.swapchain.as_ref().ok_or_else(|| {
            AshError::VulkanError("Swapchain unavailable after recreation".into())
        })?;
        (
            swapchain.extent,
            swapchain.format,
            swapchain.image_views.clone(),
            swapchain.images.len(),
        )
    };

    // Cleanup resources
    cleanup_pipeline(renderer);
    cleanup_framebuffers(renderer);
    cleanup_render_pass(renderer);

    renderer.update_image_views(&image_views)?;

    renderer.recreate_depth_buffer(swapchain_extent)?;
    renderer.recreate_gbuffer(swapchain_extent)?;

    if renderer.hdr_system.is_some() {
        renderer.initialize_hdr(swapchain_extent.width, swapchain_extent.height)?;
    }

    renderer.recreate_vsr_pass(swapchain_extent)?;
    renderer.create_render_pass_and_framebuffers(
        swapchain_extent,
        swapchain_format,
        &image_views,
    )?;

    renderer.recreate_frame_syncs(image_count)?;
    renderer.recreate_command_buffers()?;
    renderer.recreate_uniform_buffers(image_count)?;

    if let Some(ref mut forward_plus) = renderer.forward_plus {
        forward_plus.on_resize(swapchain_extent.width, swapchain_extent.height);
        let fp_info = forward_plus.get_lights().get_forward_plus_info();
        renderer.scene_lighting.num_tiles_x = fp_info.num_tiles[0];
        renderer.scene_lighting.num_tiles_y = fp_info.num_tiles[1];
        renderer.scene_lighting.tile_size = fp_info.tile_size;
    }

    renderer.recreate_descriptor_sets()?;
    renderer.recreate_pipeline()?;
    renderer.recreate_skybox_pipeline()?;

    log::info!("Swapchain recreation complete ({image_count} images)");
    Ok(())
}

pub(crate) fn cleanup_framebuffers(renderer: &mut Renderer) {
    for (framebuffer, id) in renderer
        .framebuffers
        .drain(..)
        .zip(renderer.framebuffer_ids.drain(..))
    {
        drop(framebuffer);
        if let Err(e) = renderer.resources.cleanup_resource(id) {
            log::warn!("Failed to cleanup framebuffer {id}: {e}");
        }
    }
}

pub(crate) fn cleanup_render_pass(renderer: &mut Renderer) {
    if let Some(render_pass_id) = renderer.render_pass_id.take() {
        if let Err(e) = renderer.resources.cleanup_resource(render_pass_id) {
            log::warn!("Failed to cleanup render pass: {e}");
        }
    }
    renderer.render_pass = None;

    if let Some(hdr_render_pass_id) = renderer.hdr_render_pass_id.take() {
        if let Err(e) = renderer.resources.cleanup_resource(hdr_render_pass_id) {
            log::warn!("Failed to cleanup HDR render pass: {e}");
        }
    }
    renderer.hdr_render_pass = None;
}

pub(crate) fn cleanup_pipeline(renderer: &mut Renderer) {
    if let Some(pipeline_id) = renderer.pipeline_id.take() {
        if let Err(e) = renderer.resources.cleanup_resource(pipeline_id) {
            log::warn!("Failed to cleanup pipeline: {e}");
        }
    }
    renderer.pipeline = None;
}
