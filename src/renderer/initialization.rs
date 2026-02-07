use crate::renderer::resources::{self, DepthBuffer};
use crate::renderer::types::*;
use crate::vulkan;
use crate::Result;
use ash::vk;
use std::sync::Arc;
use std::thread;

pub unsafe fn create_swapchain_data(
    device: &vulkan::VulkanDevice,
    alloc: &Arc<vulkan::Allocator>,
    extent: vk::Extent2D,
) -> Result<(
    vulkan::SwapchainWrapper,
    vk::RenderPass,
    Vec<vulkan::Framebuffer>,
    DepthBuffer,
)> {
    let swapchain = vulkan::SwapchainWrapper::new(device, device.headless, extent)?;

    let depth_buffer = resources::DepthBuffer::new(
        Arc::clone(&device.device),
        Arc::clone(alloc),
        swapchain.extent.width,
        swapchain.extent.height,
    )?;

    let mut render_pass = vulkan::RenderPass::builder(Arc::clone(&device.device))
        .with_swapchain_color(
            swapchain.format,
            if device.headless {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::PRESENT_SRC_KHR
            },
        )
        .with_depth_attachment(
            depth_buffer.format(),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        .build()?;
    render_pass.mark_managed_by_registry();

    let mut framebuffers = Vec::new();
    for &image_view in &swapchain.image_views {
        let attachments = [image_view, depth_buffer.view()];
        let framebuffer = vulkan::Framebuffer::new(
            Arc::clone(&device.device),
            render_pass.handle(),
            &attachments,
            swapchain.extent,
        )?;
        framebuffers.push(framebuffer);
    }

    Ok((swapchain, render_pass.handle(), framebuffers, depth_buffer))
}

pub unsafe fn create_frame_resources(
    device: &vulkan::VulkanDevice,
    swapchain_image_count: usize,
) -> Result<(
    Vec<vk::CommandBuffer>,
    Vec<vulkan::FrameSync>,
    vulkan::CommandBufferManager,
)> {
    let worker_count = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let command_manager = vulkan::CommandBufferManager::new(
        Arc::clone(&device.device),
        device.graphics_queue_family,
        worker_count,
    )?;

    let command_buffers = command_manager.allocate_primary_buffers(swapchain_image_count as u32)?;

    let mut frame_syncs = Vec::with_capacity(swapchain_image_count);
    for _ in 0..swapchain_image_count {
        let sync = vulkan::FrameSync::new(Arc::clone(&device.device))?;
        frame_syncs.push(sync);
    }

    Ok((command_buffers, frame_syncs, command_manager))
}

pub unsafe fn create_main_pipeline(
    device: &vulkan::VulkanDevice,
    swapchain_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    set_layouts: &[vk::DescriptorSetLayout],
    pipeline_cfg: &PipelineConfig,
    depth_format: vk::Format,
    pipeline_cache: vk::PipelineCache,
) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    let push_constant_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: crate::renderer::model_renderer::DRAW_PUSH_VERTEX_BYTES
            + crate::renderer::model_renderer::DRAW_PUSH_FRAGMENT_BYTES,
    }];

    let mut pipeline_layout_builder = vulkan::PipelineLayout::builder(Arc::clone(&device.device));
    for layout in set_layouts {
        pipeline_layout_builder = pipeline_layout_builder.add_set_layout(*layout);
    }
    for range in &push_constant_ranges {
        pipeline_layout_builder = pipeline_layout_builder.add_push_constant(*range);
    }
    let mut pipeline_layout = pipeline_layout_builder.build()?;
    pipeline_layout.mark_managed_by_registry();

    let mut pipeline_builder = vulkan::Pipeline::builder(Arc::clone(&device.device))
        .with_layout(pipeline_layout.handle())
        .with_render_pass(render_pass)
        .with_extent(swapchain_extent)
        .with_pipeline_cache(pipeline_cache)
        .with_depth_format(depth_format)
        .with_depth_test(vk::CompareOp::GREATER_OR_EQUAL, true)
        .with_cull_mode(vk::CullModeFlags::NONE)
        .with_front_face(vk::FrontFace::CLOCKWISE)
        .with_multisampling(pipeline_cfg.multisample_config())
        .add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/vert.vert.spv")),
            vk::ShaderStageFlags::VERTEX,
            "main",
        )?
        .add_shader_from_bytes(
            include_bytes!(concat!(env!("OUT_DIR"), "/frag.frag.spv")),
            vk::ShaderStageFlags::FRAGMENT,
            "main",
        )?;

    for specialization in &pipeline_cfg.specialization_constants {
        pipeline_builder = pipeline_builder.with_specialization_bytes(
            specialization.stage,
            specialization.constant_id,
            specialization.bytes(),
        );
    }

    let mut pipeline = pipeline_builder.build()?;
    pipeline.mark_managed_by_registry();

    Ok((pipeline.pipeline, pipeline_layout.handle()))
}
