use crate::renderer::{
    context::Context, frame_manager::FrameManager, resource_registry::ResourceId,
    types::GBufferIndices,
};
use crate::vulkan::{CommandBufferManager, SwapchainWrapper};
use crate::{AshError, Result};
use ash::vk;
use std::sync::Arc;
use std::time::Instant;

/// The Frame manages swapchain lifecycle, synchronization primitives,
/// and per-frame command submission.
pub struct Frame {
    pub swapchain: Option<SwapchainWrapper>,
    pub frame_manager: FrameManager,
    pub cmds: Arc<CommandBufferManager>,
    pub swapchain_image_view_ids: Vec<ResourceId>,

    // Rendering State
    pub start_time: Instant,
    pub gbuffer_indices: Option<GBufferIndices>,
    pub hdr_image_index: Option<u32>,
    pub last_image_index: u32,
}

impl Frame {
    /// Initializes frame-related resources and state.
    pub fn new(
        context: &Context,
        width: u32,
        height: u32,
        present_mode: vk::PresentModeKHR,
    ) -> Result<Self> {
        let extent = vk::Extent2D { width, height };

        unsafe {
            // Relocated logic from Renderer::new / Resources::new
            let mut swapchain = SwapchainWrapper::new(
                &context.device,
                context.device.headless,
                extent,
                present_mode,
            )?;

            let mut swapchain_image_view_ids = Vec::with_capacity(swapchain.image_views.len());
            for &view in &swapchain.image_views {
                let id = context.resources.register_image_view(view)?;
                swapchain_image_view_ids.push(id);
            }
            swapchain.mark_image_views_managed_by_registry();

            let worker_count = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);

            let cmds = Arc::new(CommandBufferManager::new(
                Arc::clone(&context.device.device),
                context.device.graphics_queue_family,
                worker_count,
            )?);

            let frame_manager = FrameManager::new(
                &context.device.device,
                cmds.upload_command_pool_handle(),
                swapchain.image_views.len(),
            )?;

            Ok(Self {
                swapchain: Some(swapchain),
                frame_manager,
                cmds,
                swapchain_image_view_ids,
                start_time: Instant::now(),
                gbuffer_indices: None,
                hdr_image_index: None,
                last_image_index: 0,
            })
        }
    }

    /// Acquires the next image, waits for fences, and resets the command pool.
    pub fn begin_frame(&mut self, context: &Context) -> Result<(u32, bool)> {
        // 1. Wait for GPU and reset current frame's fence
        self.frame_manager.next_frame(&context.device.device)?;

        // 2. Acquire next swapchain image
        let swapchain_loader = ash::khr::swapchain::Device::new(
            context.device.instance.instance(),
            &context.device.device,
        );
        let swapchain_khr = self
            .swapchain
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Swapchain missing".to_string()))?
            .swapchain;

        let (image_index, is_suboptimal) = self
            .frame_manager
            .acquire_next_image(&swapchain_loader, swapchain_khr)?;

        // 3. Reset the primary command pool for this frame
        self.cmds
            .reset_primary_pool(vk::CommandPoolResetFlags::empty())?;

        Ok((image_index, is_suboptimal))
    }

    /// Submits the command buffer and presents the image.
    pub fn submit_and_present(&mut self, context: &Context, image_index: u32) -> Result<bool> {
        let swapchain_loader = ash::khr::swapchain::Device::new(
            context.device.instance.instance(),
            &context.device.device,
        );
        let swapchain_khr = self
            .swapchain
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Swapchain missing".to_string()))?
            .swapchain;

        self.frame_manager.submit_and_present(
            &context.device.device,
            context.queue.graphics_queue,
            context.queue.present_queue,
            &swapchain_loader,
            swapchain_khr,
            image_index,
        )
    }
}
