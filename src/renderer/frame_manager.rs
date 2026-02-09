use super::cleanup_traits::VulkanResourceCleanup;
use crate::{AshError, Result};
use ash::vk;

/// Manages frame synchronization objects and command buffer lifecycle.
pub struct FrameManager {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    command_buffers: Vec<vk::CommandBuffer>,
    current_frame: usize,
    max_frames_in_flight: usize,
}

impl FrameManager {
    pub fn new(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::with_capacity(frames_in_flight);
        let mut render_finished_semaphores = Vec::with_capacity(frames_in_flight);
        let mut in_flight_fences = Vec::with_capacity(frames_in_flight);

        unsafe {
            for _ in 0..frames_in_flight {
                image_available_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to create semaphore: {e}"))
                        })?,
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to create semaphore: {e}"))
                        })?,
                );
                in_flight_fences.push(
                    device.create_fence(&fence_info, None).map_err(|e| {
                        AshError::VulkanError(format!("Failed to create fence: {e}"))
                    })?,
                );
            }

            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(frames_in_flight as u32);

            let command_buffers = device
                .allocate_command_buffers(&allocate_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to allocate command buffers: {e}"))
                })?;

            Ok(Self {
                image_available_semaphores,
                render_finished_semaphores,
                in_flight_fences,
                command_buffers,
                current_frame: 0,
                max_frames_in_flight: frames_in_flight,
            })
        }
    }

    pub fn next_frame(&mut self, device: &ash::Device) -> Result<()> {
        unsafe {
            device
                .wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)
                .map_err(|e| AshError::VulkanError(format!("Failed to wait for fence: {e}")))?;

            device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
                .map_err(|e| AshError::VulkanError(format!("Failed to reset fence: {e}")))?;
        }
        Ok(())
    }

    pub fn acquire_next_image(
        &self,
        swapchain_loader: &ash::khr::swapchain::Device,
        swapchain_khr: vk::SwapchainKHR,
    ) -> Result<(u32, bool)> {
        unsafe {
            let (image_index, is_suboptimal) = swapchain_loader
                .acquire_next_image(
                    swapchain_khr,
                    u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                )
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        AshError::SwapchainOutOfDate(e.to_string())
                    }
                    _ => AshError::VulkanError(format!("Failed to acquire next image: {e}")),
                })?;

            Ok((image_index, is_suboptimal))
        }
    }

    pub fn get_current_command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffers[self.current_frame]
    }

    pub fn get_current_frame_index(&self) -> usize {
        self.current_frame
    }

    pub fn reset_frame(&mut self) {
        self.current_frame = 0;
    }

    pub fn get_max_frames_in_flight(&self) -> usize {
        self.max_frames_in_flight
    }

    pub fn begin_command_buffer(&self, device: &ash::Device) -> Result<vk::CommandBuffer> {
        let cmd = self.get_current_command_buffer();
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to reset command buffer: {e}"))
                })?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(cmd, &begin_info).map_err(|e| {
                AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
            })?;
        }
        Ok(cmd)
    }

    pub fn submit_and_present(
        &mut self,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        swapchain_loader: &ash::khr::swapchain::Device,
        swapchain_khr: vk::SwapchainKHR,
        image_index: u32,
    ) -> Result<bool> {
        let cmd = self.get_current_command_buffer();
        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let command_buffers = [cmd];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            device
                .queue_submit(
                    graphics_queue,
                    &[submit_info],
                    self.in_flight_fences[self.current_frame],
                )
                .map_err(|e| AshError::VulkanError(format!("Failed to submit queue: {e}")))?;

            let swapchains = [swapchain_khr];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = swapchain_loader.queue_present(present_queue, &present_info);

            self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;

            match result {
                Ok(suboptimal) => Ok(suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
                Err(e) => Err(AshError::VulkanError(format!(
                    "Failed to present queue: {e}"
                ))),
            }
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            for &s in &self.image_available_semaphores {
                device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished_semaphores {
                device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight_fences {
                device.destroy_fence(f, None);
            }
        }
    }
}

impl VulkanResourceCleanup for FrameManager {
    fn cleanup_with_device(&mut self, device: &ash::Device) -> Result<(), String> {
        self.destroy(device);
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "FrameManager"
    }
}
