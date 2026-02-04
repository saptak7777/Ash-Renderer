use ash::vk;

/// Returns true if the provided format contains a stencil component.
pub fn has_stencil_component(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
    )
}

/// Helper to execute a single-use command buffer on a queue.
pub fn execute_single_use<F>(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    recorder: F,
) -> crate::Result<()>
where
    F: FnOnce(vk::CommandBuffer),
{
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    unsafe {
        let command_buffers = device.allocate_command_buffers(&alloc_info).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to allocate command buffer: {e}"))
        })?;
        let command_buffer = command_buffers[0];

        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
            })?;

        recorder(command_buffer);

        device.end_command_buffer(command_buffer).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to end command buffer: {e}"))
        })?;

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .map_err(|e| crate::AshError::VulkanError(format!("Failed to submit queue: {e}")))?;

        device.queue_wait_idle(queue).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to wait for queue idle: {e}"))
        })?;

        device.free_command_buffers(command_pool, &command_buffers);
    }

    Ok(())
}

/// Find a suitable memory type
pub fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..properties.memory_type_count {
        let type_bits = 1 << i;
        let has_properties = properties.memory_types[i as usize]
            .property_flags
            .contains(required);

        if (type_filter & type_bits) != 0 && has_properties {
            return Some(i);
        }
    }
    None
}

/// Helper to begin a single-time command buffer.
pub unsafe fn begin_single_time_commands(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> crate::Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info).map_err(|e| {
        crate::AshError::VulkanError(format!("Failed to allocate command buffer: {e}"))
    })?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device
        .begin_command_buffer(command_buffer, &begin_info)
        .map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
        })?;

    Ok(command_buffer)
}

/// Helper to end and submit a single-time command buffer.
pub unsafe fn end_single_time_commands(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> crate::Result<()> {
    device
        .end_command_buffer(command_buffer)
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
    let submit_infos = [submit_info];

    device
        .queue_submit(queue, &submit_infos, vk::Fence::null())
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to submit queue: {e}")))?;
    device
        .queue_wait_idle(queue)
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to wait for queue idle: {e}")))?;

    device.free_command_buffers(command_pool, &command_buffers);

    Ok(())
}
/// Execute a command buffer and wait for a fence instead of the entire queue.
///
/// This is more efficient than `execute_single_time_commands()` because it
/// waits on a specific fence rather than the entire queue, allowing the CPU
/// to resume work after only this specific submission is finished.
///
/// # Performance
///
/// This function is optimized for one-off operations (uploads, transitions). For high-frequency
/// per-frame operations, consider using a persistent command buffer or a
/// dedicated transfer queue to avoid the overhead of fence creation and destruction.
///
/// # Safety
///
/// Caller must ensure:
/// - `command_pool` is a valid, existing Vulkan command pool.
/// - `queue` is a valid Vulkan queue compatible with the command buffer.
/// - The closure `f` does not outlive the command buffer execution.
/// - Sufficient synchronization is handled externally if the closure accesses shared resources.
/// - The GPU timeout is 60 seconds; extremely long operations may fail with a timeout error.
#[inline]
pub unsafe fn execute_single_use_fenced<F>(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    f: F,
) -> crate::Result<()>
where
    F: FnOnce(vk::CommandBuffer),
{
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    f(command_buffer);

    device
        .end_command_buffer(command_buffer)
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

    let fence_info = vk::FenceCreateInfo::default();
    let fence = device
        .create_fence(&fence_info, None)
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to create fence: {e}")))?;

    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
    let submit_infos = [submit_info];

    device
        .queue_submit(queue, &submit_infos, fence)
        .map_err(|e| crate::AshError::VulkanError(format!("Failed to submit queue: {e}")))?;

    device
        .wait_for_fences(&[fence], true, 60_000_000_000)
        .map_err(|e| {
            if e == vk::Result::TIMEOUT {
                crate::AshError::VulkanError(
                    "GPU timeout (60s) in execute_single_use_fenced. The GPU may have hung."
                        .to_string(),
                )
            } else {
                crate::AshError::VulkanError(format!("Failed to wait for fence: {e}"))
            }
        })?;

    device.destroy_fence(fence, None);
    device.free_command_buffers(command_pool, &command_buffers);

    Ok(())
}
