//! Global Cluster Buffer for VCGS
//!
//! Stores static cluster data (CullObjectData) for all meshes in a single GPU-resident buffer.
//! Used by compute shaders (via BDA) to access the cluster hierarchy.

use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use crate::renderer::vcgs::culling::CullObjectData;
use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// Global Cluster Buffer
///
/// - Storage Buffer | Shader Device Address | Transfer Dst
pub struct GlobalClusterBuffer {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,

    buffer: vk::Buffer,
    allocation: Mutex<vk_mem::Allocation>,
    device_address: vk::DeviceAddress,
    capacity_bytes: u64,
    offset_bytes: AtomicU64,

    destroyed: bool,
}

impl GlobalClusterBuffer {
    /// Create a new global cluster buffer
    ///
    /// # Arguments
    /// * `capacity_mb` - Buffer capacity in Megabytes
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        capacity_mb: u32,
    ) -> Result<Self> {
        let capacity_bytes = (capacity_mb as u64) * 1024 * 1024;

        log::info!("Creating GlobalClusterBuffer: {} MB", capacity_mb);

        let (buffer, allocation) = allocator.create_buffer_with_flags_and_name(
            capacity_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
            Some("GlobalClusterBuffer".to_string()),
        )?;

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = device.get_buffer_device_address(&address_info);

        log::info!("GlobalClusterBuffer BDA: 0x{:016X}", device_address);

        Ok(Self {
            device,
            allocator,
            buffer,
            allocation: Mutex::new(allocation),
            device_address,
            capacity_bytes,
            offset_bytes: AtomicU64::new(0),
            destroyed: false,
        })
    }

    /// Upload clusters to the buffer
    ///
    /// Returns the start index of the uploaded clusters (index = offset / sizeof(CullObjectData))
    pub unsafe fn upload_clusters(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        clusters: &[CullObjectData],
    ) -> Result<u32> {
        if clusters.is_empty() {
            return Ok(0);
        }

        let element_size = std::mem::size_of::<CullObjectData>() as u64;
        let size = clusters.len() as u64 * element_size;

        // Alignment check (CullObjectData should be 16-byte aligned, but check anyway)
        // Global buffer is just a linear array of structs.

        let offset = self.offset_bytes.fetch_add(size, Ordering::SeqCst);

        if offset + size > self.capacity_bytes {
            return Err(AshError::VulkanError(
                "GlobalClusterBuffer: Overflow".to_string(),
            ));
        }

        self.upload_data(
            command_pool,
            queue,
            self.buffer,
            offset,
            bytemuck::cast_slice(clusters),
        )?;

        // Return index
        Ok((offset / element_size) as u32)
    }

    unsafe fn upload_data(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        dst_buffer: vk::Buffer,
        dst_offset: u64,
        data: &[u8],
    ) -> Result<()> {
        // Reuse upload logic from DualHeapGeometryBuffer or similar utility
        // For now, simpler inline implementation

        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = self
            .device
            .allocate_command_buffers(&cmd_info)
            .map_err(|e| AshError::VulkanError(format!("Alloc cmd: {}", e)))?;
        let cmd = cmd_buffers[0];

        self.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| AshError::VulkanError(format!("Begin cmd: {}", e)))?;

        if data.len() <= 65536 {
            self.device
                .cmd_update_buffer(cmd, dst_buffer, dst_offset, data);
        } else {
            for (i, chunk) in data.chunks(65536).enumerate() {
                let chunk_offset = dst_offset + (i * 65536) as u64;
                self.device
                    .cmd_update_buffer(cmd, dst_buffer, chunk_offset, chunk);
            }
        }

        self.device
            .end_command_buffer(cmd)
            .map_err(|e| AshError::VulkanError(format!("End cmd: {}", e)))?;

        let submit = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
        self.device
            .queue_submit(queue, &[submit], vk::Fence::null())
            .map_err(|e| AshError::VulkanError(format!("Submit: {}", e)))?;

        self.device
            .queue_wait_idle(queue)
            .map_err(|e| AshError::VulkanError(format!("Wait: {}", e)))?;

        self.device.free_command_buffers(command_pool, &cmd_buffers);

        Ok(())
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    pub unsafe fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        if let Ok(mut alloc) = self.allocation.lock() {
            self.allocator.destroy_buffer(self.buffer, &mut alloc);
        }
        log::info!("GlobalClusterBuffer destroyed");
    }
}

impl Drop for GlobalClusterBuffer {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
