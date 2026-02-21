//! Global Cluster Buffer for VCGS
//!
//! Stores static cluster data (CullObjectData) for all meshes in a single GPU-resident buffer.
//! Used by compute shaders (via BDA) to access the cluster hierarchy.

use ash::vk;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
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
    /// # Safety
    /// The caller must ensure that the device and allocator are valid.
    ///
    /// # Arguments
    /// * `capacity_mb` - Buffer capacity in Megabytes
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        capacity_mb: u32,
    ) -> Result<Self> {
        let capacity_bytes = (capacity_mb as u64) * 1024 * 1024;

        log::info!("Creating GlobalClusterBuffer: {capacity_mb} MB");

        let (buffer, allocation) = unsafe {
            allocator.create_buffer_with_flags_and_name(
                capacity_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
                vk_mem::AllocationCreateFlags::empty(),
                Some("GlobalClusterBuffer".to_string()),
            )?
        };

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { device.get_buffer_device_address(&address_info) };

        log::info!("GlobalClusterBuffer BDA: {device_address:#018X}");

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

    /// Records a copy of clusters from a staging buffer into the global buffer.
    ///
    /// # Safety
    /// The caller must ensure that the command buffer is in a recording state,
    /// and that the staging buffer and its offset are valid.
    ///
    /// # Arguments
    /// * `command_buffer` - Command buffer to record the copy command into.
    /// * `staging_buffer` - Source buffer containing the clusters.
    /// * `staging_offset` - Offset in the staging buffer.
    /// * `cluster_count` - Number of clusters to copy.
    ///
    /// Returns the start index of the uploaded clusters in the global buffer.
    pub unsafe fn upload_clusters(
        &self,
        command_buffer: vk::CommandBuffer,
        staging_buffer: vk::Buffer,
        staging_offset: u64,
        cluster_count: u32,
    ) -> Result<u32> {
        if cluster_count == 0 {
            return Ok(0);
        }

        let element_size = std::mem::size_of::<CullObjectData>() as u64;
        let size = cluster_count as u64 * element_size;

        // Atomically reserve space in the buffer
        let dst_offset = self.offset_bytes.fetch_add(size, Ordering::SeqCst);

        if dst_offset + size > self.capacity_bytes {
            return Err(AshError::VulkanError(
                "GlobalClusterBuffer: Overflow".to_string(),
            ));
        }

        let region = vk::BufferCopy::default()
            .src_offset(staging_offset)
            .dst_offset(dst_offset)
            .size(size);

        // Record the copy command
        unsafe {
            self.device
                .cmd_copy_buffer(command_buffer, staging_buffer, self.buffer, &[region]);
        }

        // Return the start index (base index in the global cluster array)
        Ok((dst_offset / element_size) as u32)
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Be very careful calling this manually.
    ///
    /// # Safety
    /// The caller must ensure that the GPU is idle and no resources are currently in use.
    pub unsafe fn destroy(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        if let Ok(mut alloc) = self.allocation.lock() {
            unsafe {
                self.allocator.destroy_buffer(self.buffer, &mut alloc);
            }
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
