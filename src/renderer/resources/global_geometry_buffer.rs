//! Dual Heap Geometry Buffer for BDA-based Vertex Pulling
//!
//! Separates vertex data (BDA-accessible) from index data (traditional binding)
//! to ensure Intel Arc compatibility and prepare for Hardware Ray Tracing.

use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// 16-byte alignment for buffer_reference_align in GLSL
const VERTEX_ALIGNMENT: u64 = 16;

/// Allocation result containing offsets and addresses
#[derive(Debug, Clone, Copy)]
pub struct MeshAllocation {
    /// Byte offset into the vertex heap
    pub vertex_offset: u64,
    /// Device address for this mesh's vertices (base + offset)
    pub vertex_device_address: vk::DeviceAddress,
    /// Number of vertices allocated
    pub vertex_count: u32,
    /// Byte offset into the index heap
    pub index_offset: u64,
    /// Number of indices allocated
    pub index_count: u32,
}

/// Dual Heap Geometry Buffer
///
/// - Vertex Heap: STORAGE_BUFFER | SHADER_DEVICE_ADDRESS | TRANSFER_DST
/// - Index Heap: INDEX_BUFFER | TRANSFER_DST (No BDA flag for Intel Arc stability)
pub struct DualHeapGeometryBuffer {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,

    // Vertex Heap (BDA-enabled)
    vertex_heap: vk::Buffer,
    vertex_allocation: Mutex<vk_mem::Allocation>,
    vertex_device_address: vk::DeviceAddress,
    vertex_capacity: u64,
    vertex_offset: AtomicU64,

    // Index Heap (Traditional)
    index_buffer: vk::Buffer,
    index_allocation: Mutex<vk_mem::Allocation>,
    index_capacity: u64,
    index_offset: AtomicU64,
}

impl DualHeapGeometryBuffer {
    /// Create a new dual heap geometry buffer
    ///
    /// # Arguments
    /// * `device` - Vulkan device
    /// * `allocator` - VMA allocator
    /// * `vertex_capacity_mb` - Vertex heap size in megabytes
    /// * `index_capacity_mb` - Index heap size in megabytes
    ///
    /// # Safety
    /// Caller must ensure device and allocator are valid
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        vertex_capacity_mb: u32,
        index_capacity_mb: u32,
    ) -> Result<Self> {
        let vertex_capacity = (vertex_capacity_mb as u64) * 1024 * 1024;
        let index_capacity =
            ((index_capacity_mb as u64) * 1024 * 1024) / std::mem::size_of::<u32>() as u64;

        log::info!(
            "Creating DualHeapGeometryBuffer: vertex={} MB, index={} MB",
            vertex_capacity_mb,
            index_capacity_mb
        );

        // Create Vertex Heap with BDA support
        let (vertex_heap, vertex_allocation) = allocator.create_buffer_with_flags_and_name(
            vertex_capacity,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
            Some("Vertex Heap (BDA)".to_string()),
        )?;

        // Get device address for vertex heap
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(vertex_heap);
        let vertex_device_address = device.get_buffer_device_address(&address_info);

        log::info!("Vertex Heap BDA: 0x{:016X}", vertex_device_address);

        // Create Index Heap (No BDA flag for Intel Arc compatibility)
        let (index_buffer, index_allocation) = allocator.create_buffer_with_flags_and_name(
            index_capacity * std::mem::size_of::<u32>() as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
            Some("Index Heap".to_string()),
        )?;

        log::info!("✅ DualHeapGeometryBuffer created successfully");

        Ok(Self {
            device,
            allocator,
            vertex_heap,
            vertex_allocation: Mutex::new(vertex_allocation),
            vertex_device_address,
            vertex_capacity,
            vertex_offset: AtomicU64::new(0),
            index_buffer,
            index_allocation: Mutex::new(index_allocation),
            index_capacity,
            index_offset: AtomicU64::new(0),
        })
    }

    /// Upload vertices to the vertex heap
    ///
    /// # Safety
    /// Caller must ensure device, command pool, and queue are valid
    pub unsafe fn upload_vertices<T: bytemuck::Pod>(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        vertices: &[T],
    ) -> Result<u64> {
        let size = std::mem::size_of_val(vertices) as u64;

        // Align to 16 bytes for buffer_reference_align
        let aligned_size = (size + VERTEX_ALIGNMENT - 1) & !(VERTEX_ALIGNMENT - 1);

        let offset = self.vertex_offset.fetch_add(aligned_size, Ordering::SeqCst);

        if offset + aligned_size > self.vertex_capacity {
            return Err(AshError::VulkanError(
                "DualHeapGeometryBuffer: Vertex heap overflow".to_string(),
            ));
        }

        self.upload_data(
            command_pool,
            queue,
            self.vertex_heap,
            offset,
            bytemuck::cast_slice(vertices),
        )?;

        Ok(offset)
    }

    /// Upload indices to the index heap
    ///
    /// # Safety
    /// Caller must ensure device, command pool, and queue are valid
    pub unsafe fn upload_indices(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        indices: &[u32],
    ) -> Result<u64> {
        let count = indices.len() as u64;

        let offset = self.index_offset.fetch_add(count, Ordering::SeqCst);

        if offset + count > self.index_capacity {
            return Err(AshError::VulkanError(
                "DualHeapGeometryBuffer: Index heap overflow".to_string(),
            ));
        }

        let byte_offset = offset * std::mem::size_of::<u32>() as u64;

        self.upload_data(
            command_pool,
            queue,
            self.index_buffer,
            byte_offset,
            bytemuck::cast_slice(indices),
        )?;

        Ok(byte_offset)
    }

    /// Upload data to a buffer using cmd_update_buffer
    unsafe fn upload_data(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        dst_buffer: vk::Buffer,
        dst_offset: u64,
        data: &[u8],
    ) -> Result<()> {
        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = self
            .device
            .allocate_command_buffers(&cmd_info)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to allocate command buffer: {e}"))
            })?;
        let cmd_buffer = cmd_buffers[0];

        self.device
            .begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| AshError::VulkanError(format!("Failed to begin command buffer: {e}")))?;

        // Split into 65536-byte chunks for cmd_update_buffer
        if data.len() <= 65536 {
            self.device
                .cmd_update_buffer(cmd_buffer, dst_buffer, dst_offset, data);
        } else {
            for (i, chunk) in data.chunks(65536).enumerate() {
                let chunk_offset = dst_offset + (i * 65536) as u64;
                self.device
                    .cmd_update_buffer(cmd_buffer, dst_buffer, chunk_offset, chunk);
            }
        }

        self.device
            .end_command_buffer(cmd_buffer)
            .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

        self.device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .map_err(|e| AshError::VulkanError(format!("Failed to submit queue: {e}")))?;

        self.device
            .queue_wait_idle(queue)
            .map_err(|e| AshError::VulkanError(format!("Failed to wait for queue: {e}")))?;

        self.device.free_command_buffers(command_pool, &cmd_buffers);

        Ok(())
    }

    /// Get the base device address of the vertex heap
    pub fn vertex_heap_address(&self) -> vk::DeviceAddress {
        self.vertex_device_address
    }

    /// Get the index buffer handle for traditional binding
    pub fn index_buffer_handle(&self) -> vk::Buffer {
        self.index_buffer
    }

    /// Get current usage statistics
    pub fn usage_stats(&self) -> (u64, u64, u64, u64) {
        let vertex_used = self.vertex_offset.load(Ordering::Relaxed);
        let index_used = self.index_offset.load(Ordering::Relaxed);
        (
            vertex_used,
            self.vertex_capacity,
            index_used,
            self.index_capacity,
        )
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use by the GPU
    pub unsafe fn destroy(&mut self) {
        let mut vertex_alloc = self.vertex_allocation.lock().unwrap();
        let mut index_alloc = self.index_allocation.lock().unwrap();

        self.allocator
            .destroy_buffer(self.vertex_heap, &mut vertex_alloc);
        self.allocator
            .destroy_buffer(self.index_buffer, &mut index_alloc);

        log::info!("DualHeapGeometryBuffer destroyed");
    }
}

impl Drop for DualHeapGeometryBuffer {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
