//! Global Geometry Buffer for Vertex Pulling
//!
//! Replaces per-mesh vertex buffers with a single unified SSBO.
//! All geometry is stored in one massive buffer, accessed via offsets.

use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::vulkan::Allocator;
use crate::{AshError, Result};

use super::gpu_buffer::GpuBuffer;
use super::mesh::Vertex;
use crate::renderer::SkinnedVertex;

/// Allocation result containing offsets into the global buffers
#[derive(Debug, Clone, Copy)]
pub struct MeshAllocation {
    /// Byte offset into the global vertex buffer
    pub vertex_offset: u64,
    /// Number of vertices allocated
    pub vertex_count: u32,
    /// Byte offset into the global index buffer
    pub index_offset: u64,
    /// Number of indices allocated
    pub index_count: u32,
}

/// Global geometry buffer manager for vertex pulling
pub struct GlobalGeometryBuffer {
    /// Static vertex buffer (Vertex: 64 bytes)
    static_vertex_buffer: GpuBuffer<u8>,
    /// Skinned vertex buffer (SkinnedVertex: 64 bytes)
    skinned_vertex_buffer: GpuBuffer<u8>,
    /// Unified index buffer (all meshes)
    index_buffer: GpuBuffer<u32>,
    /// Current allocation offset for static vertices
    static_vertex_offset: AtomicU64,
    /// Current allocation offset for skinned vertices
    skinned_vertex_offset: AtomicU64,
    /// Current allocation offset for indices
    index_offset: AtomicU64,
    /// Total capacity in bytes for static vertices
    static_vertex_capacity: u64,
    /// Total capacity in bytes for skinned vertices
    skinned_vertex_capacity: u64,
    /// Total capacity in indices
    index_capacity: u64,
}

impl GlobalGeometryBuffer {
    /// Create a new global geometry buffer
    ///
    /// # Arguments
    /// * `allocator` - VMA allocator
    /// * `vertex_capacity_mb` - Static vertex buffer size in megabytes
    /// * `index_capacity_mb` - Index buffer size in megabytes
    ///
    /// # Safety
    /// Caller must ensure allocator is valid and buffers are destroyed before allocator
    pub unsafe fn new(
        allocator: Arc<Allocator>,
        vertex_capacity_mb: u32,
        index_capacity_mb: u32,
    ) -> Result<Self> {
        let static_vertex_capacity = (vertex_capacity_mb as u64) * 1024 * 1024;
        let skinned_vertex_capacity = (vertex_capacity_mb as u64) * 1024 * 1024; // Same size for now
        let index_capacity =
            ((index_capacity_mb as u64) * 1024 * 1024) / std::mem::size_of::<u32>() as u64;

        log::info!(
            "Creating GlobalGeometryBuffer: static_vertex={} MB, skinned_vertex={} MB, index={} MB",
            vertex_capacity_mb,
            vertex_capacity_mb,
            index_capacity_mb
        );

        // Create static vertex buffer
        let static_vertex_buffer = GpuBuffer::new(
            Arc::clone(&allocator),
            static_vertex_capacity as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            None,
        )?;

        // Create skinned vertex buffer
        let skinned_vertex_buffer = GpuBuffer::new(
            Arc::clone(&allocator),
            skinned_vertex_capacity as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            None,
        )?;

        // Create index buffer
        let index_buffer = GpuBuffer::new(
            allocator,
            index_capacity as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk_mem::MemoryUsage::AutoPreferDevice,
            None,
        )?;

        log::info!("✅ GlobalGeometryBuffer created successfully");

        Ok(Self {
            static_vertex_buffer,
            skinned_vertex_buffer,
            index_buffer,
            static_vertex_offset: AtomicU64::new(0),
            skinned_vertex_offset: AtomicU64::new(0),
            index_offset: AtomicU64::new(0),
            static_vertex_capacity,
            skinned_vertex_capacity,
            index_capacity,
        })
    }

    /// Allocate space for a mesh and upload data
    ///
    /// # Arguments
    /// * `device` - Vulkan device
    /// * `command_pool` - Command pool for upload
    /// * `queue` - Queue for upload
    /// * `vertices` - Vertex data (standard or skinned)
    /// * `indices` - Optional index data
    /// * `is_skinned` - Whether this mesh uses skinned vertices
    ///
    /// # Safety
    /// Caller must ensure device, command pool, and queue are valid
    pub unsafe fn allocate_mesh(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        vertices: &[u8],
        indices: Option<&[u32]>,
        is_skinned: bool,
    ) -> Result<MeshAllocation> {
        let vertex_size = vertices.len() as u64;
        let index_count = indices.map_or(0, |i| i.len() as u32);

        // Allocate vertex space (route to correct buffer)
        let (vertex_offset, vertex_capacity, vertex_buffer_handle) = if is_skinned {
            let offset = self
                .skinned_vertex_offset
                .fetch_add(vertex_size, Ordering::SeqCst);
            if offset + vertex_size > self.skinned_vertex_capacity {
                return Err(AshError::VulkanError(
                    "GlobalGeometryBuffer: Skinned vertex buffer full".to_string(),
                ));
            }
            (
                offset,
                self.skinned_vertex_capacity,
                self.skinned_vertex_buffer.handle(),
            )
        } else {
            let offset = self
                .static_vertex_offset
                .fetch_add(vertex_size, Ordering::SeqCst);
            if offset + vertex_size > self.static_vertex_capacity {
                return Err(AshError::VulkanError(
                    "GlobalGeometryBuffer: Static vertex buffer full".to_string(),
                ));
            }
            (
                offset,
                self.static_vertex_capacity,
                self.static_vertex_buffer.handle(),
            )
        };

        // Allocate index space
        let index_offset = if let Some(idx) = indices {
            let offset = self
                .index_offset
                .fetch_add(idx.len() as u64, Ordering::SeqCst);
            if offset + idx.len() as u64 > self.index_capacity {
                return Err(AshError::VulkanError(
                    "GlobalGeometryBuffer: Index buffer full".to_string(),
                ));
            }
            offset
        } else {
            0
        };

        // Upload vertex data
        self.upload_data(
            device,
            command_pool,
            queue,
            vertex_buffer_handle,
            vertex_offset,
            vertices,
        )?;

        // Upload index data if present
        if let Some(idx) = indices {
            let index_bytes = bytemuck::cast_slice(idx);
            self.upload_data(
                device,
                command_pool,
                queue,
                self.index_buffer.handle(),
                index_offset * std::mem::size_of::<u32>() as u64,
                index_bytes,
            )?;
        }

        Ok(MeshAllocation {
            vertex_offset,
            vertex_count: (vertex_size / std::mem::size_of::<Vertex>() as u64) as u32,
            index_offset,
            index_count,
        })
    }

    /// Upload data to a buffer using staging buffer
    unsafe fn upload_data(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        dst_buffer: vk::Buffer,
        dst_offset: u64,
        data: &[u8],
    ) -> Result<()> {
        // For now, use a simple approach - create staging via VMA from the vertex_buffer's allocator
        // In production, we'd pass allocator as a parameter
        // This is a simplified version that will be improved in Phase 2

        // Create command buffer
        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd_buffers = device.allocate_command_buffers(&cmd_info).map_err(|e| {
            AshError::VulkanError(format!("Failed to allocate command buffer: {e}"))
        })?;
        let cmd_buffer = cmd_buffers[0];

        device
            .begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| AshError::VulkanError(format!("Failed to begin command buffer: {e}")))?;

        // For initial implementation, we'll use update_buffer for small uploads
        // This is less efficient but avoids the staging buffer complexity
        if data.len() <= 65536 {
            device.cmd_update_buffer(cmd_buffer, dst_buffer, dst_offset, data);
        } else {
            // For larger uploads, we'll need to implement proper staging
            // For now, split into chunks
            for (i, chunk) in data.chunks(65536).enumerate() {
                let chunk_offset = dst_offset + (i * 65536) as u64;
                device.cmd_update_buffer(cmd_buffer, dst_buffer, chunk_offset, chunk);
            }
        }

        device
            .end_command_buffer(cmd_buffer)
            .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);

        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .map_err(|e| AshError::VulkanError(format!("Failed to submit queue: {e}")))?;

        device
            .queue_wait_idle(queue)
            .map_err(|e| AshError::VulkanError(format!("Failed to wait for queue: {e}")))?;

        // Cleanup
        device.free_command_buffers(command_pool, &cmd_buffers);

        Ok(())
    }

    /// Get static vertex buffer handle
    pub fn static_vertex_buffer(&self) -> vk::Buffer {
        self.static_vertex_buffer.handle()
    }

    /// Get skinned vertex buffer handle
    pub fn skinned_vertex_buffer(&self) -> vk::Buffer {
        self.skinned_vertex_buffer.handle()
    }

    /// Get index buffer handle
    pub fn index_buffer(&self) -> vk::Buffer {
        self.index_buffer.handle()
    }

    /// Get current usage statistics
    pub fn usage_stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        let static_vertex_used = self.static_vertex_offset.load(Ordering::Relaxed);
        let skinned_vertex_used = self.skinned_vertex_offset.load(Ordering::Relaxed);
        let index_used = self.index_offset.load(Ordering::Relaxed);
        (
            static_vertex_used,
            self.static_vertex_capacity,
            skinned_vertex_used,
            self.skinned_vertex_capacity,
            index_used,
            self.index_capacity,
        )
    }
}
