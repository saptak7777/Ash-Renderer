use crate::renderer::resources::{BufferAllocation, BufferPool};
use ash::vk;
use std::sync::Arc;

/// Safe vertex buffer wrapper with automatic cleanup
pub struct VertexBuffer {
    allocation: BufferAllocation,
    pool: Arc<BufferPool>,
    vertex_count: u32,
    name: Option<String>,
}

impl VertexBuffer {
    /// Creates a new vertex buffer
    ///
    /// # Safety
    ///
    /// The buffer pool must remain valid for the lifetime of this buffer.
    pub unsafe fn new(
        pool: Arc<BufferPool>,
        vertices: &[f32],
        name: Option<String>,
    ) -> crate::Result<Self> {
        let size = std::mem::size_of_val(vertices) as u64;

        let allocation = pool.allocate(
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk_mem::MemoryUsage::AutoPreferHost,
            name.clone(),
        )?;

        let vertex_count = (vertices.len() / 3) as u32;

        if let Some(ref n) = name {
            log::info!("Created vertex buffer '{n}' with {vertex_count} vertices");
        } else {
            log::info!("Created vertex buffer with {vertex_count} vertices");
        }

        Ok(Self {
            allocation,
            pool,
            vertex_count,
            name,
        })
    }

    /// Returns the Vulkan buffer handle
    pub fn handle(&self) -> vk::Buffer {
        self.allocation.buffer
    }

    /// Returns the number of vertices
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Returns the vertex size in bytes
    pub fn size(&self) -> u64 {
        self.allocation.size
    }

    /// Returns the name if set
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        if let Some(ref name) = self.name {
            log::debug!("Destroying vertex buffer '{name}'");
        }
        self.pool.deallocate(self.allocation.clone());
    }
}

impl std::fmt::Debug for VertexBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexBuffer")
            .field("buffer", &self.allocation.buffer)
            .field("vertex_count", &self.vertex_count)
            .field("size", &self.allocation.size)
            .field("name", &self.name)
            .finish()
    }
}
