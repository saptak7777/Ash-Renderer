use std::{collections::HashMap, ptr, sync::Arc};

use ash::{vk, Device};
use bytemuck::{bytes_of, Pod, Zeroable};
use vk_mem::Alloc;

use crate::renderer::resources::BufferHandle;
use crate::renderer::{Material, Mesh, Vertex};
use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// GPU-resident mesh data managed by `ModelRenderer`.
pub struct UploadedMesh {
    vertex_buffer: BufferHandle,
    index_buffer: Option<BufferHandle>,
    vertex_count: u32,
    index_count: u32,
}

impl MaterialPushConstants {
    pub fn from_material(material: &Material, base_color_binding: Option<u32>) -> Self {
        Self {
            base_color_factor: material.color,
            metallic_factor: material.metallic,
            roughness_factor: material.roughness,
            alpha_cutoff: 0.1,
            alpha_mode: 0,
            base_color_texture_set: base_color_binding.map(|b| b as i32).unwrap_or(-1),
            normal_texture_set: -1,
            metallic_roughness_texture_set: -1,
            occlusion_texture_set: -1,
            emissive_texture_set: -1,
            emissive_factor: material.emissive,
            _padding: [0; 12],
        }
    }
}

impl UploadedMesh {
    pub fn vertex_buffer(&self) -> vk::Buffer {
        self.vertex_buffer.handle()
    }

    pub fn index_buffer(&self) -> Option<vk::Buffer> {
        self.index_buffer.as_ref().map(|buffer| buffer.handle())
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }
}

/// Caches GPU buffers for meshes so multiple entities can reuse uploads.
pub struct ModelRenderer {
    allocator: Arc<Allocator>,
    device: Arc<Device>,
    meshes: HashMap<String, UploadedMesh>,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Mat4Push(pub [f32; 16]);

impl From<glam::Mat4> for Mat4Push {
    fn from(mat: glam::Mat4) -> Self {
        Self(mat.to_cols_array())
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshPushConstants {
    pub model: Mat4Push,
    pub view: Mat4Push,
    pub projection: Mat4Push,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct MaterialPushConstants {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_cutoff: f32,
    pub alpha_mode: i32,
    pub base_color_texture_set: i32,
    pub normal_texture_set: i32,
    pub metallic_roughness_texture_set: i32,
    pub occlusion_texture_set: i32,
    pub emissive_texture_set: i32,
    pub emissive_factor: [f32; 4],
    pub _padding: [u8; 12],
}

impl ModelRenderer {
    pub fn new(allocator: Arc<Allocator>, device: Arc<Device>) -> Self {
        Self {
            allocator,
            device,
            meshes: HashMap::new(),
        }
    }

    pub fn ensure_mesh(
        &mut self,
        key: &str,
        mesh: &Mesh,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<&UploadedMesh> {
        if !self.meshes.contains_key(key) {
            let uploaded = self.upload_mesh(mesh, command_pool, queue)?;
            self.meshes.insert(key.to_string(), uploaded);
        }

        self.meshes
            .get(key)
            .ok_or_else(|| AshError::VulkanError(format!("Mesh '{key}' not found after upload")))
    }

    pub fn get(&self, key: &str) -> Option<&UploadedMesh> {
        self.meshes.get(key)
    }

    pub fn clear(&mut self) {
        self.meshes.clear();
    }

    pub fn uploaded_meshes(&self) -> impl Iterator<Item = (&str, &UploadedMesh)> {
        self.meshes.iter().map(|(k, v)| (k.as_str(), v))
    }

    fn upload_mesh(
        &self,
        mesh: &Mesh,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<UploadedMesh> {
        let vertex_count = mesh.vertices.len() as u32;
        let vertex_size = (mesh.vertices.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;

        let vertex_buffer = self.allocate_and_fill_buffer(
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            mesh.vertices.as_ptr() as *const u8,
            vertex_size,
            command_pool,
            queue,
        )?;

        let (index_buffer, index_count) = if let Some(indices) = mesh.indices.as_ref() {
            let index_size = (indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;
            let buffer = self.allocate_and_fill_buffer(
                index_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                indices.as_ptr() as *const u8,
                index_size,
                command_pool,
                queue,
            )?;
            (Some(buffer), indices.len() as u32)
        } else {
            (None, 0)
        };

        Ok(UploadedMesh {
            vertex_buffer,
            index_buffer,
            vertex_count,
            index_count,
        })
    }

    fn allocate_and_fill_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        data_ptr: *const u8,
        data_size: vk::DeviceSize,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<BufferHandle> {
        unsafe {
            let (staging_buffer, mut staging_alloc) = self
                .allocator
                .vma
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferHost,
                        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create staging buffer: {e}"))
                })?;

            let mapped = self
                .allocator
                .vma
                .map_memory(&mut staging_alloc)
                .map_err(|e| AshError::VulkanError(format!("Failed to map staging buffer: {e}")))?;
            let mapped = mapped.cast::<u8>();
            ptr::copy_nonoverlapping(data_ptr, mapped, data_size as usize);
            self.allocator.vma.unmap_memory(&mut staging_alloc);

            let device_buffer = BufferHandle::new(
                Arc::clone(&self.allocator),
                size,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
                None,
            )?;

            self.copy_buffer(
                command_pool,
                queue,
                staging_buffer,
                device_buffer.handle(),
                size,
            )?;

            self.allocator
                .vma
                .destroy_buffer(staging_buffer, &mut staging_alloc);

            Ok(device_buffer)
        }
    }

    fn copy_buffer(
        &self,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffers =
                self.device
                    .allocate_command_buffers(&alloc_info)
                    .map_err(|e| {
                        AshError::VulkanError(format!("Failed to allocate command buffer: {e}"))
                    })?;
            let command_buffer = command_buffers[0];

            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to begin command buffer: {e}"))
                })?;

            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };

            self.device
                .cmd_copy_buffer(command_buffer, src, dst, &[region]);

            self.device
                .end_command_buffer(command_buffer)
                .map_err(|e| AshError::VulkanError(format!("Failed to end command buffer: {e}")))?;

            let submit_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&submit_buffers);

            self.device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to submit copy command: {e}"))
                })?;
            self.device.queue_wait_idle(queue).map_err(|e| {
                AshError::VulkanError(format!("Failed to wait for queue idle: {e}"))
            })?;
            self.device
                .free_command_buffers(command_pool, &command_buffers);
        }

        Ok(())
    }

    /// Record a draw call for a single uploaded mesh using push constants.
    ///
    /// # Safety
    /// Caller must ensure the command buffer is recording and that the provided pipeline layout is
    /// compatible with the push constant ranges used here. The referenced mesh buffers must remain
    /// valid for the duration of the call.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn draw_mesh(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        uploaded: &UploadedMesh,
        model_matrix: glam::Mat4,
        view_matrix: glam::Mat4,
        projection_matrix: glam::Mat4,
        material: &MaterialPushConstants,
    ) {
        if command_buffer == vk::CommandBuffer::null() {
            log::error!("ModelRenderer::draw_mesh called with null command buffer");
            return;
        }

        let vertex_buffer = uploaded.vertex_buffer();
        if vertex_buffer == vk::Buffer::null() {
            log::warn!("Uploaded mesh missing vertex buffer, skipping draw");
            return;
        }

        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);

        if let Some(index_buffer) = uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
        }

        let push = MeshPushConstants {
            model: model_matrix.into(),
            view: view_matrix.into(),
            projection: projection_matrix.into(),
        };

        self.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytes_of(&push),
        );

        let material_offset = std::mem::size_of::<MeshPushConstants>() as u32;
        self.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            material_offset,
            bytes_of(material),
        );

        if let Some(index_buffer) = uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );

            let count = uploaded.index_count();
            if count == 0 {
                self.device
                    .cmd_draw(command_buffer, uploaded.vertex_count(), 1, 0, 0);
            } else {
                self.device
                    .cmd_draw_indexed(command_buffer, count, 1, 0, 0, 0);
            }
        } else {
            self.device
                .cmd_draw(command_buffer, uploaded.vertex_count(), 1, 0, 0);
        }
    }
}
