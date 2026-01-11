use std::{collections::HashMap, ptr, sync::Arc};

use ash::{vk, Device};
use bytemuck::{bytes_of, Pod, Zeroable};
use vk_mem::Alloc;

use crate::renderer::resources::BufferHandle;
use crate::renderer::{MaterialHandle, Mesh, SkinnedVertex, Vertex};
use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// GPU-resident mesh data managed by `ModelRenderer`.
pub struct UploadedMesh {
    vertex_buffer: BufferHandle,
    index_buffer: Option<BufferHandle>,
    vertex_count: u32,
    index_count: u32,
    pub clusters: Vec<crate::renderer::resources::mesh::MeshCluster>,
}

impl MaterialPushConstants {
    pub fn new(material_handle: MaterialHandle) -> Self {
        Self {
            material_handle,
            ..Default::default()
        }
    }

    pub fn with_material_buffer_index(mut self, index: u32) -> Self {
        self.material_buffer_index = index;
        self
    }

    pub fn with_debug_path(mut self, path: u32) -> Self {
        self.debug_path = path;
        self
    }

    pub fn with_receive_shadows(mut self, enabled: bool) -> Self {
        if enabled {
            self.flags |= 1 << 0;
        } else {
            self.flags &= !(1 << 0);
        }
        self
    }

    pub fn with_debug_visualization(mut self, enabled: bool) -> Self {
        self.debug_visualization_enabled = enabled as u32;
        self
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

    pub fn clusters(&self) -> &[crate::renderer::resources::mesh::MeshCluster] {
        &self.clusters
    }
}

/// Caches GPU-side data for meshes.
pub struct ModelRenderer {
    alloc: Arc<Allocator>,
    device: Arc<Device>,
    cache: HashMap<String, UploadedMesh>,
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
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct MaterialPushConstants {
    pub material_handle: MaterialHandle,
    pub debug_path: u32, // 0: None, 1: GPU-Driven, 2: Legacy
    pub flags: u32,
    pub material_buffer_index: u32,
    pub debug_visualization_enabled: u32, // 0: Disabled, 1: Enabled
    pub _padding: [u32; 3],
}

pub const DRAW_PUSH_VERTEX_BYTES: u32 = 128;
pub const DRAW_PUSH_FRAGMENT_BYTES: u32 = 32;

/// Unified push constants block mirroring the GLSL layout offsets.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawPushConstants {
    // Vertex stage (0-127)
    model: Mat4Push,
    joint_offset: u32,
    use_instancing: u32,
    instance_buffer_index: u32,
    joint_buffer_index: u32,
    _vertex_padding: [u32; 12],

    // Fragment stage (128-159)
    material_index: u32,
    debug_path: u32,
    flags: u32,
    material_buffer_index: u32,
    debug_visualization_enabled: u32,
    _fragment_padding: [u32; 3],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShadowPushConstants {
    pub light_space_matrix: Mat4Push,
    pub model: Mat4Push,
    pub joint_offset: u32,
    pub use_instancing: u32,
    pub instance_buffer_index: u32,
    pub joint_buffer_index: u32,
    pub base_color_index: i32,
    pub _padding: [u32; 3],
}

/// Context for draw calls to reduce argument count
pub struct DrawContext<'a> {
    pub command_buffer: vk::CommandBuffer,
    pub pipeline_layout: vk::PipelineLayout,
    pub uploaded: &'a UploadedMesh,
    pub material: &'a MaterialPushConstants,
    pub instance_buffer_index: u32,
    pub joint_buffer_index: u32,
}

/// Parameters for indirect draw with count buffer
pub struct IndirectDrawCountParams {
    pub indirect_buffer: vk::Buffer,
    pub indirect_offset: vk::DeviceSize,
    pub count_buffer: vk::Buffer,
    pub count_offset: vk::DeviceSize,
    pub max_draw_count: u32,
    pub stride: u32,
}

impl ModelRenderer {
    pub fn new(alloc: Arc<Allocator>, device: Arc<Device>) -> Self {
        Self {
            alloc,
            device,
            cache: HashMap::new(),
        }
    }

    pub fn ensure_mesh(
        &mut self,
        key: &str,
        mesh: &Mesh,
        pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<&UploadedMesh> {
        // Human Pattern: Assertive on logic that should never happen
        if key.is_empty() {
            return Err(AshError::VulkanError(
                "ModelRenderer: Empty key provided for mesh upload".to_string(),
            ));
        }

        if !self.cache.contains_key(key) {
            let uploaded = self.upload_mesh(mesh, pool, queue)?;
            self.cache.insert(key.to_string(), uploaded);
        }

        self.cache
            .get(key)
            .ok_or_else(|| AshError::VulkanError(format!("Mesh '{key}' lost during cache lookup")))
    }

    pub fn get(&self, key: &str) -> Option<&UploadedMesh> {
        // Defensive: handle empty key gracefully here for safety in loops
        if key.is_empty() {
            return None;
        }
        self.cache.get(key)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn uploaded_meshes(&self) -> impl Iterator<Item = (&str, &UploadedMesh)> {
        self.cache.iter().map(|(k, v)| (k.as_str(), v))
    }

    fn upload_mesh(
        &self,
        mesh: &Mesh,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<UploadedMesh> {
        let (v_count, v_ptr, v_size) = if !mesh.skinned_vertices.is_empty() {
            let count = mesh.skinned_vertices.len();
            let size = (count * std::mem::size_of::<SkinnedVertex>()) as vk::DeviceSize;
            (
                count as u32,
                mesh.skinned_vertices.as_ptr() as *const u8,
                size,
            )
        } else {
            let count = mesh.vertices.len();
            let size = (count * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
            (count as u32, mesh.vertices.as_ptr() as *const u8, size)
        };

        let vertex_buffer = self.allocate_and_fill_buffer(
            v_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            v_ptr,
            v_size,
            command_pool,
            queue,
        )?;

        let (index_buffer, i_count) = if let Some(indices) = mesh.indices.as_ref() {
            let i_size = (indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;
            let buffer = self.allocate_and_fill_buffer(
                i_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                indices.as_ptr() as *const u8,
                i_size,
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
            vertex_count: v_count,
            index_count: i_count,
            clusters: mesh.clusters.clone(),
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
                .alloc
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

            {
                let mut guard = self
                    .alloc
                    .map_allocation_guarded(&mut staging_alloc, size)?;
                // Copy data to staging
                ptr::copy_nonoverlapping(data_ptr, guard.as_mut_ptr(), data_size as usize);
            }

            let device_buffer = BufferHandle::new(
                Arc::clone(&self.alloc),
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

            self.alloc
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
    pub unsafe fn draw_mesh(&self, ctx: &DrawContext, model_matrix: glam::Mat4, joint_offset: u32) {
        if ctx.command_buffer == vk::CommandBuffer::null() {
            log::error!("ModelRenderer::draw_mesh called with null command buffer");
            return;
        }

        let vertex_buffer = ctx.uploaded.vertex_buffer();
        if vertex_buffer == vk::Buffer::null() {
            log::warn!("Uploaded mesh missing vertex buffer, skipping draw");
            return;
        }

        self.device
            .cmd_bind_vertex_buffers(ctx.command_buffer, 0, &[vertex_buffer], &[0]);

        if let Some(index_buffer) = ctx.uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
        }

        let material_handle = ctx.material.material_handle;
        log::debug!(
            target: "renderer::push_constants",
            "draw_mesh: material_handle={:?} (idx={}, ver={}), buffer_index={}, debug_path={}, flags={:#X}",
            material_handle,
            material_handle.index,
            material_handle.version,
            ctx.material.material_buffer_index,
            ctx.material.debug_path,
            ctx.material.flags
        );

        // RAGE pattern: Validate material index, fallback to default if invalid
        let material_index = if material_handle.index < 1024 {
            material_handle.index
        } else {
            log::warn!(
                "Invalid material index {}, using default (0)",
                material_handle.index
            );
            0
        };

        let push = DrawPushConstants {
            model: model_matrix.into(),
            joint_offset,
            use_instancing: 0,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            _vertex_padding: [0; 12],
            material_index: ((material_handle.version as u32) << 16) | material_index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _fragment_padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_buffer) = ctx.uploaded.index_buffer() {
            let count = ctx.uploaded.index_count();
            // log::debug!(
            //     "DEBUG: Binding index buffer {:?}, count={}",
            //     index_buffer,
            //     count
            // );
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );

            // log::info!(
            //     "DEBUG: draw_mesh - index_count={}, vertex_count={}",
            //     count,
            //     ctx.uploaded.vertex_count()
            // );
            if count == 0 {
                // log::info!(
                //     "DEBUG: Calling cmd_draw with vertex_count={}",
                //     ctx.uploaded.vertex_count()
                // );
                self.device
                    .cmd_draw(ctx.command_buffer, ctx.uploaded.vertex_count(), 1, 0, 0);
            } else {
                // log::info!("DEBUG: Calling cmd_draw_indexed with index_count={count}");
                self.device
                    .cmd_draw_indexed(ctx.command_buffer, count, 1, 0, 0, 0);
                // log::info!("DEBUG: cmd_draw_indexed completed successfully");
            }
        } else {
            // log::info!(
            //     "DEBUG: draw_mesh - NO index_buffer! Calling cmd_draw with vertex_count={}",
            //     ctx.uploaded.vertex_count()
            // );
            self.device
                .cmd_draw(ctx.command_buffer, ctx.uploaded.vertex_count(), 1, 0, 0);
        }
    }

    /// Draw multiple instances of a mesh
    ///
    /// # Safety
    /// Command buffer must be in recording state and instances must be valid.
    pub unsafe fn draw_mesh_instanced(
        &self,
        ctx: &DrawContext,
        instance_count: u32,
        first_instance: u32,
    ) {
        if instance_count == 0 {
            return;
        }

        let vertex_buffer = ctx.uploaded.vertex_buffer();
        self.device
            .cmd_bind_vertex_buffers(ctx.command_buffer, 0, &[vertex_buffer], &[0]);

        let material_handle = ctx.material.material_handle;
        log::debug!(
            target: "renderer::push_constants",
            "draw_mesh_instanced: material_handle={:?} (idx={}, ver={}), buffer_index={}, debug_path={}, flags={:#X}",
            material_handle,
            material_handle.index,
            material_handle.version,
            ctx.material.material_buffer_index,
            ctx.material.debug_path,
            ctx.material.flags
        );
        let push = DrawPushConstants {
            model: glam::Mat4::IDENTITY.into(),
            joint_offset: 0,
            use_instancing: 1,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            _vertex_padding: [0; 12],
            material_index: ((material_handle.version as u32) << 16) | material_handle.index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _fragment_padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_buffer) = ctx.uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_draw_indexed(
                ctx.command_buffer,
                ctx.uploaded.index_count(),
                instance_count,
                0,
                0,
                first_instance,
            );
        } else {
            self.device.cmd_draw(
                ctx.command_buffer,
                ctx.uploaded.vertex_count(),
                instance_count,
                0,
                first_instance,
            );
        }
    }

    /// Draw multiple instances using an indirect buffer
    ///
    /// # Safety
    /// Command buffer must be in recording state and indirect buffer must be valid with correct layout.
    pub unsafe fn draw_mesh_indirect(
        &self,
        ctx: &DrawContext,
        indirect_buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        let vertex_buffer = ctx.uploaded.vertex_buffer();
        self.device
            .cmd_bind_vertex_buffers(ctx.command_buffer, 0, &[vertex_buffer], &[0]);

        let material_handle = ctx.material.material_handle;
        log::debug!(
            target: "renderer::push_constants",
            "draw_mesh_indirect: material_handle={:?} (idx={}, ver={}), buffer_index={}, debug_path={}, flags={:#X}",
            material_handle,
            material_handle.index,
            material_handle.version,
            ctx.material.material_buffer_index,
            ctx.material.debug_path,
            ctx.material.flags
        );
        let push = DrawPushConstants {
            model: glam::Mat4::IDENTITY.into(),
            joint_offset: 0,
            use_instancing: 1,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            _vertex_padding: [0; 12],
            material_index: ((material_handle.version as u32) << 16) | material_handle.index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _fragment_padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_buffer) = ctx.uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_draw_indexed_indirect(
                ctx.command_buffer,
                indirect_buffer,
                offset,
                draw_count,
                stride,
            );
        } else {
            self.device.cmd_draw_indirect(
                ctx.command_buffer,
                indirect_buffer,
                offset,
                draw_count,
                stride,
            );
        }
    }

    /// Draw multiple instances using indirect count buffer
    ///
    /// # Safety
    /// Command buffer must be in recording state and all buffers must be valid for the current frame.
    pub unsafe fn draw_mesh_indirect_count(
        &self,
        ctx: &DrawContext,
        params: &IndirectDrawCountParams,
    ) {
        let vertex_buffer = ctx.uploaded.vertex_buffer();
        self.device
            .cmd_bind_vertex_buffers(ctx.command_buffer, 0, &[vertex_buffer], &[0]);

        let material_handle = ctx.material.material_handle;
        let push = DrawPushConstants {
            model: glam::Mat4::IDENTITY.into(),
            joint_offset: 0,
            use_instancing: 1,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            _vertex_padding: [0; 12],
            material_index: ((material_handle.version as u32) << 16) | material_handle.index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _fragment_padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_buffer) = ctx.uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_draw_indexed_indirect_count(
                ctx.command_buffer,
                params.indirect_buffer,
                params.indirect_offset,
                params.count_buffer,
                params.count_offset,
                params.max_draw_count,
                params.stride,
            );
        } else {
            // Non-indexed indirect count draw not typically used for meshes but supported
            self.device.cmd_draw_indirect_count(
                ctx.command_buffer,
                params.indirect_buffer,
                params.indirect_offset,
                params.count_buffer,
                params.count_offset,
                params.max_draw_count,
                params.stride,
            );
        }
    }

    /// Record a draw call for shadow mapping.
    ///
    /// # Safety
    /// Caller must ensure the command buffer is recording and that the provided pipeline layout is
    /// compatible with the shadow push constant ranges.
    pub unsafe fn draw_mesh_shadow(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        uploaded: &UploadedMesh,
        push: &ShadowPushConstants,
    ) {
        self.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &bytes_of(push)[0..144],
        );

        self.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            144,
            &bytes_of(push)[144..148],
        );

        if let Some(index_buffer) = uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.device
                .cmd_draw_indexed(command_buffer, uploaded.index_count(), 1, 0, 0, 0);
        } else {
            self.device
                .cmd_draw(command_buffer, uploaded.vertex_count(), 1, 0, 0);
        }
    }

    /// Record an instanced draw call for shadow mapping.
    ///
    /// # Safety
    /// Caller must ensure the command buffer is recording and that the provided pipeline layout is
    /// compatible with the shadow push constant ranges.
    pub unsafe fn draw_mesh_instanced_shadow(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_layout: vk::PipelineLayout,
        uploaded: &UploadedMesh,
        instance_count: u32,
        first_instance: u32,
        push: &ShadowPushConstants,
    ) {
        if instance_count == 0 {
            return;
        }

        self.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &bytes_of(push)[0..144],
        );

        self.device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            144,
            &bytes_of(push)[144..148],
        );

        if let Some(index_buffer) = uploaded.index_buffer() {
            self.device.cmd_bind_index_buffer(
                command_buffer,
                index_buffer,
                0,
                vk::IndexType::UINT32,
            );
            self.device.cmd_draw_indexed(
                command_buffer,
                uploaded.index_count(),
                instance_count,
                0,
                0,
                first_instance,
            );
        } else {
            self.device.cmd_draw(
                command_buffer,
                uploaded.vertex_count(),
                instance_count,
                0,
                first_instance,
            );
        }
    }
}
