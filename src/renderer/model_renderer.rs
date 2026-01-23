use std::{collections::HashMap, sync::Arc};

use ash::{vk, Device};
use bytemuck::{bytes_of, Pod, Zeroable};

use crate::renderer::resources::global_geometry_buffer::DualHeapGeometryBuffer;
use crate::renderer::{MaterialHandle, Mesh};
use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// GPU-resident mesh data managed by `ModelRenderer`.
pub struct UploadedMesh {
    pub index_offset: Option<u64>, // Byte offset into Global Index Heap
    vertex_count: u32,
    index_count: u32,
    pub clusters: Vec<crate::renderer::resources::mesh::MeshCluster>,

    // BDA Fields for Vertex Pulling
    pub vertex_heap_address: Option<vk::DeviceAddress>, // Device address for this mesh's vertices
    pub vertex_offset: Option<u64>,                     // Byte offset into the Vertex Heap
    pub is_skinned: bool,                               // True if uploaded as SkinnedVertex
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
    pub fn has_indices(&self) -> bool {
        self.index_offset.is_some()
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
    device: Arc<Device>,
    pub geometry_buffer: Arc<DualHeapGeometryBuffer>,
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
    vertex_heap_ptr: u64,
    is_skinned: u32,
    _vertex_padding: [u32; 9],

    // Fragment stage (128-159)
    material_index: u32,
    debug_path: u32,
    flags: u32,
    material_buffer_index: u32,
    debug_visualization_enabled: u32,
    _padding: [u32; 3],
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
    pub fn new(
        _alloc: Arc<Allocator>,
        device: Arc<Device>,
        geometry_buffer: Arc<DualHeapGeometryBuffer>,
    ) -> Self {
        Self {
            device,
            geometry_buffer,
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
        let v_count = if !mesh.skinned_vertices.is_empty() {
            mesh.skinned_vertices.len() as u32
        } else {
            mesh.vertices.len() as u32
        };

        // Upload vertices to the Vertex Heap
        let (vertex_offset, vertex_heap_address) = unsafe {
            let offset = if !mesh.skinned_vertices.is_empty() {
                self.geometry_buffer
                    .upload_vertices(command_pool, queue, &mesh.skinned_vertices)?
            } else {
                self.geometry_buffer
                    .upload_vertices(command_pool, queue, &mesh.vertices)?
            };

            let address = self.geometry_buffer.vertex_heap_address() + offset;
            (Some(offset), Some(address))
        };

        // Upload indices to the Index Heap
        let (index_offset, i_count) = if let Some(indices) = mesh.indices.as_ref() {
            let offset = unsafe {
                self.geometry_buffer
                    .upload_indices(command_pool, queue, indices)?
            };
            (Some(offset), indices.len() as u32)
        } else {
            (None, 0)
        };

        Ok(UploadedMesh {
            index_offset,
            vertex_count: v_count,
            index_count: i_count,
            clusters: mesh.clusters.clone(),
            vertex_heap_address,
            vertex_offset,
            is_skinned: !mesh.skinned_vertices.is_empty(),
        })
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

        // BDA Vertex Pulling: No vertex buffer binding needed

        if let Some(index_offset) = ctx.uploaded.index_offset {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                self.geometry_buffer.index_buffer_handle(),
                index_offset,
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

        let vertex_ptr = ctx.uploaded.vertex_heap_address.unwrap_or(0);

        // DIAGNOSTIC: Check for Silent Killer #1 - Null Pointer
        if vertex_ptr == 0 {
            log::error!(
                "CRITICAL: vertex_heap_ptr is NULL (0x0)! Skipping draw to prevent GPU hang. Mesh: vertex_count={}, index_count={}",
                ctx.uploaded.vertex_count(),
                ctx.uploaded.index_count()
            );
            return; // SAFETY: Do not submit draw calls with NULL BDA pointers
        }

        log::debug!(
            "Draw mesh: Ptr=0x{:X}, Indices={}, Vertices={}, IndexOffset={:?}, Skinned={}, JointBuf={}",
            vertex_ptr,
            ctx.uploaded.index_count(),
            ctx.uploaded.vertex_count(),
            ctx.uploaded.index_offset,
            ctx.uploaded.is_skinned,
            ctx.joint_buffer_index
        );

        let push = DrawPushConstants {
            model: model_matrix.into(),
            joint_offset,
            use_instancing: 0,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            vertex_heap_ptr: vertex_ptr,
            is_skinned: ctx.uploaded.is_skinned as u32,
            _vertex_padding: [0; 9],
            material_index: ((material_handle.version as u32) << 16) | material_index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(_index_offset) = ctx.uploaded.index_offset {
            self.device.cmd_draw_indexed(
                ctx.command_buffer,
                ctx.uploaded.index_count(),
                1,
                0,
                0,
                0,
            );
        } else {
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

        // BDA Vertex Pulling: No vertex buffer binding needed

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

        let vertex_ptr = ctx.uploaded.vertex_heap_address.unwrap_or(0);

        // DIAGNOSTIC: Check for Silent Killer #1 - Null Pointer (Instanced Path)
        if vertex_ptr == 0 {
            log::error!(
                "CRITICAL (Instanced): vertex_heap_ptr is NULL (0x0)! Skipping draw to prevent GPU hang. Instances={}, Indices={}",
                instance_count,
                ctx.uploaded.index_count()
            );
            return; // SAFETY: Do not submit instanced draw calls with NULL BDA pointers
        }

        log::debug!(
            "Draw mesh instanced: Ptr=0x{:X}, Instances={}, Indices={}, FirstInstance={}, Skinned={}, JointBuf={}",
            vertex_ptr,
            instance_count,
            ctx.uploaded.index_count(),
            first_instance,
            ctx.uploaded.is_skinned,
            ctx.joint_buffer_index
        );

        let push = DrawPushConstants {
            model: glam::Mat4::IDENTITY.into(),
            joint_offset: 0,
            use_instancing: 1,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            vertex_heap_ptr: vertex_ptr,
            is_skinned: ctx.uploaded.is_skinned as u32,
            _vertex_padding: [0; 9],
            material_index: ((material_handle.version as u32) << 16) | material_handle.index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_offset) = ctx.uploaded.index_offset {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                self.geometry_buffer.index_buffer_handle(),
                index_offset,
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
        // BDA Vertex Pulling: No vertex buffer binding needed

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
            vertex_heap_ptr: ctx.uploaded.vertex_heap_address.unwrap_or(0),
            is_skinned: ctx.uploaded.is_skinned as u32,
            _vertex_padding: [0; 9],
            material_index: ((material_handle.version as u32) << 16) | material_handle.index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_offset) = ctx.uploaded.index_offset {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                self.geometry_buffer.index_buffer_handle(),
                index_offset,
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
        // BDA Vertex Pulling: No vertex buffer binding needed

        let material_handle = ctx.material.material_handle;
        let push = DrawPushConstants {
            model: glam::Mat4::IDENTITY.into(),
            joint_offset: 0,
            use_instancing: 1,
            instance_buffer_index: ctx.instance_buffer_index,
            joint_buffer_index: ctx.joint_buffer_index,
            vertex_heap_ptr: ctx.uploaded.vertex_heap_address.unwrap_or(0),
            is_skinned: ctx.uploaded.is_skinned as u32,
            _vertex_padding: [0; 9],
            material_index: ((material_handle.version as u32) << 16) | material_handle.index as u32,
            debug_path: ctx.material.debug_path,
            flags: ctx.material.flags,
            material_buffer_index: ctx.material.material_buffer_index,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            _padding: [0; 3],
        };

        let push_bytes = bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        if let Some(index_offset) = ctx.uploaded.index_offset {
            self.device.cmd_bind_index_buffer(
                ctx.command_buffer,
                self.geometry_buffer.index_buffer_handle(),
                index_offset,
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
    /// Public method to upload a mesh directly (used for internal meshes like Skybox)
    pub fn upload_mesh_data(
        &self,
        mesh: &Mesh,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<UploadedMesh> {
        self.upload_mesh(mesh, command_pool, queue)
    }
}
