use std::{collections::HashMap, sync::Arc};

use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};

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
}

impl MaterialPushConstants {
    pub fn new(handle: MaterialHandle) -> Self {
        Self {
            light_ptr_low: 0,
            light_ptr_high: 0,
            tile_ptr_low: 0,
            tile_ptr_high: 0,
            material_handle: handle,
            flags: 0,
            material_buffer_index: 0,
            debug_visualization_enabled: 0,
            skybox_index: 0,
            _padding_1: 0,
            _padding_2: 0,
            _padding_3: 0,
        }
    }

    pub fn with_material_buffer_index(mut self, index: u32) -> Self {
        self.material_buffer_index = index;
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
    pub light_ptr_low: u32,
    pub light_ptr_high: u32,
    pub tile_ptr_low: u32,
    pub tile_ptr_high: u32,
    pub material_handle: MaterialHandle, // 4 bytes (u16+u16)
    pub flags: u32,
    pub material_buffer_index: u32,
    pub debug_visualization_enabled: u32,
    pub skybox_index: u32,
    pub _padding_1: u32,
    pub _padding_2: u32,
    pub _padding_3: u32,
}

pub const DRAW_PUSH_VERTEX_BYTES: u32 = 128;
pub const DRAW_PUSH_FRAGMENT_BYTES: u32 = 32;

/// Unified push constants block mirroring the GLSL layout offsets.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawPushConstants {
    // Pointer stage (0-55)
    frame_ptr_low: u32,     // 0
    frame_ptr_high: u32,    // 4
    vertex_ptr_low: u32,    // 8
    vertex_ptr_high: u32,   // 12
    instance_ptr_low: u32,  // 16
    instance_ptr_high: u32, // 20
    material_ptr_low: u32,  // 24
    material_ptr_high: u32, // 28
    index_ptr_low: u32,     // 32
    index_ptr_high: u32,    // 36
    light_ptr_low: u32,     // 40
    light_ptr_high: u32,    // 44
    tile_ptr_low: u32,      // 48
    tile_ptr_high: u32,     // 52

    // Texture indices (56-63)
    vsm_page_index: u32,  // 56
    vsm_cache_index: u32, // 60

    // Control stage (64-127)
    model: Mat4Push,                  // 64
    material_index: u32,              // 128
    use_instancing: u32,              // 132
    flags: u32,                       // 136
    debug_path: u32,                  // 140
    debug_visualization_enabled: u32, // 144
    skybox_index: u32,                // 148
    _padding: [u32; 2],               // 152 (Total 160)
}

/// Context for draw calls with BDA support
pub struct DrawContext<'a> {
    pub command_buffer: vk::CommandBuffer,
    pub pipeline_layout: vk::PipelineLayout,
    pub uploaded: &'a UploadedMesh,
    pub material: &'a MaterialPushConstants,

    // BDA Pointers for the current frame
    pub frame_ptr: u64,
    pub vertex_ptr: u64,
    pub instance_ptr: u64,
    pub material_ptr: u64,
    pub index_ptr: u64,
    pub light_ptr: u64,
    pub tile_ptr: u64,

    // Bindless Indices
    pub vsm_page_index: u32,
    pub vsm_cache_index: u32,
    pub skybox_index: u32,

    pub model: glam::Mat4, // Carrying the transform from RenderCommand
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
        let v_count = mesh.vertices.len() as u32;

        // Upload vertices to the Vertex Heap
        let (vertex_offset, vertex_heap_address) = unsafe {
            let offset =
                self.geometry_buffer
                    .upload_vertices(command_pool, queue, &mesh.vertices)?;

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
        })
    }

    /// Specialized draw call for shadow map generation.
    /// Uses traditional instancing for CPU-side batch submission.
    ///
    /// # Safety
    /// Command buffer must be in recording state and instances must be valid.
    pub unsafe fn draw_shadow_batch(
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
        let vertex_ptr = ctx.uploaded.vertex_heap_address.unwrap_or(0);

        if vertex_ptr == 0 {
            log::error!("CRITICAL (Instanced): vertex_heap_ptr is NULL!");
            return;
        }

        let push = DrawPushConstants {
            frame_ptr_low: ctx.frame_ptr as u32,
            frame_ptr_high: (ctx.frame_ptr >> 32) as u32,
            vertex_ptr_low: vertex_ptr as u32,
            vertex_ptr_high: (vertex_ptr >> 32) as u32,
            instance_ptr_low: ctx.instance_ptr as u32,
            instance_ptr_high: (ctx.instance_ptr >> 32) as u32,
            material_ptr_low: ctx.material_ptr as u32,
            material_ptr_high: (ctx.material_ptr >> 32) as u32,
            index_ptr_low: ctx.index_ptr as u32,
            index_ptr_high: (ctx.index_ptr >> 32) as u32,
            light_ptr_low: ctx.light_ptr as u32,
            light_ptr_high: (ctx.light_ptr >> 32) as u32,
            tile_ptr_low: ctx.tile_ptr as u32,
            tile_ptr_high: (ctx.tile_ptr >> 32) as u32,
            vsm_page_index: ctx.vsm_page_index,
            vsm_cache_index: ctx.vsm_cache_index,
            model: ctx.model.into(),
            material_index: material_handle.index as u32,
            use_instancing: 0,
            flags: ctx.material.flags,
            debug_path: 0,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            skybox_index: ctx.material.skybox_index,
            _padding: [0; 2],
        };

        let push_bytes = bytemuck::bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        // Software Index Pulling (Phase 1):
        // We use cmd_draw to drive gl_VertexIndex as an iterator into the index heap.
        // firstVertex MUST be the offset into the index buffer (in elements).
        let first_vertex = (ctx.uploaded.index_offset.unwrap_or(0) / 4) as u32;

        if ctx.uploaded.has_indices() {
            self.device.cmd_draw(
                ctx.command_buffer,
                ctx.uploaded.index_count(),
                instance_count,
                first_vertex,
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

    /// Draw multiple instances from various meshes using an indirect count buffer.
    /// This is the "Ultra Modern" path where the GPU handles all culling and counting.
    ///
    /// # Safety
    /// Command buffer must be in recording state and all buffers must be valid.
    pub unsafe fn draw_indirect_count(&self, ctx: &DrawContext, params: &IndirectDrawCountParams) {
        let vertex_ptr = ctx.uploaded.vertex_heap_address.unwrap_or(0);
        if vertex_ptr == 0 {
            log::error!("CRITICAL: vertex_heap_ptr is NULL in draw_indirect_count! Skipping draw.");
            return;
        }

        // We use a dummy push constant block because the shader pulls EVERYTHING from BDA
        let push = DrawPushConstants {
            frame_ptr_low: ctx.frame_ptr as u32,
            frame_ptr_high: (ctx.frame_ptr >> 32) as u32,
            vertex_ptr_low: vertex_ptr as u32,
            vertex_ptr_high: (vertex_ptr >> 32) as u32,
            instance_ptr_low: ctx.instance_ptr as u32,
            instance_ptr_high: (ctx.instance_ptr >> 32) as u32,
            material_ptr_low: ctx.material_ptr as u32,
            material_ptr_high: (ctx.material_ptr >> 32) as u32,
            index_ptr_low: ctx.index_ptr as u32,
            index_ptr_high: (ctx.index_ptr >> 32) as u32,
            light_ptr_low: ctx.light_ptr as u32,
            light_ptr_high: (ctx.light_ptr >> 32) as u32,
            tile_ptr_low: ctx.tile_ptr as u32,
            tile_ptr_high: (ctx.tile_ptr >> 32) as u32,
            vsm_page_index: ctx.vsm_page_index,
            vsm_cache_index: ctx.vsm_cache_index,
            model: ctx.model.into(),
            material_index: ctx.material.material_handle.index as u32,
            use_instancing: 1,
            flags: ctx.material.flags,
            debug_path: 0,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            skybox_index: ctx.material.skybox_index,
            _padding: [0; 2],
        };

        let push_bytes = bytemuck::bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        // Software Index Pulling (Phase 1): Always use non-indexed indirect draw
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

    /// Draw multiple instances using indirect count buffer
    ///
    /// # Safety
    /// Command buffer must be in recording state and all buffers must be valid for the current frame.
    pub unsafe fn draw_mesh_indirect_count(
        &self,
        ctx: &DrawContext,
        params: &IndirectDrawCountParams,
    ) {
        let material_handle = ctx.material.material_handle;
        let push = DrawPushConstants {
            frame_ptr_low: ctx.frame_ptr as u32,
            frame_ptr_high: (ctx.frame_ptr >> 32) as u32,
            vertex_ptr_low: ctx.vertex_ptr as u32,
            vertex_ptr_high: (ctx.vertex_ptr >> 32) as u32,
            instance_ptr_low: ctx.instance_ptr as u32,
            instance_ptr_high: (ctx.instance_ptr >> 32) as u32,
            material_ptr_low: ctx.material_ptr as u32,
            material_ptr_high: (ctx.material_ptr >> 32) as u32,
            index_ptr_low: ctx.index_ptr as u32,
            index_ptr_high: (ctx.index_ptr >> 32) as u32,
            light_ptr_low: ctx.light_ptr as u32,
            light_ptr_high: (ctx.light_ptr >> 32) as u32,
            tile_ptr_low: ctx.tile_ptr as u32,
            tile_ptr_high: (ctx.tile_ptr >> 32) as u32,
            vsm_page_index: ctx.vsm_page_index,
            vsm_cache_index: ctx.vsm_cache_index,
            model: ctx.model.into(),
            material_index: material_handle.index as u32,
            use_instancing: 1,
            flags: ctx.material.flags,
            debug_path: 0,
            debug_visualization_enabled: ctx.material.debug_visualization_enabled,
            skybox_index: ctx.skybox_index,
            _padding: [0; 2],
        };

        let push_bytes = bytemuck::bytes_of(&push);

        self.device.cmd_push_constants(
            ctx.command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            push_bytes,
        );

        // Software Index Pulling (Phase 1): Always use non-indexed indirect draw
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
