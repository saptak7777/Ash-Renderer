#![allow(deprecated)]

#[cfg(feature = "gltf_loading")]
use archetype_asset::ModelLoader;
use ash::vk;
use std::sync::Arc;
use vk_mem::Alloc;

use super::texture::{Texture, TextureData};
use crate::renderer::Material;

/// Vertex struct with position, normal, UV, and color
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
    pub tangent: [f32; 4],
}

/// Descriptor describing CPU-side mesh data ready for upload.
#[derive(Debug, Clone)]
pub struct MeshDescriptor {
    pub key: String,
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub texture: Option<TextureData>,
    pub normal_texture: Option<TextureData>,
    pub metallic_roughness_texture: Option<TextureData>,
    pub occlusion_texture: Option<TextureData>,
    pub emissive_texture: Option<TextureData>,
    pub material_properties: Option<MaterialProperties>,
}

/// Descriptor describing material properties for renderer registration.
#[derive(Debug, Clone)]
pub struct MaterialDescriptor {
    pub material: Material,
}

/// Surface properties extracted from GLTF materials.
#[derive(Debug, Clone, Copy)]
pub struct MaterialProperties {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 4],
    pub occlusion_strength: f32,
    pub normal_scale: f32,
}

impl Default for MaterialProperties {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            emissive_factor: [0.0, 0.0, 0.0, 1.0],
            occlusion_strength: 1.0,
            normal_scale: 1.0,
        }
    }
}

impl Vertex {
    /// Vulkan vertex binding description
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    /// Vulkan vertex attribute descriptions
    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 5] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 44,
            },
        ]
    }
}

/// GPU Mesh with vertex/index buffers uploaded (PHASE 3)
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Option<Vec<u32>>,
    pub texture_data: Option<TextureData>,
    pub texture: Option<Texture>,
    pub normal_texture_data: Option<TextureData>,
    pub normal_texture: Option<Texture>,
    pub metallic_roughness_texture_data: Option<TextureData>,
    pub metallic_roughness_texture: Option<Texture>,
    pub occlusion_texture_data: Option<TextureData>,
    pub occlusion_texture: Option<Texture>,
    pub emissive_texture_data: Option<TextureData>,
    pub emissive_texture: Option<Texture>,
    material_properties: Option<MaterialProperties>,

    // Phase 3: GPU buffers
    pub vertex_buffer: Option<vk::Buffer>,
    pub vertex_allocation: Option<vk_mem::Allocation>,
    pub index_buffer: Option<vk::Buffer>,
    pub index_allocation: Option<vk_mem::Allocation>,

    // Phase 6: Bindless indices
    pub texture_index: Option<u32>,
    pub normal_texture_index: Option<u32>,
    pub metallic_roughness_texture_index: Option<u32>,
    pub occlusion_texture_index: Option<u32>,
    pub emissive_texture_index: Option<u32>,

    allocator: Option<Arc<crate::vulkan::Allocator>>,
}

impl Mesh {
    /// Creates a colored cube mesh
    pub fn create_cube() -> Self {
        Self::create_named_cube("Cube")
    }

    pub fn create_named_cube(name: impl Into<String>) -> Self {
        // Each face has its own set of vertices to ensure correct normals and UVs
        let vertices = vec![
            // Front face (red)
            Vertex {
                position: [-1.0, -1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                color: [1.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
                color: [1.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
                color: [1.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
                color: [1.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            // Back face (green)
            Vertex {
                position: [1.0, -1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [0.0, 0.0],
                color: [0.0, 1.0, 0.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [1.0, 0.0],
                color: [0.0, 1.0, 0.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [1.0, 1.0],
                color: [0.0, 1.0, 0.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [0.0, 1.0],
                color: [0.0, 1.0, 0.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            // Top face (blue)
            Vertex {
                position: [-1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 0.0],
                color: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 0.0],
                color: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 1.0],
                color: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 1.0],
                color: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            // Bottom face (yellow)
            Vertex {
                position: [-1.0, -1.0, -1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, -1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, 1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0, 1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            // Right face (cyan)
            Vertex {
                position: [1.0, -1.0, 1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                color: [0.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, -1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [1.0, 0.0],
                color: [0.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [1.0, 1.0],
                color: [0.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [0.0, 1.0],
                color: [0.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            // Left face (magenta)
            Vertex {
                position: [-1.0, -1.0, -1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                color: [1.0, 0.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0, 1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [1.0, 0.0],
                color: [1.0, 0.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, 1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [1.0, 1.0],
                color: [1.0, 0.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [0.0, 1.0],
                color: [1.0, 0.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0, // front
            4, 5, 6, 6, 7, 4, // back
            8, 9, 10, 10, 11, 8, // top
            12, 13, 14, 14, 15, 12, // bottom
            16, 17, 18, 18, 19, 16, // right
            20, 21, 22, 22, 23, 20, // left
        ];

        log::info!(
            "Created cube mesh with {} vertices and {} indices",
            vertices.len(),
            indices.len()
        );

        Self {
            name: name.into(),
            vertices,
            indices: Some(indices),
            texture_data: None,
            texture: None,
            normal_texture_data: None,
            normal_texture: None,
            metallic_roughness_texture_data: None,
            metallic_roughness_texture: None,
            occlusion_texture_data: None,
            occlusion_texture: None,
            emissive_texture_data: None,
            emissive_texture: None,
            material_properties: Some(MaterialProperties::default()),
            vertex_buffer: None,
            vertex_allocation: None,
            index_buffer: None,
            index_allocation: None,
            texture_index: None,
            normal_texture_index: None,
            metallic_roughness_texture_index: None,
            occlusion_texture_index: None,
            emissive_texture_index: None,
            allocator: None,
        }
    }

    /// Loads a mesh from a GLB file
    pub fn from_gltf(path: &str) -> crate::Result<Self> {
        let path_obj = std::path::Path::new(path);
        let bytes = std::fs::read(path_obj)
            .map_err(|e| crate::AshError::VulkanError(format!("Failed to read file: {e}")))?;

        let loader = ModelLoader::new();
        let model = loader.load_glb(&bytes).map_err(|e| {
            crate::AshError::VulkanError(format!("Archetype asset load error: {e}"))
        })?;

        let source_mesh = model
            .meshes
            .first()
            .ok_or_else(|| crate::AshError::VulkanError("No meshes found in GLB".to_string()))?;

        // Access mesh data
        let mesh_data = source_mesh.vertices();

        let mut vertices = Vec::with_capacity(mesh_data.vertices.len() / 16);
        for chunk in mesh_data.vertices.chunks(16) {
            if chunk.len() < 16 {
                break;
            }

            // Normals are now valid in v0.1.3
            let normal = [chunk[3], chunk[4], chunk[5]];

            vertices.push(Vertex {
                position: [chunk[0], chunk[1], chunk[2]],
                normal,
                uv: [chunk[6], chunk[7]],
                color: [chunk[12], chunk[13], chunk[14]],
                tangent: [chunk[8], chunk[9], chunk[10], chunk[11]],
            });
        }

        let indices = Some(mesh_data.indices.clone());

        let name = path_obj
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();

        let mut material_properties = Some(MaterialProperties::default());
        let mut texture_data = None;
        let mut normal_texture_data = None;
        let mut metallic_roughness_texture_data = None;
        let mut occlusion_texture_data = None;
        let mut emissive_texture_data = None;

        if let Some(idx) = source_mesh.material_index {
            if let Some(mat) = model.materials.get(idx) {
                // Map properties
                let props = MaterialProperties {
                    base_color_factor: mat.base_color_factor,
                    metallic_factor: mat.metallic_factor,
                    roughness_factor: mat.roughness_factor,
                    emissive_factor: [
                        mat.emissive_factor[0],
                        mat.emissive_factor[1],
                        mat.emissive_factor[2],
                        1.0,
                    ],
                    occlusion_strength: mat.occlusion_strength,
                    normal_scale: mat.normal_scale,
                };
                material_properties = Some(props);

                // Helper to map textures
                let get_texture = |idx: Option<usize>| -> Option<TextureData> {
                    let idx = idx?;
                    let tex = model.textures.get(idx)?;
                    Some(TextureData {
                        width: tex.width,
                        height: tex.height,
                        pixels: tex.data.clone(),
                    })
                };

                texture_data = get_texture(mat.base_color_texture);
                normal_texture_data = get_texture(mat.normal_texture);
                metallic_roughness_texture_data = get_texture(mat.metallic_roughness_texture);
                occlusion_texture_data = get_texture(mat.occlusion_texture);
                emissive_texture_data = get_texture(mat.emissive_texture);
            }
        }

        Ok(Self {
            name,
            vertices,
            indices,
            texture_data,
            texture: None,
            normal_texture_data,
            normal_texture: None,
            metallic_roughness_texture_data,
            metallic_roughness_texture: None,
            occlusion_texture_data,
            occlusion_texture: None,
            emissive_texture_data,
            emissive_texture: None,
            material_properties,
            vertex_buffer: None,
            vertex_allocation: None,
            index_buffer: None,
            index_allocation: None,
            texture_index: None,
            normal_texture_index: None,
            metallic_roughness_texture_index: None,
            occlusion_texture_index: None,
            emissive_texture_index: None,
            allocator: None,
        })
    }

    /// Builds a mesh from a descriptor without uploading to the GPU.
    pub fn from_descriptor(descriptor: &MeshDescriptor) -> Self {
        Self {
            name: descriptor.key.clone(),
            vertices: descriptor.vertices.clone(),
            indices: descriptor.indices.clone(),
            texture_data: descriptor.texture.clone(),
            texture: None,
            normal_texture_data: descriptor.normal_texture.clone(),
            normal_texture: None,
            metallic_roughness_texture_data: descriptor.metallic_roughness_texture.clone(),
            metallic_roughness_texture: None,
            occlusion_texture_data: descriptor.occlusion_texture.clone(),
            occlusion_texture: None,
            emissive_texture_data: descriptor.emissive_texture.clone(),
            emissive_texture: None,
            material_properties: descriptor.material_properties,
            vertex_buffer: None,
            vertex_allocation: None,
            index_buffer: None,
            index_allocation: None,
            texture_index: None,
            normal_texture_index: None,
            metallic_roughness_texture_index: None,
            occlusion_texture_index: None,
            emissive_texture_index: None,
            allocator: None,
        }
    }

    /// Upload mesh data to GPU (Phase 3)
    /// # Safety
    /// Caller must ensure device and queues are valid
    pub unsafe fn upload_to_gpu(
        &mut self,
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> crate::Result<()> {
        log::info!("Uploading mesh '{}' to GPU...", self.name);

        // Create vertex buffer
        let vertex_size = (self.vertices.len() * std::mem::size_of::<Vertex>()) as u64;
        log::info!(
            "  Vertex buffer size: {} bytes ({} vertices)",
            vertex_size,
            self.vertices.len()
        );

        let (staging_buffer, mut staging_alloc) = allocator
            .vma
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(vertex_size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to create staging buffer: {e}"))
            })?;

        // Copy vertex data to staging buffer
        let data_ptr = allocator.vma.map_memory(&mut staging_alloc).map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to map staging buffer: {e}"))
        })?;
        std::ptr::copy_nonoverlapping(
            self.vertices.as_ptr() as *const u8,
            data_ptr,
            vertex_size as usize,
        );
        allocator.vma.unmap_memory(&mut staging_alloc);

        // Create device-local vertex buffer
        let (vertex_buffer, vertex_alloc) = allocator
            .vma
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(vertex_size)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    ..Default::default()
                },
            )
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to create vertex buffer: {e}"))
            })?;

        // Copy from staging to device buffer
        Self::copy_buffer(
            device.as_ref(),
            command_pool,
            queue,
            staging_buffer,
            vertex_buffer,
            vertex_size,
        )?;

        // Cleanup staging buffer
        allocator
            .vma
            .destroy_buffer(staging_buffer, &mut staging_alloc);

        self.vertex_buffer = Some(vertex_buffer);
        self.vertex_allocation = Some(vertex_alloc);

        // Upload indices if present
        if let Some(ref indices) = self.indices {
            let index_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
            log::info!(
                "  Index buffer size: {} bytes ({} indices)",
                index_size,
                indices.len()
            );

            let (staging_buffer, mut staging_alloc) = allocator
                .vma
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(index_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferHost,
                        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to create staging buffer: {e}"))
                })?;

            let data_ptr = allocator.vma.map_memory(&mut staging_alloc).map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to map staging buffer: {e}"))
            })?;
            std::ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                data_ptr,
                index_size as usize,
            );
            allocator.vma.unmap_memory(&mut staging_alloc);

            let (index_buffer, index_alloc) = allocator
                .vma
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(index_size)
                        .usage(
                            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to create index buffer: {e}"))
                })?;

            Self::copy_buffer(
                device.as_ref(),
                command_pool,
                queue,
                staging_buffer,
                index_buffer,
                index_size,
            )?;

            allocator
                .vma
                .destroy_buffer(staging_buffer, &mut staging_alloc);

            self.index_buffer = Some(index_buffer);
            self.index_allocation = Some(index_alloc);
        }

        self.allocator = Some(allocator);
        log::info!("✅ Mesh '{}' uploaded to GPU successfully", self.name);

        if self.texture.is_none() {
            if let Some(ref texture_data) = self.texture_data {
                log::info!("Uploading texture for mesh '{}'", self.name);
                let texture = Texture::from_data(
                    Arc::clone(self.allocator.as_ref().expect("allocator set after upload")),
                    Arc::clone(&device),
                    command_pool,
                    queue,
                    texture_data,
                    vk::Format::R8G8B8A8_SRGB,
                    Some(&self.name),
                )?;
                self.texture = Some(texture);
                self.texture_data = None;
            }
        }

        Ok(())
    }

    /// Ensure the mesh's texture is uploaded to GPU memory.
    ///
    /// This helper is used by the model renderer when vertex/index buffers are
    /// pooled elsewhere but textures still need to be available for sampling.
    ///
    /// # Safety
    /// The caller must guarantee that the allocator, device, command pool, and queue remain
    /// valid for the duration of the upload and that no other operations use the same command
    /// pool concurrently.
    pub unsafe fn ensure_texture(
        &mut self,
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> crate::Result<()> {
        #[allow(clippy::too_many_arguments)]
        unsafe fn upload_texture_map(
            mesh_name: &str,
            map_name: &str,
            allocator: &Arc<crate::vulkan::Allocator>,
            device: &Arc<ash::Device>,
            command_pool: vk::CommandPool,
            queue: vk::Queue,
            texture: &mut Option<Texture>,
            data: &mut Option<TextureData>,
            format: vk::Format,
        ) -> crate::Result<()> {
            if texture.is_none() {
                if let Some(texture_data) = data.take() {
                    log::info!("Uploading {map_name} texture for mesh '{mesh_name}'");
                    let gpu_texture = Texture::from_data(
                        Arc::clone(allocator),
                        Arc::clone(device),
                        command_pool,
                        queue,
                        &texture_data,
                        format,
                        Some(&format!("{mesh_name}_{map_name}")),
                    )?;
                    *texture = Some(gpu_texture);
                }
            }
            Ok(())
        }

        upload_texture_map(
            &self.name,
            "albedo",
            &allocator,
            &device,
            command_pool,
            queue,
            &mut self.texture,
            &mut self.texture_data,
            vk::Format::R8G8B8A8_SRGB,
        )?;
        upload_texture_map(
            &self.name,
            "normal",
            &allocator,
            &device,
            command_pool,
            queue,
            &mut self.normal_texture,
            &mut self.normal_texture_data,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        upload_texture_map(
            &self.name,
            "metallic_roughness",
            &allocator,
            &device,
            command_pool,
            queue,
            &mut self.metallic_roughness_texture,
            &mut self.metallic_roughness_texture_data,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        upload_texture_map(
            &self.name,
            "occlusion",
            &allocator,
            &device,
            command_pool,
            queue,
            &mut self.occlusion_texture,
            &mut self.occlusion_texture_data,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        upload_texture_map(
            &self.name,
            "emissive",
            &allocator,
            &device,
            command_pool,
            queue,
            &mut self.emissive_texture,
            &mut self.emissive_texture_data,
            vk::Format::R8G8B8A8_SRGB,
        )?;

        Ok(())
    }

    /// Helper: Copy buffer using a command buffer
    /// # Safety
    /// Caller must ensure device and queues are valid
    unsafe fn copy_buffer(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: u64,
    ) -> crate::Result<()> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = device.allocate_command_buffers(&alloc_info)?;
        let command_buffer = command_buffers[0];

        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        device.cmd_copy_buffer(command_buffer, src, dst, &[copy_region]);

        device.end_command_buffer(command_buffer)?;

        let submit_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&submit_buffers);

        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
        device.free_command_buffers(command_pool, &command_buffers);

        Ok(())
    }

    /// Returns vertex count
    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Returns index count (if available)
    pub fn index_count(&self) -> Option<u32> {
        self.indices.as_ref().map(|i| i.len() as u32)
    }

    /// Check if mesh has GPU buffers uploaded
    pub fn is_uploaded(&self) -> bool {
        self.vertex_buffer.is_some()
    }

    /// Returns the GPU texture if available
    pub fn texture(&self) -> Option<&Texture> {
        self.texture.as_ref()
    }

    pub fn normal_texture(&self) -> Option<&Texture> {
        self.normal_texture.as_ref()
    }

    pub fn metallic_roughness_texture(&self) -> Option<&Texture> {
        self.metallic_roughness_texture.as_ref()
    }

    pub fn occlusion_texture(&self) -> Option<&Texture> {
        self.occlusion_texture.as_ref()
    }

    pub fn emissive_texture(&self) -> Option<&Texture> {
        self.emissive_texture.as_ref()
    }

    /// Base color factor extracted from GLTF material if available.
    pub fn base_color_factor(&self) -> Option<[f32; 4]> {
        self.material_properties
            .as_ref()
            .map(|props| props.base_color_factor)
    }

    pub fn material_properties(&self) -> Option<&MaterialProperties> {
        self.material_properties.as_ref()
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        if let Some(ref allocator) = self.allocator {
            unsafe {
                // Destroy vertex buffer if still allocated
                if let (Some(buffer), Some(mut allocation)) =
                    (self.vertex_buffer.take(), self.vertex_allocation.take())
                {
                    allocator.vma.destroy_buffer(buffer, &mut allocation);
                }

                // Destroy index buffer if still allocated
                if let (Some(buffer), Some(mut allocation)) =
                    (self.index_buffer.take(), self.index_allocation.take())
                {
                    allocator.vma.destroy_buffer(buffer, &mut allocation);
                }
            }
        }

        log::debug!("Mesh '{}' dropped", self.name);
    }
}
