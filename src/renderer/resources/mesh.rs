#[cfg(feature = "gltf_loading")]
use ash::vk;
use std::sync::Arc;
use vk_mem::Alloc;

use super::texture::{Texture, TextureData};
use super::texture_compressor::{CompressionFormat, TextureCompressor};
use crate::renderer::Material;

/// Mesh Cluster for fine-grained culling (Nanite Phase 3)
#[derive(Debug, Clone, Copy, Default)]
pub struct MeshCluster {
    pub first_index: u32,
    pub index_count: u32,
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}

/// Vertex struct with position, normal, UV, and color
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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
    pub key: Arc<str>,
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
    pub alpha_cutoff: f32,
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
            alpha_cutoff: 0.1,
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

/// Submesh descriptor for multi-material meshes (Phase 2)
#[derive(Debug, Clone, Default)]
pub struct SubmeshDescriptor {
    pub start_index: u32,
    pub index_count: u32,
    pub material_slot: u32, // Index into material_handles
    pub name: Arc<str>,
}

/// GPU Mesh with vertex/index buffers uploaded (PHASE 3)
#[derive(Default, Clone)]
pub struct Mesh {
    pub name: Arc<str>,
    pub vertices: Vec<Vertex>,
    pub skinned_vertices: Vec<crate::renderer::SkinnedVertex>,
    pub indices: Option<Vec<u32>>,
    pub texture_data: Option<TextureData>,
    pub texture: Option<Arc<Texture>>,
    pub texture_path: Option<std::path::PathBuf>,

    // Phase 2: Multi-material support foundation
    pub material_handle: Option<u32>, // Single material (Phase 1)
    pub material_handles: Vec<u32>,   // Multiple materials (future)
    pub submeshes: Vec<SubmeshDescriptor>, // Future submesh descriptors

    pub normal_texture_data: Option<TextureData>,
    pub normal_texture: Option<Arc<Texture>>,
    pub normal_texture_path: Option<std::path::PathBuf>,
    pub metallic_roughness_texture_data: Option<TextureData>,
    pub metallic_roughness_texture: Option<Arc<Texture>>,
    pub metallic_roughness_texture_path: Option<std::path::PathBuf>,
    pub occlusion_texture_data: Option<TextureData>,
    pub occlusion_texture: Option<Arc<Texture>>,
    pub occlusion_texture_path: Option<std::path::PathBuf>,
    pub emissive_texture_data: Option<TextureData>,
    pub emissive_texture: Option<Arc<Texture>>,
    pub emissive_texture_path: Option<std::path::PathBuf>,
    pub material_properties: Option<MaterialProperties>,

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

    pub clusters: Vec<MeshCluster>,

    allocator: Option<Arc<crate::vulkan::Allocator>>,
}

impl Mesh {
    /// Creates a colored cube mesh
    pub fn create_cube() -> Self {
        Self::create_named_cube("Cube")
    }

    pub fn create_named_cube(name: impl Into<Arc<str>>) -> Self {
        // Each face has its own set of vertices to ensure correct normals and UVs
        let vertices = vec![
            // Front face (red)
            Vertex {
                position: [-1.0, -1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            // Back face (green)
            Vertex {
                position: [1.0, -1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                normal: [0.0, 0.0, -1.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [-1.0, 0.0, 0.0, 1.0],
            },
            // Top face (blue)
            Vertex {
                position: [-1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            // Bottom face (yellow)
            Vertex {
                position: [-1.0, -1.0, -1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, -1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, 1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0, 1.0],
                normal: [0.0, -1.0, 0.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            },
            // Right face (cyan)
            Vertex {
                position: [1.0, -1.0, 1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, -1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                normal: [1.0, 0.0, 0.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, -1.0, 1.0],
            },
            // Left face (magenta)
            Vertex {
                position: [-1.0, -1.0, -1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0, 1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, 1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [1.0, 1.0],
                color: [1.0, 1.0, 1.0],
                tangent: [0.0, 0.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [0.0, 1.0],
                color: [1.0, 1.0, 1.0],
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

        let clusters = Self::generate_clusters(&indices, &vertices);

        Self {
            name: name.into(),
            vertices,
            skinned_vertices: Vec::new(),
            indices: Some(indices),
            texture_data: None,
            texture: None,
            texture_path: None,
            material_handle: None,
            material_handles: Vec::new(),
            submeshes: Vec::new(),
            normal_texture_data: None,
            normal_texture: None,
            normal_texture_path: None,
            metallic_roughness_texture_data: None,
            metallic_roughness_texture: None,
            metallic_roughness_texture_path: None,
            occlusion_texture_data: None,
            occlusion_texture: None,
            occlusion_texture_path: None,
            emissive_texture_data: None,
            emissive_texture: None,
            emissive_texture_path: None,
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
            clusters,
            allocator: None,
        }
    }

    /// Split mesh into clusters for fine-grained culling
    pub fn generate_clusters(indices: &[u32], vertices: &[Vertex]) -> Vec<MeshCluster> {
        let cluster_size = 128 * 3; // 128 triangles
        let mut clusters = Vec::new();

        for chunk_indices in indices.chunks(cluster_size) {
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];

            for &idx in chunk_indices {
                if let Some(v) = vertices.get(idx as usize) {
                    for i in 0..3 {
                        min[i] = min[i].min(v.position[i]);
                        max[i] = max[i].max(v.position[i]);
                    }
                }
            }

            let center = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];

            let mut radius_sq = 0.0f32;
            for &idx in chunk_indices {
                if let Some(v) = vertices.get(idx as usize) {
                    let d = [
                        v.position[0] - center[0],
                        v.position[1] - center[1],
                        v.position[2] - center[2],
                    ];
                    let dist_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    radius_sq = radius_sq.max(dist_sq);
                }
            }

            // Find global offset in indices buffer
            let first_index = (chunk_indices.as_ptr() as usize - indices.as_ptr() as usize)
                / std::mem::size_of::<u32>();

            clusters.push(MeshCluster {
                first_index: first_index as u32,
                index_count: chunk_indices.len() as u32,
                bounds_center: center,
                bounds_radius: radius_sq.sqrt(),
            });
        }

        log::debug!("Generated {} clusters for mesh", clusters.len());
        clusters
    }

    pub fn from_descriptor(descriptor: &MeshDescriptor) -> Self {
        let clusters = Self::generate_clusters(
            descriptor.indices.as_ref().unwrap_or(&vec![]),
            &descriptor.vertices,
        );

        Self {
            name: descriptor.key.clone(),
            vertices: descriptor.vertices.clone(),
            skinned_vertices: Vec::new(),
            indices: descriptor.indices.clone(),
            texture_data: descriptor.texture.clone(),
            texture: None,
            texture_path: None,
            material_handle: None,
            material_handles: Vec::new(),
            submeshes: Vec::new(),
            normal_texture_data: descriptor.normal_texture.clone(),
            normal_texture: None,
            normal_texture_path: None,
            metallic_roughness_texture_data: descriptor.metallic_roughness_texture.clone(),
            metallic_roughness_texture: None,
            metallic_roughness_texture_path: None,
            occlusion_texture_data: descriptor.occlusion_texture.clone(),
            occlusion_texture: None,
            occlusion_texture_path: None,
            emissive_texture_data: descriptor.emissive_texture.clone(),
            emissive_texture: None,
            emissive_texture_path: None,
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
            clusters,
            allocator: None,
        }
    }

    /// Merges multiple meshes into a single mesh.
    ///
    /// Useful for reducing draw calls when materials are compatible or ignored.
    pub fn merge(meshes: Vec<Self>) -> crate::Result<Self> {
        if meshes.is_empty() {
            return Err(crate::AshError::VulkanError(
                "No meshes to merge".to_string(),
            ));
        }

        let mut iter = meshes.into_iter();
        let mut merged = iter.next().unwrap();

        for mut mesh in iter {
            let vertex_offset = merged.vertices.len() as u32;
            let index_offset = merged.indices.as_ref().map_or(0, |i| i.len()) as u32;

            // Append vertices using mem::take to avoid moving out of Drop type
            let vertices = std::mem::take(&mut mesh.vertices);
            merged.vertices.extend(vertices);

            // Append indices with vertex offset
            if let Some(idx_list) = mesh.indices.take() {
                let merged_indices = merged.indices.get_or_insert_with(Vec::new);
                for idx in idx_list {
                    merged_indices.push(idx + vertex_offset);
                }
            }

            // Append submeshes with index offset
            let submeshes = std::mem::take(&mut mesh.submeshes);
            for mut submesh in submeshes {
                submesh.start_index += index_offset;
                merged.submeshes.push(submesh);
            }
        }

        // Recalculate clusters for the final merged mesh
        if let Some(ref indices) = merged.indices {
            merged.clusters = Self::generate_clusters(indices, &merged.vertices);
        }

        Ok(merged)
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
        {
            let mut guard =
                unsafe { allocator.map_allocation_guarded(&mut staging_alloc, vertex_size)? };
            guard.copy_from_slice(&self.vertices);
        }

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

            {
                let mut guard =
                    unsafe { allocator.map_allocation_guarded(&mut staging_alloc, index_size)? };
                guard.copy_from_slice(indices);
            }

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
                self.texture = Some(Arc::new(texture));
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
        vram_budget: &mut crate::renderer::vram_budget::VramBudget,
        compression_enabled: bool,
    ) -> crate::Result<()> {
        #[allow(clippy::too_many_arguments)]
        unsafe fn upload_texture_map(
            mesh_name: &str,
            map_name: &str,
            allocator: &Arc<crate::vulkan::Allocator>,
            device: &Arc<ash::Device>,
            command_pool: vk::CommandPool,
            queue: vk::Queue,
            texture: &mut Option<Arc<Texture>>,
            data: &mut Option<TextureData>,
            srgb: bool,
            compression: CompressionFormat,
            vram_budget: &mut crate::renderer::vram_budget::VramBudget,
            compression_enabled: bool,
        ) -> crate::Result<()> {
            if texture.is_none() {
                if let Some(mut texture_data) = data.take() {
                    let mut format = compression.to_vk_format(srgb);

                    if compression_enabled && compression != CompressionFormat::None {
                        log::debug!(
                            "Compressing {map_name} texture for mesh '{mesh_name}' using {compression:?}"
                        );
                        match compression {
                            CompressionFormat::Bc7 => {
                                if let Ok(compressed) =
                                    TextureCompressor::compress_bc7(&texture_data)
                                {
                                    texture_data.pixels = compressed;
                                } else {
                                    log::warn!("BC7 compression failed for {map_name} on '{mesh_name}', falling back to uncompressed.");
                                    format = CompressionFormat::None.to_vk_format(srgb);
                                }
                            }
                            CompressionFormat::Bc5 => {
                                if let Ok(compressed) =
                                    TextureCompressor::compress_bc5(&texture_data)
                                {
                                    log::debug!("BC5 compressed size: {} bytes", compressed.len());
                                    texture_data.pixels = compressed;
                                } else {
                                    log::warn!("BC5 compression failed for {map_name} on '{mesh_name}', falling back to uncompressed.");
                                    format = CompressionFormat::None.to_vk_format(srgb);
                                }
                            }
                            _ => {}
                        }
                    } else {
                        // Force uncompressed if global toggle is off or None requested
                        format = CompressionFormat::None.to_vk_format(srgb);
                    }

                    // Estimate size with mipmaps (base * 1.33)
                    let estimated_total =
                        (texture_data.pixels.len() as f64 * 1.33) as vk::DeviceSize;

                    if let Err(e) = vram_budget.can_allocate(estimated_total) {
                        log::warn!(
                            "VRAM budget exceeded for {map_name} texture on mesh '{mesh_name}': {e}. Using fallback."
                        );
                        // Create 1x1 fallback
                        let fallback_data = TextureData::solid_color(if map_name == "normal" {
                            [128, 128, 255, 255]
                        } else {
                            [255, 255, 255, 255]
                        });
                        let gpu_texture = Texture::from_data(
                            Arc::clone(allocator),
                            Arc::clone(device),
                            command_pool,
                            queue,
                            &fallback_data,
                            format,
                            Some(&format!("{mesh_name}_{map_name}_fallback")),
                        )?;
                        *texture = Some(Arc::new(gpu_texture));
                        vram_budget.allocate(4); // Minimal
                    } else {
                        log::info!(
                            "Uploading {map_name} texture for mesh '{mesh_name}' (Estimated: {}MB)",
                            estimated_total / 1024 / 1024
                        );
                        let gpu_texture = Texture::from_data(
                            Arc::clone(allocator),
                            Arc::clone(device),
                            command_pool,
                            queue,
                            &texture_data,
                            format,
                            Some(&format!("{mesh_name}_{map_name}")),
                        )?;
                        *texture = Some(Arc::new(gpu_texture));
                        vram_budget.allocate(estimated_total);
                    }
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
            true, // srgb
            CompressionFormat::Bc7,
            vram_budget,
            compression_enabled,
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
            false, // unorm
            CompressionFormat::Bc5,
            vram_budget,
            compression_enabled,
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
            false,                  // unorm
            CompressionFormat::Bc7, // or Bc5 if 2 channels, but MR is often packed
            vram_budget,
            compression_enabled,
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
            false, // unorm
            CompressionFormat::Bc7,
            vram_budget,
            compression_enabled,
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
            true, // srgb
            CompressionFormat::Bc7,
            vram_budget,
            compression_enabled,
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
        self.texture.as_deref()
    }

    pub fn normal_texture(&self) -> Option<&Texture> {
        self.normal_texture.as_deref()
    }

    pub fn metallic_roughness_texture(&self) -> Option<&Texture> {
        self.metallic_roughness_texture.as_deref()
    }

    pub fn occlusion_texture(&self) -> Option<&Texture> {
        self.occlusion_texture.as_deref()
    }

    pub fn emissive_texture(&self) -> Option<&Texture> {
        self.emissive_texture.as_deref()
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

    /// Request loading of textures via the streamer
    pub fn request_loads(&self, streamer: &crate::renderer::resources::TextureStreamer) {
        if self.texture.is_none() {
            if let Some(path) = &self.texture_path {
                streamer.request_load(path, 0, "base_color");
            }
        }
        if self.normal_texture.is_none() {
            if let Some(path) = &self.normal_texture_path {
                streamer.request_load(path, 0, "normal");
            }
        }
        if self.metallic_roughness_texture.is_none() {
            if let Some(path) = &self.metallic_roughness_texture_path {
                streamer.request_load(path, 0, "metallic_roughness");
            }
        }
        if self.occlusion_texture.is_none() {
            if let Some(path) = &self.occlusion_texture_path {
                streamer.request_load(path, 0, "occlusion");
            }
        }
        if self.emissive_texture.is_none() {
            if let Some(path) = &self.emissive_texture_path {
                streamer.request_load(path, 0, "emissive");
            }
        }
    }

    /// Poll the streamer for loaded textures
    pub fn poll_streaming(&mut self, streamer: &crate::renderer::resources::TextureStreamer) {
        if self.texture.is_none() {
            if let Some(path) = &self.texture_path {
                if let Some(tex) = streamer.try_get(path) {
                    self.texture = Some(tex);
                }
            }
        }
        if self.normal_texture.is_none() {
            if let Some(path) = &self.normal_texture_path {
                if let Some(tex) = streamer.try_get(path) {
                    self.normal_texture = Some(tex);
                }
            }
        }
        if self.metallic_roughness_texture.is_none() {
            if let Some(path) = &self.metallic_roughness_texture_path {
                if let Some(tex) = streamer.try_get(path) {
                    self.metallic_roughness_texture = Some(tex);
                }
            }
        }
        if self.occlusion_texture.is_none() {
            if let Some(path) = &self.occlusion_texture_path {
                if let Some(tex) = streamer.try_get(path) {
                    self.occlusion_texture = Some(tex);
                }
            }
        }
        if self.emissive_texture.is_none() {
            if let Some(path) = &self.emissive_texture_path {
                if let Some(tex) = streamer.try_get(path) {
                    self.emissive_texture = Some(tex);
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_meshes() {
        let m1 = Mesh::create_named_cube("Cube1");
        let m2 = Mesh::create_named_cube("Cube2");

        let v1_count = m1.vertices.len();
        let i1_count = m1.indices.as_ref().unwrap().len();

        let merged = Mesh::merge(vec![m1, m2]).expect("Merge failed");

        assert_eq!(merged.vertices.len(), v1_count * 2);
        assert_eq!(merged.indices.as_ref().unwrap().len(), i1_count * 2);

        // Verify index offset: first index of second mesh should be v1_count
        let indices = merged.indices.as_ref().unwrap();
        assert_eq!(indices[i1_count], v1_count as u32);

        // Clusters are generated for merged mesh
        assert!(!merged.clusters.is_empty());
    }

    #[test]
    fn test_merge_empty() {
        let result = Mesh::merge(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_gltf_first_mock() {
        let m1 = Mesh::create_named_cube("First");
        let m2 = Mesh::create_named_cube("Second");
        let mut meshes = vec![m1, m2];
        let first = meshes.remove(0);
        assert_eq!(&*first.name, "First");
    }

    #[test]
    fn test_mesh_selection_logic() {
        let meshes = vec![
            Mesh::create_named_cube("PartA"),
            Mesh::create_named_cube("PartB"),
        ];

        // Test index selection
        assert_eq!(&*meshes.first().unwrap().name, "PartA");
        assert_eq!(&*meshes.get(1).unwrap().name, "PartB");
        assert!(meshes.get(2).is_none());

        // Test named selection
        let part_b = meshes.iter().find(|m| &*m.name == "PartB");
        assert!(part_b.is_some());
        assert_eq!(&*part_b.unwrap().name, "PartB");

        let missing = meshes.iter().find(|m| &*m.name == "Missing");
        assert!(missing.is_none());
    }

    #[test]
    fn test_arc_str_clone_cost() {
        use std::time::Instant;
        let iterations = 100_000;

        let name_str = "A very long mesh name that would definitely require a heap allocation for každá single clone if it were a String";

        // Benchmark String cloning
        let s = name_str.to_string();
        let start = Instant::now();
        let mut string_clones = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            string_clones.push(s.clone());
        }
        let string_duration = start.elapsed();
        log::debug!("String cloning ({iterations} iterations): {string_duration:?}");

        // Benchmark Arc<str> cloning
        let arc: Arc<str> = name_str.into();
        let start = Instant::now();
        let mut arc_clones = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            arc_clones.push(Arc::clone(&arc));
        }
        let arc_duration = start.elapsed();
        log::debug!("Arc cloning ({iterations} iterations): {arc_duration:?}");

        assert!(
            arc_duration < string_duration,
            "Arc<str> should be faster than String for cloning"
        );
        let speedup = string_duration.as_secs_f64() / arc_duration.as_secs_f64();
        log::debug!("Speedup factor: {:.2}x", speedup);
    }
}
