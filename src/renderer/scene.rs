use crate::renderer::features::{DirectionalLight, PointLight, SceneLighting, SpotLight};
use crate::renderer::resources::uniform::{MaterialUniform, StorageBuffer};
use crate::renderer::resources::DualHeapGeometryBuffer;
use crate::renderer::resources::TransformSystem;
use crate::renderer::vcgs::culling::CullObjectData;
use crate::renderer::vcgs::OcclusionCulling;
use crate::renderer::*;
use crate::vulkan::Allocator;
use crate::{AshError, Result};
use ash::vk;
use std::collections::HashSet;
use std::sync::{Arc, RwLock}; // Removed Mutex
                              // Added AshError

pub type BindlessDescriptorSet = crate::vulkan::BindlessManager;
pub type CpuMesh = crate::renderer::resources::Mesh;
pub type StandardMemoryAllocator = crate::vulkan::Allocator;

/// Represents a 3D scene containing models, materials, and lights.
/// This decouples scene data from the Renderer logic.
pub struct Scene {
    pub model_renderer: ModelRenderer,
    pub material_manager: MaterialManager,
    pub occlusion_culling: OcclusionCulling,

    // Lighting Data
    pub point_lights: Vec<PointLight>,
    pub directional_lights: Vec<DirectionalLight>,
    pub spot_lights: Vec<SpotLight>,
    pub scene_lighting: SceneLighting, // IBL settings, etc.
    pub skybox_texture_index: u32,

    // Metadata & Tracking (Moved from Renderer)
    pub mesh_data: Vec<MeshData>,
    pub uploaded_material_indices: HashSet<u32>,
    pub transform_system: TransformSystem,

    // Buffers moved from Renderer
    pub global_cluster_buffer: Option<Arc<GlobalClusterBuffer>>,
    pub material_storage_buffer: Option<Arc<RwLock<StorageBuffer<MaterialUniform>>>>,
}

impl Scene {
    pub fn register_material(&mut self, material: &Material) -> Result<MaterialHandle> {
        let handle = self.material_manager.register_material(material)?;

        if !self.uploaded_material_indices.contains(&handle.index) {
            if let Some(buffer_arc) = self.material_storage_buffer.as_ref() {
                let mut buffer = buffer_arc.write().unwrap();
                unsafe {
                    buffer.write_element_at(handle.index as usize, &material.to_uniform())?;
                }
            }
            self.uploaded_material_indices.insert(handle.index);
        }

        Ok(handle)
    }

    pub fn new(
        device: Arc<ash::Device>,
        alloc: Arc<Allocator>,
        geometry_buffer: Arc<DualHeapGeometryBuffer>,
    ) -> Result<Self> {
        let model_renderer =
            ModelRenderer::new(Arc::clone(&alloc), Arc::clone(&device), geometry_buffer);
        let transform_system = TransformSystem::new(Arc::clone(&device), Arc::clone(&alloc))?;

        Ok(Self {
            model_renderer,
            material_manager: MaterialManager::new(),
            occlusion_culling: OcclusionCulling::new(),
            point_lights: Vec::new(),
            directional_lights: Vec::new(),
            spot_lights: Vec::new(),
            scene_lighting: SceneLighting::default(),
            skybox_texture_index: 0,
            mesh_data: Vec::new(),
            uploaded_material_indices: HashSet::new(),
            transform_system,
            global_cluster_buffer: None,
            material_storage_buffer: None,
        })
    }

    /// Register mesh metadata in the scene.
    pub fn register_mesh_handle(&mut self, data: MeshData) -> u32 {
        let handle = self.mesh_data.len() as u32;
        self.mesh_data.push(data);
        handle
    }

    /// Uploads a mesh to the GPU and registers it with the scene.
    pub fn upload_mesh(
        &mut self,
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        queue: &vk::Queue,
        mesh: &mut CpuMesh,
        asset_manager: &mut AssetManager,
        staging_resources: &mut Vec<crate::renderer::resources::BufferHandle>,
        material_override: Option<MaterialHandle>,
    ) -> Result<u32> {
        // 0. Strict check for cluster buffer
        if self.global_cluster_buffer.is_none() {
            return Err(AshError::vulkan("Critical: Global Cluster Buffer missing during mesh upload. Ensure scene.global_cluster_buffer is assigned."));
        }

        let key = mesh.name.clone();

        // 1. Upload geometry to ModelRenderer
        self.model_renderer
            .ensure_mesh(&key, mesh, command_pool, *queue)?;

        // 2. Register textures with bindless manager via AssetManager
        asset_manager.ingest_mesh_textures(
            device,
            allocator.clone(),
            &command_pool,
            queue,
            mesh,
        )?;

        // 3. Register Material
        // Logic: Prioritize material_override > mesh.material_properties > default
        let mut material_handle =
            material_override.unwrap_or_else(|| self.material_manager.default_material());

        if material_override.is_none() {
            if let Some(props) = &mesh.material_properties {
                let material = Material {
                    name: format!("{}_material", &*mesh.name),
                    color: props.base_color_factor,
                    metallic: props.metallic_factor,
                    roughness: props.roughness_factor,
                    emissive: props.emissive_factor,
                    occlusion_strength: props.occlusion_strength,
                    normal_scale: props.normal_scale,
                    alpha_cutoff: props.alpha_cutoff,
                    tint_index: -1,
                    is_transparent: props.base_color_factor[3] < 1.0,
                    texture_index: mesh.texture_index,
                    normal_texture_index: mesh.normal_texture_index,
                    metallic_roughness_texture_index: mesh.metallic_roughness_texture_index,
                    occlusion_texture_index: mesh.occlusion_texture_index,
                    emissive_texture_index: mesh.emissive_texture_index,
                };

                material_handle = self.register_material(&material)?;
            }
        }

        // 4. Cluster Upload
        let mut cluster_start_index = 0;
        if let Some(buffer) = self.global_cluster_buffer.as_ref() {
            if !mesh.clusters.is_empty() {
                // Convert MeshCluster to CullObjectData
                let cull_objects: Vec<CullObjectData> = mesh
                    .clusters
                    .iter()
                    .map(|c| {
                        let identity = glam::Mat4::IDENTITY;
                        let cols = identity.to_cols_array_2d();

                        CullObjectData {
                            model_row0: cols[0],
                            model_row1: cols[1],
                            model_row2: cols[2],
                            model_row3: cols[3],
                            bounds: crate::renderer::vcgs::CullBoundingBox {
                                center: [
                                    c.bounds_center[0],
                                    c.bounds_center[1],
                                    c.bounds_center[2],
                                    c.bounds_radius,
                                ],
                                extents: [c.bounds_radius, c.bounds_radius, c.bounds_radius, 0.0],
                            },
                            parent_index: c.parent_index,
                            first_index: c.first_index,
                            index_count: c.index_count,
                            error_metric: c.error_metric,
                            flags: 1, // Enabled
                            material_index: material_handle.index,
                            ..Default::default()
                        }
                    })
                    .collect();

                // Create staging buffer for clusters
                let total_size =
                    (cull_objects.len() * std::mem::size_of::<CullObjectData>()) as u64;
                let mut staging_buffer_handle = unsafe {
                    crate::renderer::resources::BufferHandle::new_with_flags(
                        Arc::clone(&allocator),
                        total_size,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        vk_mem::MemoryUsage::AutoPreferHost,
                        vk_mem::AllocationCreateFlags::MAPPED
                            | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        Some("ClusterStaging".to_string()),
                    )?
                };

                // Copy data to staged memory
                {
                    let allocation_info = allocator
                        .vma
                        .get_allocation_info(staging_buffer_handle.allocation());
                    if !allocation_info.mapped_data.is_null() {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                cull_objects.as_ptr() as *const u8,
                                allocation_info.mapped_data as *mut u8,
                                total_size as usize,
                            );
                            // CRITICAL: Flush memory to ensure GPU visibility before copy command
                            allocator.vma.flush_allocation(
                                staging_buffer_handle.allocation(),
                                0,
                                vk::WHOLE_SIZE,
                            )?;
                        }
                    } else {
                        // Fallback: Map it
                        let mut guard = unsafe {
                            allocator.map_allocation_guarded(
                                staging_buffer_handle.allocation_mut(),
                                total_size as u64,
                            )?
                        };
                        guard.copy_from_slice(bytemuck::cast_slice::<CullObjectData, u8>(
                            &cull_objects,
                        ));
                    }
                }

                // Record upload command
                cluster_start_index = unsafe {
                    buffer.upload_clusters(
                        command_buffer,
                        staging_buffer_handle.handle(),
                        0,
                        cull_objects.len() as u32,
                    )?
                };

                // Safety: Push staging buffer to the list to ensure it outlives GPU execution
                staging_resources.push(staging_buffer_handle);
            }
        }

        // 5. Calculate bounding box from mesh vertices
        let bounds = if !mesh.vertices.is_empty() {
            let mut min = glam::Vec3::splat(f32::MAX);
            let mut max = glam::Vec3::splat(f32::MIN);
            for vertex in &mesh.vertices {
                let pos = glam::Vec3::from(vertex.position);
                min = min.min(pos);
                max = max.max(pos);
            }
            CullBoundingBox::from_min_max(min, max)
        } else {
            // Fallback to unit cube if no vertices
            CullBoundingBox::new(glam::Vec3::ZERO, glam::Vec3::ONE)
        };

        // 6. Store metadata
        let indices = [
            mesh.texture_index.unwrap_or(u32::MAX),
            mesh.normal_texture_index.unwrap_or(u32::MAX),
            mesh.metallic_roughness_texture_index.unwrap_or(u32::MAX),
            mesh.occlusion_texture_index.unwrap_or(u32::MAX),
        ];

        let data = MeshData {
            name: key,
            texture_indices: indices,
            emissive_index: mesh.emissive_texture_index.unwrap_or(u32::MAX),
            texture_flags: TexturePresenceFlags::from_mesh(mesh),
            material_handle,
            is_hidden: false,
            bounds,
            cluster_start_index,
            cluster_count: mesh.clusters.len() as u32,
        };

        Ok(self.register_mesh_handle(data))
    }

    /// Set the lighting configuration for the scene.
    pub fn set_lighting(&mut self, lighting: SceneLighting) {
        self.scene_lighting = lighting;
    }

    pub fn add_directional_light(&mut self, light: DirectionalLight) {
        self.directional_lights.push(light);
    }

    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }

    pub fn add_spot_light(&mut self, light: SpotLight) {
        self.spot_lights.push(light);
    }

    /// Helper to gather Buffer Device Addresses (BDA) for geometry logic.
    /// Returns (vertex_ptr, index_ptr).
    pub fn get_geometry_buffer_addresses(&self) -> (u64, u64) {
        let vertex_ptr = self.model_renderer.geometry_buffer.vertex_heap_address();
        let index_ptr = self.model_renderer.geometry_buffer.index_heap_address();
        (vertex_ptr, index_ptr)
    }
}
