#![allow(deprecated)]

use ash::vk;
use glam::{IVec4, Mat4, Vec3, Vec4};
use std::sync::Arc;
use vk_mem::Alloc;

/// Uniform buffer data for MVP matrices (Phase 5: improved memory management)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MvpMatrices {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
    pub view_proj: Mat4,
    pub prev_view_proj: Mat4, // For TAA motion vectors
    pub light_space_matrix: Mat4,
    pub normal_matrix: Mat4,
    pub camera_pos: Vec4,
    pub light_direction: Vec4,
    pub light_color: Vec4,
    pub ambient_color: Vec4,
}

/// Material parameters exposed to the GPU
#[repr(C)]
#[derive(Debug, Clone, Copy)]

pub struct MaterialUniform {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec4,
    /// x: metallic, y: roughness, z: occlusion strength, w: normal scale
    pub parameters: Vec4,
    /// Bindless texture indices (-1 when texture is absent)
    /// x: base_color, y: normal, z: metallic_roughness, w: occlusion
    pub texture_indices: IVec4,
    pub emissive_texture_index: i32,
    pub alpha_cutoff: f32,
    pub _padding: [f32; 2],
}

impl Default for MaterialUniform {
    fn default() -> Self {
        Self {
            base_color_factor: Vec4::splat(1.0),
            emissive_factor: Vec4::ZERO,
            parameters: Vec4::new(0.0, 0.5, 1.0, 1.0),
            texture_indices: IVec4::splat(-1),
            emissive_texture_index: -1,
            alpha_cutoff: 0.1,
            _padding: [0.0; 2],
        }
    }
}

impl MaterialUniform {
    pub fn set_base_color_factor(&mut self, color: Vec4) {
        self.base_color_factor = color;
    }

    pub fn set_emissive_factor(&mut self, emissive: Vec4) {
        self.emissive_factor = emissive;
    }

    pub fn set_metallic_roughness(&mut self, metallic: f32, roughness: f32) {
        self.parameters.x = metallic;
        self.parameters.y = roughness;
    }

    pub fn set_occlusion_strength(&mut self, occlusion: f32) {
        self.parameters.z = occlusion;
    }

    pub fn set_normal_scale(&mut self, normal_scale: f32) {
        self.parameters.w = normal_scale;
    }

    pub fn set_alpha_cutoff(&mut self, cutoff: f32) {
        self.alpha_cutoff = cutoff;
    }

    pub fn set_texture_indices(
        &mut self,
        base_color: i32,
        normal: i32,
        metallic_roughness: i32,
        occlusion: i32,
        emissive: i32,
    ) {
        self.texture_indices = IVec4::new(base_color, normal, metallic_roughness, occlusion);
        self.emissive_texture_index = emissive;
    }
}

impl Default for MvpMatrices {
    fn default() -> Self {
        Self {
            model: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            projection: Mat4::IDENTITY,
            view_proj: Mat4::IDENTITY,
            prev_view_proj: Mat4::IDENTITY,
            light_space_matrix: Mat4::IDENTITY,
            normal_matrix: Mat4::IDENTITY,
            camera_pos: Vec4::ZERO,
            light_direction: Vec4::new(0.0, -1.0, 0.0, 0.0),
            light_color: Vec4::splat(1.0),
            ambient_color: Vec4::splat(0.1),
        }
    }
}

impl MvpMatrices {
    fn recalc_view_proj(&mut self) {
        // Save previous view_proj for TAA motion vectors
        self.prev_view_proj = self.view_proj;
        self.view_proj = self.projection * self.view;
    }

    /// Update model matrix from position, rotation, scale
    pub fn set_model(&mut self, position: Vec3, rotation: Vec3, scale: Vec3) {
        let translation = Mat4::from_translation(position);
        let rotation_x = Mat4::from_rotation_x(rotation.x);
        let rotation_y = Mat4::from_rotation_y(rotation.y);
        let rotation_z = Mat4::from_rotation_z(rotation.z);
        let scale_mat = Mat4::from_scale(scale);

        self.model = translation * rotation_z * rotation_y * rotation_x * scale_mat;
    }

    /// Set view matrix from camera position and look-at target
    pub fn set_view(&mut self, eye: Vec3, center: Vec3, up: Vec3) {
        self.view = Mat4::look_at_rh(eye, center, up);
        self.camera_pos = eye.extend(1.0);
        self.recalc_view_proj();
    }

    /// Set perspective projection matrix
    /// Note: Vulkan has Y pointing down in NDC, so we flip Y
    pub fn set_projection(&mut self, fovy: f32, aspect: f32, near: f32, far: f32) {
        self.projection = Mat4::perspective_rh(fovy, aspect, near, far);
        // Flip Y for Vulkan's coordinate system (Y points down in NDC)
        self.projection.y_axis.y *= -1.0;

        self.recalc_view_proj();
    }

    /// Configure lighting terms for the frame
    pub fn set_lighting(&mut self, direction: Vec3, light_color: Vec3, ambient_color: Vec3) {
        self.light_direction = direction.normalize_or_zero().extend(0.0);
        self.light_color = light_color.extend(0.0);
        self.ambient_color = ambient_color.extend(0.0);
    }

    /// Set the light-space matrix for shadow mapping
    pub fn set_light_space_matrix(&mut self, matrix: Mat4) {
        self.light_space_matrix = matrix;
    }
}

/// Uniform buffer wrapper with Phase 5 improvements
pub struct UniformBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub data: MvpMatrices,
    allocator: Arc<crate::vulkan::Allocator>,
    device: Arc<ash::Device>,
    destroyed: bool,
}

impl UniformBuffer {
    /// # Safety
    /// Requires valid allocator, device, and proper vulkan context
    pub unsafe fn new(
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
    ) -> crate::Result<Self> {
        let size = std::mem::size_of::<MvpMatrices>() as u64;

        let (buffer, mut allocation) = allocator
            .vma
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to create uniform buffer: {e}"))
            })?;

        let data = MvpMatrices::default();

        {
            let data_ptr = allocator.vma.map_memory(&mut allocation).map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to map uniform buffer: {e}"))
            })?;

            std::ptr::copy_nonoverlapping(
                &data as *const MvpMatrices as *const u8,
                data_ptr,
                size as usize,
            );

            allocator
                .vma
                .flush_allocation(&allocation, 0, size)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to flush uniform buffer: {e}"))
                })?;

            allocator.vma.unmap_memory(&mut allocation);
        }

        log::info!("Created uniform buffer ({size} bytes)");

        Ok(Self {
            buffer,
            allocation,
            data,
            allocator,
            device,
            destroyed: false,
        })
    }

    /// # Safety
    /// Requires valid allocation and proper memory access
    pub unsafe fn update(&mut self) -> crate::Result<()> {
        let size = std::mem::size_of::<MvpMatrices>() as u64;

        let data_ptr = self
            .allocator
            .vma
            .map_memory(&mut self.allocation)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to map uniform buffer: {e}"))
            })?;

        std::ptr::copy_nonoverlapping(
            &self.data as *const MvpMatrices as *const u8,
            data_ptr,
            size as usize,
        );

        self.allocator
            .vma
            .flush_allocation(&self.allocation, 0, size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush uniform buffer: {e}"))
            })?;

        self.allocator.vma.unmap_memory(&mut self.allocation);

        Ok(())
    }

    /// Get mutable reference to matrices for updates
    pub fn matrices_mut(&mut self) -> &mut MvpMatrices {
        &mut self.data
    }

    /// Get reference to matrices
    pub fn matrices(&self) -> &MvpMatrices {
        &self.data
    }

    /// Phase 5: Proper cleanup - called before destruction
    pub fn cleanup(&mut self) -> crate::Result<()> {
        if self.destroyed {
            return Ok(());
        }

        log::debug!("Cleaning up uniform buffer");

        unsafe {
            // Wait for device to finish all operations
            let _ = self.device.device_wait_idle();

            // Destroy buffer and allocation
            self.allocator
                .vma
                .destroy_buffer(self.buffer, &mut self.allocation);
        }

        self.buffer = vk::Buffer::null();
        self.destroyed = true;

        Ok(())
    }
}

impl Drop for UniformBuffer {
    fn drop(&mut self) {
        let _ = self.cleanup();
        log::debug!("UniformBuffer dropped");
    }
}

/// GPU buffer wrapper for material parameters
pub struct MaterialBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub data: MaterialUniform,
    allocator: Arc<crate::vulkan::Allocator>,
    device: Arc<ash::Device>,
    destroyed: bool,
}

impl MaterialBuffer {
    /// # Safety
    /// Requires a valid allocator and device
    pub unsafe fn new(
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
    ) -> crate::Result<Self> {
        let size = std::mem::size_of::<MaterialUniform>() as u64;

        let (buffer, mut allocation) = allocator
            .vma
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to create material buffer: {e}"))
            })?;

        let data = MaterialUniform::default();

        {
            let data_ptr = allocator.vma.map_memory(&mut allocation).map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to map material buffer: {e}"))
            })?;

            std::ptr::copy_nonoverlapping(
                &data as *const MaterialUniform as *const u8,
                data_ptr,
                size as usize,
            );

            allocator
                .vma
                .flush_allocation(&allocation, 0, size)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to flush material buffer: {e}"))
                })?;

            allocator.vma.unmap_memory(&mut allocation);
        }

        log::info!("Created material buffer ({size} bytes)");

        Ok(Self {
            buffer,
            allocation,
            data,
            allocator,
            device,
            destroyed: false,
        })
    }

    /// # Safety
    /// Requires valid allocation and proper memory access
    pub unsafe fn update(&mut self) -> crate::Result<()> {
        let size = std::mem::size_of::<MaterialUniform>() as u64;

        let data_ptr = self
            .allocator
            .vma
            .map_memory(&mut self.allocation)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to map material buffer: {e}"))
            })?;

        std::ptr::copy_nonoverlapping(
            &self.data as *const MaterialUniform as *const u8,
            data_ptr,
            size as usize,
        );

        self.allocator
            .vma
            .flush_allocation(&self.allocation, 0, size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush material buffer: {e}"))
            })?;

        self.allocator.vma.unmap_memory(&mut self.allocation);

        Ok(())
    }

    pub fn uniform_mut(&mut self) -> &mut MaterialUniform {
        &mut self.data
    }

    pub fn uniform(&self) -> &MaterialUniform {
        &self.data
    }

    pub fn cleanup(&mut self) -> crate::Result<()> {
        if self.destroyed {
            return Ok(());
        }

        log::debug!("Cleaning up material buffer");

        unsafe {
            let _ = self.device.device_wait_idle();
            self.allocator
                .vma
                .destroy_buffer(self.buffer, &mut self.allocation);
        }

        self.buffer = vk::Buffer::null();
        self.destroyed = true;

        Ok(())
    }
}

impl Drop for MaterialBuffer {
    fn drop(&mut self) {
        let _ = self.cleanup();
        log::debug!("MaterialBuffer dropped");
    }
}
