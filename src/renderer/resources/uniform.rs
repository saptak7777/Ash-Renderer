#![allow(deprecated)]

use ash::vk;
use glam::{IVec4, Mat4, Vec3, Vec4};
use std::sync::Arc;
use vk_mem::Alloc;

use crate::renderer::vcgs::CullObjectData;

/// Uniform buffer data for MVP matrices
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
    pub scene_lighting: crate::renderer::features::SceneLighting,
}

/// Material parameters exposed to the GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct MaterialUniform {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec4,
    /// x: metallic, y: roughness, z: occlusion strength, w: normal scale
    pub parameters: Vec4,
    /// Bindless texture indices (-1 when texture is absent)
    /// x: base_color, y: normal, z: metallic_roughness, w: occlusion
    pub texture_indices: IVec4,
    pub emissive_texture_index: i32,
    pub tint_index: i32,
    pub alpha_cutoff: f32,
    pub _padding: [f32; 1],
}

impl Default for MaterialUniform {
    fn default() -> Self {
        Self {
            base_color_factor: Vec4::splat(1.0),
            emissive_factor: Vec4::ZERO,
            parameters: Vec4::new(0.0, 0.5, 1.0, 1.0),
            texture_indices: IVec4::splat(-1),
            emissive_texture_index: -1,
            tint_index: -1,
            alpha_cutoff: 0.1,
            _padding: [0.0; 1],
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
        tint: i32,
    ) {
        self.texture_indices = IVec4::new(base_color, normal, metallic_roughness, occlusion);
        self.emissive_texture_index = emissive;
        self.tint_index = tint;
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
            scene_lighting: crate::renderer::features::SceneLighting::default(),
        }
    }
}

impl MvpMatrices {
    fn recalc_view_proj(&mut self) {
        // Save previous view_proj for TAA motion vectors
        self.prev_view_proj = self.view_proj;
        self.view_proj = self.projection * self.view;
    }

    /// Update from a modern Transform object
    pub fn set_transform(&mut self, transform: &mut super::Transform) {
        self.model = transform.model_matrix();
        self.normal_matrix = Mat4::from_mat3(transform.normal_matrix());
    }

    /// Update model matrix from position, rotation, scale (Legacy wrapper)
    pub fn set_model(&mut self, position: Vec3, rotation: Vec3, scale: Vec3) {
        let mut transform = super::Transform::from_trs(
            position,
            glam::Quat::from_euler(glam::EulerRot::XYZ, rotation.x, rotation.y, rotation.z),
            scale,
        );
        self.set_transform(&mut transform);
    }

    /// Set view matrix from camera position and look-at target
    pub fn set_view(&mut self, eye: Vec3, center: Vec3, up: Vec3) {
        self.view = Mat4::look_at_rh(eye, center, up);
        self.camera_pos = eye.extend(1.0);
        self.recalc_view_proj();
    }

    /// Set perspective projection matrix
    /// Note: Vulkan NDC (Normalized Device Coordinates) has the Y-axis pointing down; flip Y to compensate.
    pub fn set_projection(&mut self, fovy: f32, aspect: f32, near: f32, far: f32) {
        self.projection = Mat4::perspective_rh(fovy, aspect, near, far);
        // Flip Y for Vulkan's coordinate system (Y points down in NDC)
        self.projection.y_axis.y *= -1.0;

        self.recalc_view_proj();
    }

    /// Configure lighting for the frame
    pub fn set_lighting(&mut self, lighting: &crate::renderer::features::SceneLighting) {
        self.scene_lighting = *lighting;
    }

    /// Set the light-space matrix for shadow mapping
    pub fn set_light_space_matrix(&mut self, matrix: Mat4) {
        self.light_space_matrix = matrix;
    }
}

/// Uniform buffer wrapper
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
    /// Caller must ensure that the provided allocator and device are valid and remain active for the duration of the buffer's life.
    pub unsafe fn new(
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
    ) -> crate::Result<Self> {
        let size = std::mem::size_of::<MvpMatrices>() as u64;

        let (buffer, mut allocation) = unsafe {
            allocator.vma.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    // CRITICAL FIX: Use AutoPreferDevice for BDA support on Intel Arc
                    // Intel Arc doesn't support BDA on host-visible memory for uniform buffers
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to create uniform buffer: {e}"))
        })?;

        let data = MvpMatrices::default();

        {
            let mut guard = unsafe { allocator.map_allocation_guarded(&mut allocation, size) }?;
            guard.copy_from_slice(&[data]);
        }

        allocator
            .vma
            .flush_allocation(&allocation, 0, size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush uniform buffer: {e}"))
            })?;

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

        {
            let mut guard = unsafe {
                self.allocator
                    .map_allocation_guarded(&mut self.allocation, size)
            }?;

            guard.copy_from_slice(&[self.data]);
        }

        // --- Lead Engineer Fix: Aligned Flush ---
        // Ensure the flush range is a multiple of nonCoherentAtomSize (usually 64 or 256)
        const ATOM_SIZE: u64 = 256;
        let aligned_offset = 0; // Starts at 0, so already aligned
        let aligned_size = size.div_ceil(ATOM_SIZE) * ATOM_SIZE;

        self.allocator
            .vma
            .flush_allocation(&self.allocation, aligned_offset, aligned_size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush uniform buffer: {e}"))
            })?;

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

    /// Get the GPU device address for BDA pulling
    pub fn device_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }

    /// Proper cleanup - called before destruction
    pub fn cleanup(&mut self) -> crate::Result<()> {
        if self.destroyed {
            return Ok(());
        }

        log::debug!("Cleaning up uniform buffer");

        unsafe {
            // Wait for device to finish all operations
            // (Removed: device_wait_idle() serialized stall. Parent manages GPU idle state)

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
    }
}

impl crate::renderer::cleanup_traits::VulkanResourceCleanup for UniformBuffer {
    fn cleanup_with_device(&mut self, _device: &ash::Device) -> std::result::Result<(), String> {
        self.cleanup().map_err(|e| e.to_string())
    }

    fn resource_type(&self) -> &'static str {
        "UniformBuffer"
    }
}

impl crate::renderer::resource_registry::VulkanResource for UniformBuffer {}

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
    /// Caller must ensure that the provided allocator and device are valid.
    pub unsafe fn new(
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
    ) -> crate::Result<Self> {
        let size = std::mem::size_of::<MaterialUniform>() as u64;

        let (buffer, mut allocation) = unsafe {
            allocator.vma.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to create material buffer: {e}"))
        })?;

        let data = MaterialUniform::default();

        {
            let mut guard = unsafe { allocator.map_allocation_guarded(&mut allocation, size) }?;
            guard.copy_from_slice(&[data]);
        }

        allocator
            .vma
            .flush_allocation(&allocation, 0, size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush material buffer: {e}"))
            })?;

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

        {
            let mut guard = unsafe {
                self.allocator
                    .map_allocation_guarded(&mut self.allocation, size)
            }?;

            guard.copy_from_slice(&[self.data]);
        }

        // --- Lead Engineer Fix: Aligned Flush ---
        const ATOM_SIZE: u64 = 256;
        let aligned_size = size.div_ceil(ATOM_SIZE) * ATOM_SIZE;

        self.allocator
            .vma
            .flush_allocation(&self.allocation, 0, aligned_size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush material buffer: {e}"))
            })?;

        Ok(())
    }

    pub fn uniform_mut(&mut self) -> &mut MaterialUniform {
        &mut self.data
    }

    pub fn uniform(&self) -> &MaterialUniform {
        &self.data
    }

    /// Get the GPU device address for BDA pulling
    pub fn device_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }

    pub fn cleanup(&mut self) -> crate::Result<()> {
        if self.destroyed {
            return Ok(());
        }

        log::debug!("Cleaning up material buffer");

        unsafe {
            // (Removed: device_wait_idle() serialized stall)
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

/// GPU storage buffer for per-instance data (matches CullObjectData)
pub struct InstanceBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    allocator: Arc<crate::vulkan::Allocator>,
    device: Arc<ash::Device>,
    destroyed: bool,
}

impl InstanceBuffer {
    /// Create a new instance buffer
    ///
    /// # Safety
    /// Caller must ensure that the provided allocator and device are valid.
    pub unsafe fn new(
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
        capacity: usize,
    ) -> crate::Result<Self> {
        let size = (capacity * std::mem::size_of::<CullObjectData>()) as u64;

        let (buffer, allocation) = unsafe {
            allocator.vma.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to create instance buffer: {e}"))
        })?;

        log::info!("Created instance buffer (capacity: {capacity}, size: {size} bytes)");

        Ok(Self {
            buffer,
            allocation,
            allocator,
            device,
            destroyed: false,
        })
    }

    /// Update instance buffer with new data
    ///
    /// # Safety
    /// Caller must ensure the buffer is not currently being read by the GPU (e.g., during culling or drawing). Data must fit within the allocated capacity.
    pub unsafe fn update(
        &mut self,
        data: &[crate::renderer::vcgs::CullObjectData],
    ) -> crate::Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let size = std::mem::size_of_val(data) as u64;
        {
            let mut guard = unsafe {
                self.allocator
                    .map_allocation_guarded(&mut self.allocation, size)
            }?;
            guard.copy_from_slice(data);
        }

        // --- Lead Engineer Fix: Aligned Flush ---
        const ATOM_SIZE: u64 = 256;
        let aligned_size = size.div_ceil(ATOM_SIZE) * ATOM_SIZE;

        self.allocator
            .vma
            .flush_allocation(&self.allocation, 0, aligned_size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush instance buffer: {e}"))
            })?;

        Ok(())
    }

    /// Get the GPU device address for BDA pulling
    pub fn device_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }

    pub fn cleanup(&mut self) -> crate::Result<()> {
        if self.destroyed {
            return Ok(());
        }

        unsafe {
            // (Removed: device_wait_idle() serialized stall)
            self.allocator
                .vma
                .destroy_buffer(self.buffer, &mut self.allocation);
        }

        self.buffer = vk::Buffer::null();
        self.destroyed = true;
        Ok(())
    }
}

impl Drop for InstanceBuffer {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// GPU storage buffer for generic data types
pub struct StorageBuffer<T: Copy> {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    capacity: usize,
    allocator: Arc<crate::vulkan::Allocator>,
    device: Arc<ash::Device>,
    destroyed: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy> StorageBuffer<T> {
    /// Create a new storage buffer with the given capacity
    ///
    /// # Safety
    /// Caller must ensure that the provided allocator and device are valid.
    pub unsafe fn new(
        allocator: Arc<crate::vulkan::Allocator>,
        device: Arc<ash::Device>,
        capacity: usize,
        name: &str,
    ) -> crate::Result<Self> {
        let size = (capacity * std::mem::size_of::<T>()) as u64;

        let (buffer, allocation) = unsafe {
            allocator.vma.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::MAPPED,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| {
            crate::AshError::VulkanError(format!("Failed to create storage buffer '{name}': {e}"))
        })?;

        log::debug!("Created storage buffer '{name}' (capacity: {capacity}, size: {size} bytes)");

        Ok(Self {
            buffer,
            allocation,
            capacity,
            allocator,
            device,
            destroyed: false,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Update storage buffer with new data and flush to GPU
    ///
    /// # Safety
    /// Caller must ensure that the buffer is not in use by the GPU and that the provided data slice length does not exceed the buffer's capacity.
    pub unsafe fn update(&mut self, data: &[T]) -> crate::Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        if data.len() > self.capacity {
            return Err(crate::AshError::VulkanError(format!(
                "Storage buffer update size {} exceeds capacity {}",
                data.len(),
                self.capacity
            )));
        }

        let size = std::mem::size_of_val(data) as u64;
        {
            let mut guard = unsafe {
                self.allocator
                    .map_allocation_guarded(&mut self.allocation, size)
            }?;
            guard.copy_from_slice(data);
        }

        // --- Lead Engineer Fix: Aligned Flush ---
        const ATOM_SIZE: u64 = 256;
        let aligned_size = size.div_ceil(ATOM_SIZE) * ATOM_SIZE;

        // CRITICAL: Ensure GPU sees the data
        self.allocator
            .vma
            .flush_allocation(&self.allocation, 0, aligned_size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush storage buffer: {e}"))
            })?;

        Ok(())
    }

    /// Direct write of a single element at a specific index with full-buffer coherency
    /// This is the AAA pattern used in modern game engines (UE5, Unity) for streaming updates
    /// while avoiding GPU-CPU race conditions.
    ///
    /// Uses persistent mapping (MAPPED flag) to avoid guard scope issues. The buffer is already
    /// mapped at creation time, so we write directly to the persistent pointer and flush.
    ///
    /// # Safety
    /// Caller must ensure the buffer has not been destroyed and that the index is within bounds [0, capacity).
    pub unsafe fn write_element_at(&mut self, index: usize, element: &T) -> crate::Result<()> {
        if index >= self.capacity {
            return Err(crate::AshError::VulkanError(format!(
                "Write index {index} exceeds buffer capacity {}",
                self.capacity
            )));
        }

        let element_size = std::mem::size_of::<T>();
        let offset_bytes = (index * element_size) as u64;
        // let full_size = (self.capacity * element_size) as u64; // Unused for persistent map

        // AAA Pattern: Use persistent mapping (allocated with MAPPED flag)
        let mapped_ptr = self
            .allocator
            .vma
            .get_allocation_info(&self.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            // Unlikely with current allocation flags
            let mut guard = unsafe {
                self.allocator.map_allocation_guarded(
                    &mut self.allocation,
                    (self.capacity * element_size) as u64,
                )
            }?;
            let base = guard.as_mut_ptr() as *mut T;
            unsafe {
                std::ptr::write(base.add(index), *element);
            }
        } else {
            let base = mapped_ptr as *mut T;
            unsafe {
                std::ptr::write(base.add(index), *element);
            }
        }

        // --- Lead Engineer Fix: Aligned Flush Range ---
        // Violating nonCoherentAtomSize (usually 64 or 256) causes silent data loss.
        // Index 1 (Offset 80) is unaligned on 64-byte atom GPUs, failing to update Metallic.
        const ATOM_SIZE: u64 = 256; // Global safe upper bound
        let start = offset_bytes;
        let end = start + element_size as u64;

        let aligned_start = (start / ATOM_SIZE) * ATOM_SIZE;
        let aligned_end = end.div_ceil(ATOM_SIZE) * ATOM_SIZE;
        let aligned_size = aligned_end - aligned_start;

        self.allocator
            .vma
            .flush_allocation(&self.allocation, aligned_start, aligned_size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!(
                    "Failed to flush storage buffer at index {index}: {e}"
                ))
            })?;

        Ok(())
    }

    pub fn cleanup(&mut self) -> crate::Result<()> {
        if self.destroyed {
            return Ok(());
        }

        unsafe {
            // (Removed: device_wait_idle() serialized stall)
            self.allocator
                .vma
                .destroy_buffer(self.buffer, &mut self.allocation);
        }

        self.buffer = vk::Buffer::null();
        self.destroyed = true;
        Ok(())
    }

    /// Read all data from the buffer into a Vec
    ///
    /// # Safety
    /// Caller must ensure that the buffer is host-visible. GPU-side writes to this buffer must have completed before reading.
    pub unsafe fn read_all(&mut self) -> crate::Result<Vec<T>> {
        let size = (self.capacity * std::mem::size_of::<T>()) as u64;
        let guard = unsafe {
            self.allocator
                .map_allocation_guarded(&mut self.allocation, size)
        }?;
        Ok(unsafe { guard.as_slice::<T>().to_vec() })
    }

    /// Read a single element from the buffer at the specified index
    ///
    /// # Safety
    /// Buffer must be host-visible and index must be valid [0, capacity).
    pub unsafe fn read_element_at(&self, index: usize) -> T {
        // Buffer is allocated with MAPPED flag, so we can read directly
        let mapped_ptr = self
            .allocator
            .vma
            .get_allocation_info(&self.allocation)
            .mapped_data;

        assert!(
            !mapped_ptr.is_null(),
            "Buffer should be persistently mapped"
        );
        let base = mapped_ptr as *const T;
        unsafe { std::ptr::read(base.add(index)) }
    }

    /// Get current capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the GPU device address for BDA pulling
    pub fn device_address(&self) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.get_buffer_device_address(&info) }
    }
}

impl<T: Copy> Drop for StorageBuffer<T> {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

impl<T: Copy + Send + Sync + 'static> crate::renderer::cleanup_traits::VulkanResourceCleanup
    for StorageBuffer<T>
{
    fn cleanup_with_device(&mut self, _device: &ash::Device) -> std::result::Result<(), String> {
        self.cleanup().map_err(|e| e.to_string())
    }

    fn resource_type(&self) -> &'static str {
        "StorageBuffer"
    }
}

impl<T: Copy + Send + Sync + 'static> crate::renderer::resource_registry::VulkanResource
    for StorageBuffer<T>
{
}
