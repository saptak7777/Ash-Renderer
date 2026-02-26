use glam::{Mat3, Mat4, Quat, Vec3};
use std::sync::Arc;

/// Transform flags for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformFlags {
    pub has_uniform_scale: bool,
    pub has_rotation: bool,
    pub has_translation: bool,
    pub has_scale: bool,
}

impl Default for TransformFlags {
    fn default() -> Self {
        Self {
            has_uniform_scale: true,
            has_rotation: false,
            has_translation: false,
            has_scale: false,
        }
    }
}

/// AAA-quality Transform with lazy normal matrix calculation and fast paths.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    model: Mat4,
    normal: Option<Mat3>,
    flags: TransformFlags,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            model: Mat4::IDENTITY,
            normal: Some(Mat3::IDENTITY),
            flags: TransformFlags::default(),
        }
    }

    pub fn from_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        let model = Mat4::from_scale_rotation_translation(scale, rotation, translation);

        let has_uniform_scale =
            (scale.x - scale.y).abs() < f32::EPSILON && (scale.y - scale.z).abs() < f32::EPSILON;
        let has_rotation = rotation != Quat::IDENTITY;
        let has_translation = translation != Vec3::ZERO;
        let has_scale = (scale - Vec3::ONE).length_squared() > f32::EPSILON;

        Self {
            position: translation,
            rotation,
            scale,
            model,
            normal: None, // Lazy calculation
            flags: TransformFlags {
                has_uniform_scale,
                has_rotation,
                has_translation,
                has_scale,
            },
        }
    }

    #[inline]
    pub fn model_matrix(&self) -> Mat4 {
        self.model
    }

    /// Update from components and invalidate cache
    pub fn set_trs(&mut self, translation: Vec3, rotation: Quat, scale: Vec3) {
        self.position = translation;
        self.rotation = rotation;
        self.scale = scale;
        self.model = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        self.normal = None;
        self.flags.has_uniform_scale =
            (scale.x - scale.y).abs() < f32::EPSILON && (scale.y - scale.z).abs() < f32::EPSILON;
        self.flags.has_scale = (scale - Vec3::ONE).length_squared() > f32::EPSILON;
    }

    pub fn set_rotation(&mut self, euler: Vec3) {
        self.rotation = Quat::from_euler(glam::EulerRot::XYZ, euler.x, euler.y, euler.z);
        self.model =
            Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position);
        self.normal = None;
    }

    pub fn set_model(&mut self, model: Mat4) {
        self.model = model;
        self.normal = None;
        // Optimization: non-uniform scale estimation
        self.flags.has_uniform_scale = false;
    }

    /// Get or calculate normal matrix using Unreal-style optimization
    pub fn normal_matrix(&mut self) -> Mat3 {
        if let Some(normal) = self.normal {
            return normal;
        }

        let normal = if self.flags.has_uniform_scale || !self.flags.has_scale {
            // Fast path: Just extract and normalize rotation (no inverse needed)
            self.extract_rotation_matrix()
        } else {
            // Slow path: Non-uniform scale requires inverse-transpose
            self.calculate_normal_matrix_full()
        };

        self.normal = Some(normal);
        normal
    }

    fn extract_rotation_matrix(&self) -> Mat3 {
        Mat3::from_cols(
            self.model.x_axis.truncate().normalize(),
            self.model.y_axis.truncate().normalize(),
            self.model.z_axis.truncate().normalize(),
        )
    }

    fn calculate_normal_matrix_full(&self) -> Mat3 {
        let mat3 = Mat3::from_cols(
            self.model.x_axis.truncate(),
            self.model.y_axis.truncate(),
            self.model.z_axis.truncate(),
        );
        mat3.inverse().transpose()
    }
}

/// GPU-compatible data (Unreal-style 112 bytes)
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct GpuTransformData {
    pub model_matrix: [[f32; 4]; 4],  // 64 bytes
    pub normal_matrix: [[f32; 3]; 4], // 48 bytes (padded to 16-byte alignment)
}

// Ensure layout matches plan
const _: () = assert!(std::mem::size_of::<GpuTransformData>() == 112);

/// Opaque handle (prevents direct index manipulation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformHandle(pub(crate) usize);

/// Structure-of-Arrays layout (better cache utilization)
#[derive(Default)]
struct TransformStorage {
    model_matrices: Vec<Mat4>,
    normal_matrices: Vec<Mat3>,
    flags: Vec<TransformFlags>,
}

/// AAA-quality Transform Manager with automatic batching and parallel updates.
pub struct TransformSystem {
    storage: TransformStorage,
    dirty_flags: Vec<bool>,
    // GPU Resources
    pub(crate) arena_buffer: ash::vk::Buffer,
    pub(crate) arena_alloc: vk_mem::Allocation,
    pub(crate) arena_addr: u64,
    allocator: Arc<crate::vulkan::Allocator>,
}

// Default implementation removed since constructor now requires GPU resources

impl TransformSystem {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: Arc<crate::vulkan::Allocator>,
    ) -> crate::Result<Self> {
        let arena_size = 1024 * 1024; // 1MB arena
        let (arena_buffer, arena_alloc) = unsafe {
            allocator.create_buffer_with_flags_and_name(
                arena_size,
                ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
                Some("Transform Arena".to_string()),
            )?
        };

        let arena_addr = unsafe {
            device.get_buffer_device_address(
                &ash::vk::BufferDeviceAddressInfo::default().buffer(arena_buffer),
            )
        };

        Ok(Self {
            storage: TransformStorage::default(),
            dirty_flags: Vec::new(),
            arena_buffer,
            arena_alloc,
            arena_addr,
            allocator,
        })
    }

    /// Create transform (returns opaque handle)
    pub fn create(&mut self, translation: Vec3, rotation: Quat, scale: Vec3) -> TransformHandle {
        let transform = Transform::from_trs(translation, rotation, scale);
        let index = self.storage.model_matrices.len();

        self.storage.model_matrices.push(transform.model);
        self.storage.normal_matrices.push(Mat3::IDENTITY);
        self.storage.flags.push(transform.flags);
        self.dirty_flags.push(true);

        TransformHandle(index)
    }

    /// Update transform model matrix
    pub fn set_model(&mut self, handle: TransformHandle, model: Mat4) {
        if handle.0 < self.storage.model_matrices.len() {
            self.storage.model_matrices[handle.0] = model;
            self.dirty_flags[handle.0] = true;
            self.storage.flags[handle.0].has_uniform_scale = false;
        }
    }

    /// Parallel bulk update using rayon
    pub fn update(&mut self) {
        use rayon::prelude::*;

        let storage = &mut self.storage;
        let dirty = &mut self.dirty_flags;

        let model_matrices = &storage.model_matrices;
        let flags = &storage.flags;
        let normal_matrices = &mut storage.normal_matrices;

        normal_matrices
            .par_iter_mut()
            .zip(model_matrices.par_iter())
            .zip(flags.par_iter())
            .zip(dirty.par_iter_mut())
            .for_each(|(((norm, model), flag), is_dirty)| {
                if *is_dirty {
                    *norm = if flag.has_uniform_scale || !flag.has_scale {
                        Mat3::from_cols(
                            model.x_axis.truncate().normalize(),
                            model.y_axis.truncate().normalize(),
                            model.z_axis.truncate().normalize(),
                        )
                    } else {
                        let mat3 = Mat3::from_cols(
                            model.x_axis.truncate(),
                            model.y_axis.truncate(),
                            model.z_axis.truncate(),
                        );
                        mat3.inverse().transpose()
                    };
                    *is_dirty = false;
                }
            });
    }

    /// Get GPU transform data at index
    pub fn get_gpu_data(&self, handle: TransformHandle) -> GpuTransformData {
        let model = self.storage.model_matrices[handle.0];
        let normal = self.storage.normal_matrices[handle.0];

        let normal_cols = normal.to_cols_array_2d();
        let normal_padded = [
            normal_cols[0],
            normal_cols[1],
            normal_cols[2],
            [0.0, 0.0, 0.0], // Padding for 16-byte alignment of each column
        ];

        GpuTransformData {
            model_matrix: model.to_cols_array_2d(),
            normal_matrix: normal_padded,
        }
    }

    /// Synchronize CPU data to the GPU arena buffer.
    pub fn update_buffers(&mut self) -> crate::Result<()> {
        let count = self.storage.model_matrices.len();
        if count == 0 {
            return Ok(());
        }

        let mut gpu_data = Vec::with_capacity(count);
        for i in 0..count {
            gpu_data.push(self.get_gpu_data(TransformHandle(i)));
        }

        let byte_size = (count * std::mem::size_of::<GpuTransformData>()) as u64;
        unsafe {
            let mut map = self
                .allocator
                .map_allocation_guarded(&mut self.arena_alloc, byte_size)?;
            map.copy_from_slice(&gpu_data);
        }

        // Lead Engineer Fix: Explicit flush for non-coherent host memory
        // Align flush to 256 bytes (common nonCoherentAtomSize)
        let aligned_size = byte_size.div_ceil(256) * 256;
        self.allocator
            .vma
            .flush_allocation(&self.arena_alloc, 0, aligned_size)
            .map_err(|e| {
                crate::AshError::VulkanError(format!("Failed to flush transform buffer: {e}"))
            })?;

        Ok(())
    }
}

/// MVP matrices for rendering
#[derive(Debug, Clone, Copy)]
pub struct MVP {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
}

impl MVP {
    pub fn new(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        Self {
            model,
            view,
            projection,
        }
    }

    pub fn combined(&self) -> Mat4 {
        self.projection * self.view * self.model
    }
}

/// Simple perspective camera
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn default(aspect: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov: 45.0,
            aspect,
            near: 0.5,
            far: 100.0,
        }
    }

    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        Self {
            position,
            target,
            up: Vec3::Y,
            fov: 45.0,
            aspect,
            near: 0.5,
            far: 100.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        // Reverse-Z Perspective: swap near and far planes
        let mut proj =
            Mat4::perspective_rh(self.fov.to_radians(), self.aspect, self.far, self.near);
        proj.y_axis.y *= -1.0;
        proj
    }
}

/// Temporal camera for TSR with automatic frame history tracking
///
/// Maintains current and previous frame matrices for motion vector generation.
/// Uses Halton sequence for sub-pixel jitter to improve temporal stability.
pub struct TemporalCamera {
    // Camera parameters
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub width: u32,
    pub height: u32,
    pub near: f32,
    pub far: f32,

    // Current frame matrices
    view: Mat4,
    proj: Mat4,
    view_proj: Mat4,

    // Previous frame matrices (for motion vectors)
    prev_view: Mat4,
    prev_proj: Mat4,
    prev_view_proj: Mat4,

    // Jitter state
    halton: crate::renderer::util::halton::HaltonSequence,
    current_jitter: glam::Vec2,
}

impl TemporalCamera {
    pub fn new(position: Vec3, target: Vec3, width: u32, height: u32) -> Self {
        let aspect = width as f32 / height.max(1) as f32;
        let view = Mat4::look_at_rh(position, target, Vec3::Y);
        let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 100.0, 0.5);
        proj.y_axis.y *= -1.0;
        let view_proj = proj * view;

        Self {
            position,
            target,
            up: Vec3::Y,
            fov: 45.0,
            width,
            height,
            near: 0.5,
            far: 100.0,
            view,
            proj,
            view_proj,
            prev_view: view,
            prev_proj: proj,
            prev_view_proj: view_proj,
            halton: crate::renderer::util::halton::HaltonSequence::new(2, 3),
            current_jitter: glam::Vec2::ZERO,
        }
    }

    pub fn default(width: u32, height: u32) -> Self {
        Self::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, width, height)
    }

    /// Begin new frame - swaps previous/current matrices and updates jitter
    ///
    /// # Parameters
    /// - `width`/`height` – Current render target dimensions.
    pub fn begin_frame(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        // Zero-cost swap using Rust's ownership system
        std::mem::swap(&mut self.prev_view, &mut self.view);
        std::mem::swap(&mut self.prev_proj, &mut self.proj);
        std::mem::swap(&mut self.prev_view_proj, &mut self.view_proj);

        // Update jitter for this frame
        self.current_jitter = self.halton.next_sample();

        // Recalculate matrices with new jitter
        self.update_matrices();
    }

    fn update_matrices(&mut self) {
        self.view = Mat4::look_at_rh(self.position, self.target, self.up);

        let aspect = self.width as f32 / self.height.max(1) as f32;
        let mut proj = Mat4::perspective_rh(self.fov.to_radians(), aspect, self.far, self.near);
        proj.y_axis.y *= -1.0; // Vulkan Y-flip

        // Apply sub-pixel jitter normalized to resolution (matches FrameState logic)
        let mut jittered = proj;
        let jitter_ndc_x = (self.current_jitter.x * 2.0) / self.width.max(1) as f32;
        let jitter_ndc_y = (self.current_jitter.y * 2.0) / self.height.max(1) as f32;
        *jittered.col_mut(2) = proj.col(2) + glam::Vec4::new(jitter_ndc_x, jitter_ndc_y, 0.0, 0.0);
        self.proj = jittered;
        self.view_proj = self.proj * self.view;
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.view
    }

    pub fn projection_matrix(&self) -> Mat4 {
        self.proj
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.view_proj
    }

    pub fn prev_view_proj_matrix(&self) -> Mat4 {
        self.prev_view_proj
    }

    pub fn get_motion_data(
        &self,
        model: Mat4,
        vertex_heap_ptr: u64,
    ) -> crate::renderer::resources::motion::ObjectMotionData {
        crate::renderer::resources::motion::ObjectMotionData::new(
            self.view_proj * model,
            self.prev_view_proj * model,
            vertex_heap_ptr,
        )
    }

    pub fn jitter(&self) -> glam::Vec2 {
        self.current_jitter
    }
}

impl Drop for TransformSystem {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.arena_buffer, &mut self.arena_alloc);
        }
    }
}
