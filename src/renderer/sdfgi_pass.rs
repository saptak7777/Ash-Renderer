//! Signed Distance Field Global Illumination (SDFGI)
//!
//! Implements cascaded voxel-based global illumination inspired by Godot 4.
//! Key features:
//! - 8 camera-relative cascades with power-of-two world coverage
//! - Real-time geometry voxelization using geometry shaders
//! - Cone tracing for diffuse and specular indirect lighting
//! - Temporal accumulation for stable results

use ash::vk;
use glam::{Mat4, Vec3};
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// Number of voxel cascades
pub const CASCADE_COUNT: usize = 8;

/// Voxel resolution per cascade
pub const VOXEL_RESOLUTION: u32 = 32;

/// SDFGI quality presets
#[derive(Clone, Copy, Debug, Default)]
pub enum SdfgiQuality {
    /// 4 cones, minimal GI
    Low,
    /// 6 cones, balanced
    #[default]
    Medium,
    /// 9 cones, quality
    High,
}

impl SdfgiQuality {
    /// Get cone count for diffuse tracing
    pub fn cone_count(&self) -> u32 {
        match self {
            SdfgiQuality::Low => 4,
            SdfgiQuality::Medium => 6,
            SdfgiQuality::High => 9,
        }
    }
}

/// Single voxel cascade
struct VoxelCascade {
    /// Voxel grid resolution (typically 32³)
    resolution: u32,
    /// World-space size covered by this cascade
    world_size: f32,
    /// Snapped origin in world space
    origin: Vec3,

    // 3D Textures
    albedo_img: vk::Image,
    albedo_alloc: Option<vk_mem::Allocation>,
    albedo_view: vk::ImageView,

    normal_img: vk::Image,
    normal_alloc: Option<vk_mem::Allocation>,
    normal_view: vk::ImageView,
}

impl VoxelCascade {
    fn new() -> Self {
        Self {
            resolution: VOXEL_RESOLUTION,
            world_size: 0.0,
            origin: Vec3::ZERO,
            albedo_img: vk::Image::null(),
            albedo_alloc: None,
            albedo_view: vk::ImageView::null(),
            normal_img: vk::Image::null(),
            normal_alloc: None,
            normal_view: vk::ImageView::null(),
        }
    }

    /// # Safety
    /// Device and allocator must stay valid.
    unsafe fn init(
        &mut self,
        device: &ash::Device,
        alloc: &vk_mem::Allocator,
        cascade_index: usize,
    ) -> Result<()> {
        use vk_mem::Alloc;

        // World size doubles for each cascade: 8m, 16m, 32m...
        self.world_size = 8.0 * (1 << cascade_index) as f32;

        // Create albedo 3D texture (R16G16B16A16_SFLOAT for HDR)
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_3D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: self.resolution,
                height: self.resolution,
                depth: self.resolution,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = alloc.create_image(&image_info, &alloc_info).map_err(|e| {
            crate::AshError::VulkanError(format!("SDFGI cascade {cascade_index} albedo: {e:?}"))
        })?;

        self.albedo_img = image;
        self.albedo_alloc = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.albedo_img)
            .view_type(vk::ImageViewType::TYPE_3D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.albedo_view = device.create_image_view(&view_info, None)?;

        // Create normal 3D texture
        let (image, allocation) = alloc.create_image(&image_info, &alloc_info).map_err(|e| {
            crate::AshError::VulkanError(format!("SDFGI cascade {cascade_index} normal: {e:?}"))
        })?;

        self.normal_img = image;
        self.normal_alloc = Some(allocation);

        let view_info = view_info.image(self.normal_img);
        self.normal_view = device.create_image_view(&view_info, None)?;

        Ok(())
    }

    /// Update cascade origin based on camera position (snapped to voxel grid)
    fn update_origin(&mut self, camera_pos: Vec3) {
        let voxel_size = self.world_size / self.resolution as f32;
        // Snap to voxel grid to minimize popping
        self.origin = (camera_pos / voxel_size).floor() * voxel_size;
    }

    /// # Safety
    /// Resources must not be in use.
    unsafe fn destroy(&mut self, device: &ash::Device, allocator: &vk_mem::Allocator) {
        if self.albedo_view != vk::ImageView::null() {
            device.destroy_image_view(self.albedo_view, None);
        }
        if let Some(mut a) = self.albedo_alloc.take() {
            allocator.destroy_image(self.albedo_img, &mut a);
        }

        if self.normal_view != vk::ImageView::null() {
            device.destroy_image_view(self.normal_view, None);
        }
        if let Some(mut a) = self.normal_alloc.take() {
            allocator.destroy_image(self.normal_img, &mut a);
        }
    }
}

/// Push constants for voxelization shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelizePushConstants {
    /// View-projection matrix for voxelization
    pub view_proj: [[f32; 4]; 4],
    /// Cascade origin in world space
    pub cascade_origin: [f32; 3],
    /// Voxel size (world_size / resolution)
    pub voxel_size: f32,
    /// Cascade index
    pub cascade_index: u32,
    /// Padding
    pub _padding: [u32; 3],
}

/// Push constants for cone tracing shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConeTracePushConstants {
    /// Inverse view-projection for ray reconstruction
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera position
    pub camera_pos: [f32; 3],
    /// GI intensity
    pub intensity: f32,
    /// Screen dimensions
    pub screen_size: [f32; 2],
    /// Cone count
    pub cone_count: u32,
    /// Frame index for temporal jitter
    pub frame_index: u32,
}

/// SDFGI pass
pub struct SdfgiPass {
    device: Arc<ash::Device>,

    // Cascades
    cascades: Vec<VoxelCascade>,

    // GI output buffer
    gi_img: vk::Image,
    gi_alloc: Option<vk_mem::Allocation>,
    gi_view: vk::ImageView,

    // Pipelines
    voxelize_pipeline: vk::Pipeline,
    voxelize_layout: vk::PipelineLayout,

    cone_trace_pipeline: vk::Pipeline,
    cone_trace_layout: vk::PipelineLayout,

    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    voxel_desc_layout: vk::DescriptorSetLayout,
    voxel_desc_sets: Vec<vk::DescriptorSet>,

    trace_desc_layout: vk::DescriptorSetLayout,
    trace_desc_set: vk::DescriptorSet,

    sampler: vk::Sampler,

    // State
    width: u32,
    height: u32,
    quality: SdfgiQuality,
    intensity: f32,
    frame_index: u32,

    initialized: bool,
}

impl SdfgiPass {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            cascades: Vec::new(),
            gi_img: vk::Image::null(),
            gi_alloc: None,
            gi_view: vk::ImageView::null(),
            voxelize_pipeline: vk::Pipeline::null(),
            voxelize_layout: vk::PipelineLayout::null(),
            cone_trace_pipeline: vk::Pipeline::null(),
            cone_trace_layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            voxel_desc_layout: vk::DescriptorSetLayout::null(),
            voxel_desc_sets: Vec::new(),
            trace_desc_layout: vk::DescriptorSetLayout::null(),
            trace_desc_set: vk::DescriptorSet::null(),
            sampler: vk::Sampler::null(),
            width: 0,
            height: 0,
            quality: SdfgiQuality::default(),
            intensity: 1.0,
            frame_index: 0,
            initialized: false,
        }
    }

    /// # Safety
    /// Device and allocator must stay valid for the lifetime of this pass.
    pub unsafe fn init(
        &mut self,
        alloc: &vk_mem::Allocator,
        _vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
        quality: SdfgiQuality,
    ) {
        if self.initialized {
            return;
        }

        self.width = width;
        self.height = height;
        self.quality = quality;

        // Initialize cascades
        self.cascades = (0..CASCADE_COUNT).map(|_| VoxelCascade::new()).collect();
        for (i, cascade) in self.cascades.iter_mut().enumerate() {
            cascade
                .init(&self.device, alloc, i)
                .expect("SDFGI: Cascade initialization failed");
        }

        self.create_gi_image(alloc)
            .expect("SDFGI: GI image allocation failed");
        self.create_sampler()
            .expect("SDFGI: Sampler creation failed");
        self.create_descriptors()
            .expect("SDFGI: Descriptor creation failed");
        self.create_pipelines()
            .expect("SDFGI: Pipeline creation failed");

        self.initialized = true;
        log::info!("SDFGI: Initialized with {CASCADE_COUNT} cascades");
    }

    /// # Safety
    /// This function creates Vulkan resources.
    unsafe fn create_gi_image(&mut self, alloc: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = alloc
            .create_image(&image_info, &alloc_info)
            .map_err(|e| crate::AshError::VulkanError(format!("SDFGI GI image: {e:?}")))?;

        self.gi_img = image;
        self.gi_alloc = Some(allocation);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(self.gi_img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.gi_view = self.device.create_image_view(&view_info, None)?;

        Ok(())
    }

    unsafe fn create_sampler(&mut self) -> Result<()> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

        self.sampler = self.device.create_sampler(&info, None)?;
        Ok(())
    }

    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // Voxel descriptor layout (for voxelization)
        // Binding 0: Albedo 3D storage image
        // Binding 1: Normal 3D storage image
        let voxel_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&voxel_bindings);
        self.voxel_desc_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        // Trace descriptor layout (for cone tracing)
        // Bindings 0-7: Cascade albedo textures
        // Bindings 8-15: Cascade normal textures
        // Binding 16: G-Buffer depth
        // Binding 17: G-Buffer normal
        // Binding 18: G-Buffer albedo
        // Binding 19: Output GI
        let mut trace_bindings = Vec::new();
        for i in 0..CASCADE_COUNT {
            trace_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        for i in 0..CASCADE_COUNT {
            trace_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding((CASCADE_COUNT + i) as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        // G-Buffer inputs
        for i in 0..3 {
            trace_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding((CASCADE_COUNT * 2 + i) as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        // Output
        trace_bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding((CASCADE_COUNT * 2 + 3) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&trace_bindings);
        self.trace_desc_layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: (CASCADE_COUNT * 2 + 1) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: (CASCADE_COUNT * 2 + 3) as u32,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets((CASCADE_COUNT + 1) as u32)
            .pool_sizes(&pool_sizes);

        self.descriptor_pool = self.device.create_descriptor_pool(&pool_info, None)?;

        // Allocate voxel descriptor sets (one per cascade)
        let voxel_layouts: Vec<_> = (0..CASCADE_COUNT).map(|_| self.voxel_desc_layout).collect();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&voxel_layouts);

        self.voxel_desc_sets = self.device.allocate_descriptor_sets(&alloc_info)?;

        // Allocate trace descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.trace_desc_layout));

        let sets = self.device.allocate_descriptor_sets(&alloc_info)?;
        self.trace_desc_set = sets[0];

        // Update voxel descriptors
        for (i, cascade) in self.cascades.iter().enumerate() {
            let albedo_info = vk::DescriptorImageInfo::default()
                .image_view(cascade.albedo_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let normal_info = vk::DescriptorImageInfo::default()
                .image_view(cascade.normal_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.voxel_desc_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&albedo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.voxel_desc_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&normal_info)),
            ];

            self.device.update_descriptor_sets(&writes, &[]);
        }

        Ok(())
    }

    /// Create SDFGI pipelines
    unsafe fn create_pipelines(&mut self) -> Result<()> {
        // Voxelization pipeline (geometry shader based)
        // NOTE: Actual shader compilation would happen here
        // For now, we'll create placeholder pipelines

        // Cone tracing pipeline
        {
            // NOTE: This would load the actual compiled shader
            // let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/cone_trace.comp.spv"));

            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<ConeTracePushConstants>() as u32);

            let layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&self.trace_desc_layout))
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));

            self.cone_trace_layout = self.device.create_pipeline_layout(&layout_info, None)?;

            log::info!("SDFGI: Cone trace pipeline layout created");
        }

        Ok(())
    }

    /// Update cascade origins based on camera position
    pub fn update_cascades(&mut self, camera_pos: Vec3) {
        for cascade in &mut self.cascades {
            cascade.update_origin(camera_pos);
        }
    }

    /// Get GI output view
    pub fn gi_view(&self) -> vk::ImageView {
        if !self.initialized {
            return vk::ImageView::null();
        }
        self.gi_view
    }

    /// Set GI intensity
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity.max(0.0);
    }

    /// Get GI intensity
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }

    /// Voxelize scene geometry into cascades
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn voxelize_scene(
        &mut self,
        _cmd: vk::CommandBuffer,
        _cascade_index: usize,
    ) -> Result<()> {
        if !self.initialized || self.voxelize_pipeline == vk::Pipeline::null() {
            return Ok(());
        }

        // NOTE: Actual voxelization would happen here
        // This requires:
        // 1. Binding voxelize pipeline
        // 2. Setting up viewport for orthographic projection
        // 3. Binding cascade descriptor set
        // 4. Drawing scene meshes with push constants
        // 5. Image barriers for voxel textures

        // Placeholder for now
        log::debug!("SDFGI: Voxelization pass (placeholder)");

        Ok(())
    }

    /// Trace cones through voxel cascades
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn trace_cones(
        &mut self,
        cmd: vk::CommandBuffer,
        depth_view: vk::ImageView,
        normal_view: vk::ImageView,
        albedo_view: vk::ImageView,
        inv_view_proj: Mat4,
        camera_pos: Vec3,
    ) -> Result<()> {
        if !self.initialized || self.cone_trace_pipeline == vk::Pipeline::null() {
            return Ok(());
        }

        // Update trace descriptors with G-Buffer inputs
        let sampler_info = vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // Store all image infos to keep them alive
        let mut image_infos = Vec::new();

        // Cascade samplers
        for cascade in &self.cascades {
            image_infos.push(
                sampler_info
                    .image_view(cascade.albedo_view)
                    .image_layout(vk::ImageLayout::GENERAL),
            );
            image_infos.push(
                sampler_info
                    .image_view(cascade.normal_view)
                    .image_layout(vk::ImageLayout::GENERAL),
            );
        }

        // G-Buffer inputs
        image_infos.push(sampler_info.image_view(depth_view));
        image_infos.push(sampler_info.image_view(normal_view));
        image_infos.push(sampler_info.image_view(albedo_view));

        // Output
        image_infos.push(
            vk::DescriptorImageInfo::default()
                .image_view(self.gi_view)
                .image_layout(vk::ImageLayout::GENERAL),
        );

        // Build write descriptors
        let mut writes = Vec::new();
        let mut info_idx = 0;

        // Cascade albedo textures
        for i in 0..CASCADE_COUNT {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.trace_desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_infos[info_idx])),
            );
            info_idx += 1;

            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.trace_desc_set)
                    .dst_binding((CASCADE_COUNT + i) as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&image_infos[info_idx])),
            );
            info_idx += 1;
        }

        // G-Buffer inputs
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.trace_desc_set)
                .dst_binding((CASCADE_COUNT * 2) as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_infos[info_idx])),
        );
        info_idx += 1;

        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.trace_desc_set)
                .dst_binding((CASCADE_COUNT * 2 + 1) as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_infos[info_idx])),
        );
        info_idx += 1;

        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.trace_desc_set)
                .dst_binding((CASCADE_COUNT * 2 + 2) as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_infos[info_idx])),
        );
        info_idx += 1;

        // Output
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.trace_desc_set)
                .dst_binding((CASCADE_COUNT * 2 + 3) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_infos[info_idx])),
        );

        self.device.update_descriptor_sets(&writes, &[]);

        // Transition GI image to GENERAL layout
        let barrier = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(self.gi_img)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );

        // Bind pipeline and dispatch (if pipeline exists)
        if self.cone_trace_pipeline != vk::Pipeline::null() {
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cone_trace_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.cone_trace_layout,
                0,
                std::slice::from_ref(&self.trace_desc_set),
                &[],
            );

            let push_constants = ConeTracePushConstants {
                inv_view_proj: inv_view_proj.to_cols_array_2d(),
                camera_pos: camera_pos.to_array(),
                intensity: self.intensity,
                screen_size: [self.width as f32, self.height as f32],
                cone_count: self.quality.cone_count(),
                frame_index: self.frame_index,
            };

            self.device.cmd_push_constants(
                cmd,
                self.cone_trace_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let group_x = self.width.div_ceil(8);
            let group_y = self.height.div_ceil(8);
            self.device.cmd_dispatch(cmd, group_x, group_y, 1);
        }

        Ok(())
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        for cascade in &mut self.cascades {
            cascade.destroy(&self.device, allocator);
        }

        if self.gi_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.gi_view, None);
        }
        if let Some(mut a) = self.gi_alloc.take() {
            allocator.destroy_image(self.gi_img, &mut a);
        }

        if self.sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.sampler, None);
        }

        if self.voxelize_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.voxelize_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.voxelize_layout, None);
        }

        if self.cone_trace_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.cone_trace_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.cone_trace_layout, None);
        }

        if self.descriptor_pool != vk::DescriptorPool::null() {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.voxel_desc_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.trace_desc_layout, None);
        }

        self.initialized = false;
    }
}
