//! Hi-Z Pyramid GPU Resources
//!
//! Manages GPU-side resources for hierarchical-Z occlusion culling:
//! - Hi-Z pyramid image (mip chain)
//! - Compute pipelines for pyramid generation and culling
//! - Descriptor sets and buffers

use ash::vk;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// Hi-Z pyramid mip levels (1024 → 1)
pub const HIZ_MIP_LEVELS: u32 = 10;

/// Push constants for Hi-Z generation
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HiZGeneratePushConstants {
    pub output_size: [u32; 2],
    pub mip_level: u32,
    pub _padding: u32,
}

/// GPU resources for Hi-Z pyramid
pub struct HiZPass {
    device: Arc<ash::Device>,

    // Hi-Z pyramid image (R32_SFLOAT, mip chain)
    hiz_image: vk::Image,
    hiz_allocation: Option<vk_mem::Allocation>,
    hiz_views: Vec<vk::ImageView>,
    hiz_sampler: vk::Sampler,

    // Compute resources
    generate_pipeline: vk::Pipeline,
    generate_layout: vk::PipelineLayout,

    // Descriptors
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Geometry
    width: u32,
    height: u32,
    mip_count: u32,

    initialized: bool,
}

impl HiZPass {
    /// Create a new Hi-Z pass (uninitialized)
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            hiz_image: vk::Image::null(),
            hiz_allocation: None,
            hiz_views: Vec::new(),
            hiz_sampler: vk::Sampler::null(),
            generate_pipeline: vk::Pipeline::null(),
            generate_layout: vk::PipelineLayout::null(),
            pool: vk::DescriptorPool::null(),
            layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
            width: 0,
            height: 0,
            mip_count: 0,
            initialized: false,
        }
    }

    pub unsafe fn init(
        &mut self,
        allocator: &vk_mem::Allocator,
        vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
    ) {
        // Adversarial Defense: Zero-Sized Resource
        // Minimizing a window on Windows often causes width/height to become 0.
        // Creating Vulkan images with 0 dimensions is invalid and will crash.
        if width == 0 || height == 0 {
            log::warn!(
                "HiZPass: Skipping initialization with zero dimensions (window likely minimized)"
            );
            return;
        }

        if self.initialized {
            return;
        }

        self.width = width;
        self.height = height;
        self.mip_count = Self::calculate_mip_count(width, height);

        // Hi-Z image with mip chain
        self.create_hiz_image(allocator)
            .expect("Hi-Z image allocation failed");

        self.create_sampler().expect("Hi-Z sampler creation failed");
        self.create_descriptors()
            .expect("Hi-Z descriptor setup failed");
        self.create_pipeline(vulkan_device)
            .expect("Hi-Z pipeline creation failed");

        self.initialized = true;
    }

    /// Calculate required mip levels
    fn calculate_mip_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height);
        (32 - max_dim.leading_zeros()).min(HIZ_MIP_LEVELS)
    }

    /// Create Hi-Z image with mip chain
    unsafe fn create_hiz_image(&mut self, allocator: &vk_mem::Allocator) -> Result<()> {
        use vk_mem::Alloc;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            })
            .mip_levels(self.mip_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) =
            allocator
                .create_image(&image_info, &alloc_info)
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Hi-Z image creation failed: {e:?}"))
                })?;

        self.hiz_image = image;
        self.hiz_allocation = Some(allocation);

        // Create views for each mip level
        for mip in 0..self.mip_count {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(self.hiz_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                );

            let view = self.device.create_image_view(&view_info, None)?;
            self.hiz_views.push(view);
        }

        log::debug!("HiZPass: Created image with {} views", self.hiz_views.len());
        Ok(())
    }

    /// Create sampler for Hi-Z reads
    unsafe fn create_sampler(&mut self) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(self.mip_count as f32);

        self.hiz_sampler = self.device.create_sampler(&sampler_info, None)?;
        Ok(())
    }

    /// Create descriptor layout and pool
    unsafe fn create_descriptors(&mut self) -> Result<()> {
        // Layout: binding 0 = input sampler, binding 1 = output storage image
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        self.layout = self
            .device
            .create_descriptor_set_layout(&layout_info, None)?;

        // Pool for mip_count - 1 sets (one per mip transition)
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: self.mip_count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: self.mip_count,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.mip_count)
            .pool_sizes(&pool_sizes);

        self.pool = self.device.create_descriptor_pool(&pool_info, None)?;

        // Allocate sets
        let layouts: Vec<_> = (0..self.mip_count).map(|_| self.layout).collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);

        self.descriptor_sets = self.device.allocate_descriptor_sets(&alloc_info)?;

        // Update descriptor sets for each mip transition
        for mip in 0..(self.mip_count as usize - 1) {
            let src_view = self.hiz_views[mip];
            let dst_view = self.hiz_views[mip + 1];

            let sampler_info = vk::DescriptorImageInfo::default()
                .sampler(self.hiz_sampler)
                .image_view(src_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let storage_info = vk::DescriptorImageInfo::default()
                .image_view(dst_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[mip])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&sampler_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[mip])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&storage_info)),
            ];

            self.device.update_descriptor_sets(&writes, &[]);
        }

        Ok(())
    }

    /// Create compute pipeline
    unsafe fn create_pipeline(&mut self, _vulkan_device: &VulkanDevice) -> Result<()> {
        // Load shader module
        let shader_path = std::path::Path::new("shaders/hiz_generate.spv");
        let shader_code = std::fs::read(shader_path)?;

        let shader_module_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&shader_code));

        let shader_module = self
            .device
            .create_shader_module(&shader_module_info, None)?;

        // Push constant range
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<HiZGeneratePushConstants>() as u32);

        // Pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        self.generate_layout = self.device.create_pipeline_layout(&layout_info, None)?;

        // Compute pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.generate_layout);

        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| e)?;

        self.generate_pipeline = pipelines[0];
        self.device.destroy_shader_module(shader_module, None);

        Ok(())
    }

    /// Build Hi-Z pyramid from depth buffer
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn build_pyramid(
        &self,
        cmd: vk::CommandBuffer,
        depth_image: vk::Image,
    ) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        // Depth -> Source for transfer
        let depth_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_READ,
            old_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: depth_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[depth_barrier],
        );

        // Hi-Z mip 0 -> Destination for transfer
        let hiz_barrier = vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: self.hiz_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[hiz_barrier],
        );

        let blit_region = vk::ImageBlit::default()
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .layer_count(1),
            )
            .src_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: self.width as i32,
                    y: self.height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: self.width as i32,
                    y: self.height as i32,
                    z: 1,
                },
            ]);

        self.device.cmd_blit_image(
            cmd,
            depth_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.hiz_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit_region],
            vk::Filter::NEAREST,
        );

        // Transition mip 0 to shader read
        let mip0_read_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(self.hiz_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[mip0_read_barrier],
        );

        // Generate mip chain
        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.generate_pipeline);

        for mip in 1..self.mip_count {
            let mip_width = (self.width >> mip).max(1);
            let mip_height = (self.height >> mip).max(1);

            // Transition current mip to general (storage write)
            let mip_barrier = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.hiz_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[mip_barrier],
            );

            // Bind descriptor set for this mip transition
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.generate_layout,
                0,
                &[self.descriptor_sets[(mip - 1) as usize]],
                &[],
            );

            // Push constants
            let push = HiZGeneratePushConstants {
                output_size: [mip_width, mip_height],
                mip_level: mip,
                _padding: 0,
            };

            self.device.cmd_push_constants(
                cmd,
                self.generate_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );

            // Dispatch
            let group_x = mip_width.div_ceil(8);
            let group_y = mip_height.div_ceil(8);
            self.device.cmd_dispatch(cmd, group_x, group_y, 1);

            // Transition this mip to shader read for next iteration
            let read_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.hiz_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[read_barrier],
            );
        }

        // Restore depth to attachment optimal
        let depth_restore = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(depth_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            );

        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[depth_restore],
        );

        Ok(())
    }

    /// Get Hi-Z image for culling shader
    pub fn hiz_image(&self) -> vk::Image {
        self.hiz_image
    }

    /// Get Hi-Z sampler for culling shader
    pub fn hiz_sampler(&self) -> vk::Sampler {
        self.hiz_sampler
    }

    /// Get complete Hi-Z image view (all mips)
    pub fn hiz_view(&self) -> Option<vk::ImageView> {
        self.hiz_views.first().copied()
    }

    /// Resize the Hi-Z pyramid
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn resize(
        &mut self,
        allocator: &vk_mem::Allocator,
        vulkan_device: &VulkanDevice,
        width: u32,
        height: u32,
    ) {
        // Adversarial Defense: Guard against zero dimensions on resize.
        if width == 0 || height == 0 {
            return;
        }

        if width == self.width && height == self.height {
            return;
        }

        self.destroy(allocator);
        self.init(allocator, vulkan_device, width, height);
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use.
    pub unsafe fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        if !self.initialized {
            return;
        }

        for view in self.hiz_views.drain(..) {
            self.device.destroy_image_view(view, None);
        }

        if self.hiz_image != vk::Image::null() {
            if let Some(mut alloc) = self.hiz_allocation.take() {
                allocator.destroy_image(self.hiz_image, &mut alloc);
            }
            self.hiz_image = vk::Image::null();
        }

        if self.hiz_sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.hiz_sampler, None);
            self.hiz_sampler = vk::Sampler::null();
        }

        if self.generate_pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.generate_pipeline, None);
            self.generate_pipeline = vk::Pipeline::null();
        }

        if self.generate_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.generate_layout, None);
            self.generate_layout = vk::PipelineLayout::null();
        }

        if self.pool != vk::DescriptorPool::null() {
            self.device.destroy_descriptor_pool(self.pool, None);
            self.pool = vk::DescriptorPool::null();
        }

        if self.layout != vk::DescriptorSetLayout::null() {
            self.device.destroy_descriptor_set_layout(self.layout, None);
            self.layout = vk::DescriptorSetLayout::null();
        }

        self.initialized = false;
        log::info!("HiZPass: Resources destroyed");
    }
}
