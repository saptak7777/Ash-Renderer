//! VSM GPU Resources - Physical Cache, Page Table, and Buffers

use ash::vk;
use std::sync::Arc;

use super::buffer::VsmRequestBuffer;
use super::data::VsmGlobalInfo;
use crate::vulkan::Allocator;
use crate::{AshError, Result};

/// VSM Configuration
#[derive(Debug, Clone)]
pub struct VsmConfig {
    /// Virtual shadow map resolution (e.g., 16384 for 16k x 16k)
    pub virtual_resolution: u32,
    /// Physical cache resolution (actual GPU memory, e.g., 4096)
    pub physical_resolution: u32,
    /// Page size in pixels (typically 128)
    pub page_size: u32,
    /// Maximum number of page requests per frame
    // NOTE: This value is passed to analyze.glsl via Specialization Constants as the primary sync mechanism.
    pub max_requests_per_frame: u32,
    /// Enable debug visualization
    pub debug_mode: bool,
    /// Number of clipmap levels for directional lights (0 = disabled, typical: 8)
    pub clipmap_levels: u32,
    /// World-space extent of clipmap level 0 (in meters, e.g., 100.0)
    pub clipmap_base_extent: f32,
    /// Clipmap radius for calculation (often half of extent)
    pub clipmap_radius: f32,
}

impl Default for VsmConfig {
    fn default() -> Self {
        Self {
            virtual_resolution: 16384,
            physical_resolution: 4096,
            page_size: 128,
            max_requests_per_frame: 1024,
            debug_mode: false,
            clipmap_levels: 8,
            clipmap_base_extent: 100.0,
            clipmap_radius: 50.0,
        }
    }
}

impl VsmConfig {
    /// Calculate number of pages in virtual space
    pub fn virtual_page_count(&self) -> u32 {
        let pages_per_axis = self.virtual_resolution / self.page_size;
        pages_per_axis * pages_per_axis
    }

    /// Calculate number of pages in physical cache
    pub fn physical_page_count(&self) -> u32 {
        let pages_per_axis = self.physical_resolution / self.page_size;
        pages_per_axis * pages_per_axis
    }

    /// Calculate page table resolution (one texel per virtual page)
    pub fn page_table_resolution(&self) -> u32 {
        self.virtual_resolution / self.page_size
    }
}

/// GPU-ready VSM metadata
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmMetadata {
    /// Virtual resolution (width/height)
    pub virtual_resolution: u32,
    /// Physical resolution (width/height)
    pub physical_resolution: u32,
    /// Page size in pixels
    pub page_size: u32,
    /// Page table resolution
    pub page_table_resolution: u32,
    /// Number of physical pages available
    pub physical_page_count: u32,
    /// Current frame index (for LRU)
    pub frame_index: u32,
    /// Debug flags
    pub debug_flags: u32,
    /// Padding
    pub _padding: u32,
}

/// Page request entry (written by analysis shader)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PageRequest {
    /// Virtual page X coordinate
    pub virtual_x: u32,
    /// Virtual page Y coordinate
    pub virtual_y: u32,
    /// Request priority (distance from camera)
    pub priority: f32,
    /// Layer index (Clipmap Level)
    pub layer: u32,
}

/// Physical page allocation (written by allocator)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PageAllocation {
    /// Virtual page X
    pub virtual_x: u32,
    /// Virtual page Y
    pub virtual_y: u32,
    /// Physical page X
    pub physical_x: u32,
    /// Physical page Y
    pub physical_y: u32,
    /// Layer index (Clipmap Level) - Added for Texture2DArray support
    pub layer: u32,
    /// Flags (Bit 0: Dirty/Update Required)
    pub flags: u32,
    /// Padding to align to 16 bytes
    pub _padding: [u32; 2],
}

/// Safe padding for alignment in atomic buffers (Count + Padding)
use super::buffer::ATOMIC_HEADER_SIZE;
const _: () = assert!(ATOMIC_HEADER_SIZE == 16);
const _: () = assert!(ATOMIC_HEADER_SIZE == std::mem::size_of::<[u32; 4]>() as u64);

/// VSM GPU Resources
pub struct VsmResources {
    pub device: Arc<ash::Device>,
    pub allocator: Arc<Allocator>,
    pub config: VsmConfig,

    /// Physical cache texture (R32G32_FLOAT for variance moments: depth, depth^2)
    pub physical_cache: vk::Image,
    physical_cache_alloc: Option<vk_mem::Allocation>,
    pub physical_cache_view: vk::ImageView,
    pub physical_cache_sampler: vk::Sampler,

    /// Page table texture (R32_UINT - packed physical coordinates)
    pub page_table: vk::Image,
    pub page_table_alloc: Option<vk_mem::Allocation>,
    pub page_table_view: vk::ImageView,
    pub page_table_sampler: vk::Sampler,

    /// Request buffers (one per frame in flight to avoid CPU/GPU race)
    pub request_buffers: Vec<VsmRequestBuffer>,

    /// Physical depth buffer (D32_SFLOAT) for shadow rendering
    pub physical_depth_image: vk::Image,
    physical_depth_image_alloc: Option<vk_mem::Allocation>,
    pub physical_depth_view: vk::ImageView,

    /// Allocation buffer (SSBO)
    pub allocation_buffer: vk::Buffer,
    allocation_buffer_alloc: Option<vk_mem::Allocation>,

    /// Metadata uniform buffer
    pub metadata_buffer: vk::Buffer,
    metadata_buffer_alloc: Option<vk_mem::Allocation>,

    /// Bindless indices
    pub physical_cache_index: u32,
    pub page_table_index: u32,
    pub physical_cache_storage_index: u32,
    pub page_table_storage_index: u32,

    pub default_uint_texture: crate::renderer::resources::Texture,
    pub default_array_texture: crate::renderer::resources::Texture,
}

impl VsmResources {
    /// Get GPU device address of the metadata buffer
    pub fn metadata_address(&self) -> u64 {
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(self.metadata_buffer);
        unsafe { self.device.get_buffer_device_address(&addr_info) }
    }
}

impl VsmResources {
    /// Create new VSM resources
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of these resources.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        command_pool: vk::CommandPool, // Added for texture creation
        queue: vk::Queue,              // Added for texture creation
        config: VsmConfig,
        frames_in_flight: u32,
    ) -> Result<Self> {
        log::info!(
            "Creating VSM resources: virtual={}x{}, physical={}x{}, page_size={}",
            config.virtual_resolution,
            config.virtual_resolution,
            config.physical_resolution,
            config.physical_resolution,
            config.page_size
        );

        // Create physical cache (depth texture)
        let cache_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32G32_SFLOAT) // Store variance moments (depth, depth^2)
            .extent(vk::Extent3D {
                width: config.physical_resolution,
                height: config.physical_resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (physical_cache, physical_cache_alloc) =
            unsafe { allocator.create_image(&cache_info, vk_mem::MemoryUsage::AutoPreferDevice) }
                .map_err(|e| {
                AshError::VulkanError(format!("Failed to create physical cache: {e:?}"))
            })?;

        let cache_view_info = vk::ImageViewCreateInfo::default()
            .image(physical_cache)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R32G32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let physical_cache_view = unsafe { device.create_image_view(&cache_view_info, None) }
            .map_err(|e| AshError::VulkanError(format!("Failed to create cache view: {e:?}")))?;

        // Create sampler for physical cache
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(1.0);

        let physical_cache_sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .map_err(|e| AshError::VulkanError(format!("Failed to create sampler: {e:?}")))?;

        // Create physical depth buffer for Z-testing
        let depth_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width: config.physical_resolution,
                height: config.physical_resolution,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (physical_depth_image, physical_depth_image_alloc) =
            unsafe { allocator.create_image(&depth_info, vk_mem::MemoryUsage::AutoPreferDevice) }
                .map_err(|e| {
                AshError::VulkanError(format!("Failed to create physical depth cache: {e:?}"))
            })?;

        let depth_view_info = vk::ImageViewCreateInfo::default()
            .image(physical_depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let physical_depth_view = unsafe { device.create_image_view(&depth_view_info, None) }
            .map_err(|e| AshError::VulkanError(format!("Failed to create depth view: {e:?}")))?;

        // Create page table (R32_UINT texture array for clipmaps)
        let table_res = config.page_table_resolution();
        let array_layers = config.clipmap_levels.max(1); // At least 1 layer
        let table_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_UINT)
            .extent(vk::Extent3D {
                width: table_res,
                height: table_res,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (page_table, page_table_alloc) =
            unsafe { allocator.create_image(&table_info, vk_mem::MemoryUsage::AutoPreferDevice) }
                .map_err(|e| AshError::VulkanError(format!("Failed to create page table: {e:?}")))?;

        let table_view_info = vk::ImageViewCreateInfo::default()
            .image(page_table)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(vk::Format::R32_UINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: array_layers,
            });

        let page_table_view = unsafe { device.create_image_view(&table_view_info, None) }
            .map_err(|e| AshError::VulkanError(format!("Failed to create table view: {e:?}")))?;

        // Create sampler for page table (nearest neighbor for integer texture)
        let page_table_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(1.0);

        let page_table_sampler = unsafe { device.create_sampler(&page_table_sampler_info, None) }
            .map_err(|e| {
            AshError::VulkanError(format!("Failed to create page table sampler: {e:?}"))
        })?;

        // Create request buffer (SSBO)
        // Usage: STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST (Write by GPU, Read by CPU)
        let request_size = (config.max_requests_per_frame as usize
            * std::mem::size_of::<PageRequest>()) as vk::DeviceSize
            + ATOMIC_HEADER_SIZE;

        let mut request_buffers = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            let (request_handle, request_alloc) = unsafe {
                allocator.create_buffer_with_flags(
                    request_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk_mem::MemoryUsage::AutoPreferHost,
                    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                )
            }
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create request buffer: {e:?}"))
            })?;

            request_buffers.push(VsmRequestBuffer {
                buffer: request_handle,
                allocation: request_alloc,
                size_bytes: request_size,
                max_requests: config.max_requests_per_frame,
            });
        }

        // Create allocation buffer (SSBO)
        // Usage: STORAGE_BUFFER | TRANSFER_DST (Write by CPU, Read by GPU)
        let alloc_size = (config.physical_page_count() as usize
            * std::mem::size_of::<PageAllocation>()) as vk::DeviceSize
            + ATOMIC_HEADER_SIZE;

        let (allocation_buffer, allocation_buffer_alloc) = unsafe {
            allocator.create_buffer_with_flags(
                alloc_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            )
        }
        .map_err(|e| AshError::VulkanError(format!("Failed to create allocation buffer: {e:?}")))?;

        // Create metadata buffer (CPU-writable for per-frame updates)
        let metadata_size = std::mem::size_of::<VsmGlobalInfo>() as vk::DeviceSize;

        let (metadata_buffer, metadata_buffer_alloc) = unsafe {
            allocator.create_buffer_with_flags(
                metadata_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            )
        }
        .map_err(|e| AshError::VulkanError(format!("Failed to create metadata buffer: {e:?}")))?;

        // Create default textures for VSM bindless slots
        let default_uint_texture = crate::renderer::resources::Texture::create_vsm_default_uint(
            Arc::clone(&allocator),
            Arc::clone(&device),
            command_pool,
            queue,
        )?;

        let default_array_texture = crate::renderer::resources::Texture::create_vsm_default_array(
            Arc::clone(&allocator),
            Arc::clone(&device),
            command_pool,
            queue,
            config.clipmap_levels,
        )?;

        log::info!("VSM resources created successfully");

        // EXPLICIT INITIALIZATION: Clear Page Table and transition to GENERAL layout
        unsafe {
            crate::vulkan::utils::execute_single_use(&device, command_pool, queue, |cmd_buffer| {
                // 1. Transition UNDEFINED -> TRANSFER_DST_OPTIMAL (Sync2)
                let barrier_start = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(page_table)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    });

                let image_barriers_start = [barrier_start];
                let dep_info_start =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers_start);
                device.cmd_pipeline_barrier2(cmd_buffer, &dep_info_start);
                // 2. Clear to INVALID_PAGE sentinel (0xFFFFFFFF)
                let clear_value = vk::ClearColorValue {
                    uint32: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: array_layers,
                };

                device.cmd_clear_color_image(
                    cmd_buffer,
                    page_table,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear_value,
                    &[range],
                );

                // 3. Transition TRANSFER_DST_OPTIMAL -> GENERAL (Used by Compute/Fragment shaders) (Sync2)
                let barrier_end = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::COMPUTE_SHADER
                            | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    )
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(page_table)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: array_layers,
                    });

                let image_barriers_end = [barrier_end];
                let dep_info_end =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers_end);
                device.cmd_pipeline_barrier2(cmd_buffer, &dep_info_end);
            })?;
        }

        log::info!("VSM resources created successfully");

        Ok(Self {
            device,
            allocator,
            config,
            physical_cache,
            physical_cache_alloc: Some(physical_cache_alloc),
            physical_cache_view,
            physical_cache_sampler,

            physical_depth_image,
            physical_depth_image_alloc: Some(physical_depth_image_alloc),
            physical_depth_view,

            page_table,
            page_table_alloc: Some(page_table_alloc),
            page_table_view,
            page_table_sampler,
            request_buffers,
            allocation_buffer,
            allocation_buffer_alloc: Some(allocation_buffer_alloc),
            metadata_buffer,
            metadata_buffer_alloc: Some(metadata_buffer_alloc),
            physical_cache_index: u32::MAX, // To be registered
            page_table_index: u32::MAX,     // To be registered
            physical_cache_storage_index: u32::MAX,
            page_table_storage_index: u32::MAX,
            default_uint_texture,
            default_array_texture,
        })
    }

    /// Register VSM resources with bindless manager
    pub fn register_bindless(
        &mut self,
        bindless_manager: &mut crate::vulkan::BindlessManager,
    ) -> Result<()> {
        self.page_table_index =
            bindless_manager.add_page_table(self.page_table_view, self.page_table_sampler)?;

        self.physical_cache_index = bindless_manager
            .add_sampled_image(self.physical_cache_view, self.physical_cache_sampler)?;

        // Register for storage access as well
        self.physical_cache_storage_index =
            bindless_manager.add_storage_image(self.physical_cache_view)?;
        self.page_table_storage_index =
            bindless_manager.add_storage_image_2d_array(self.page_table_view)?;

        log::info!(
            "VSM Registered: page_index={}, cache_index={}",
            self.page_table_index,
            self.physical_cache_index
        );

        Ok(())
    }

    /// Update metadata buffer (Global Info) with current state
    pub fn update_global_info(&self, global_info: &VsmGlobalInfo) -> Result<()> {
        let mut alloc = *self
            .metadata_buffer_alloc
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Metadata buffer not allocated".into()))?;

        unsafe {
            let ptr = self.allocator.vma.map_memory(&mut alloc).map_err(|e| {
                AshError::VulkanError(format!("Failed to map metadata buffer: {e:?}"))
            })?;

            std::ptr::copy_nonoverlapping(
                global_info as *const VsmGlobalInfo as *const u8,
                ptr,
                std::mem::size_of::<VsmGlobalInfo>(),
            );

            self.allocator.vma.unmap_memory(&mut alloc);
        }

        Ok(())
    }

    /// Upload new allocations to the GPU buffer
    pub fn upload_allocations(&self, allocations: &[PageAllocation]) -> Result<()> {
        if allocations.is_empty() {
            return Ok(());
        }

        let mut alloc = *self
            .allocation_buffer_alloc
            .as_ref()
            .ok_or_else(|| AshError::VulkanError("Allocation buffer not allocated".into()))?;

        unsafe {
            let ptr = self.allocator.vma.map_memory(&mut alloc).map_err(|e| {
                AshError::VulkanError(format!("Failed to map allocation buffer: {e:?}"))
            })?;

            // Write count to the first 4 bytes (atomic header)
            let max_phys = self.config.physical_page_count();
            if allocations.len() as u32 > max_phys {
                log::warn!(
                    "VSM: Allocation count ({}) exceeds physical page budget ({}) - truncating.",
                    allocations.len(),
                    max_phys
                );
            }
            let count = (allocations.len() as u32).min(max_phys);
            *(ptr as *mut u32) = count;

            // Write data starting at ATOMIC_HEADER_SIZE (16 bytes)
            let data_ptr = ptr.add(ATOMIC_HEADER_SIZE as usize);
            std::ptr::copy_nonoverlapping(
                allocations.as_ptr(),
                data_ptr as *mut PageAllocation,
                count as usize,
            );

            // Flush memory to ensure GPU visibility on non-coherent heaps
            self.allocator.vma.flush_allocation(
                &alloc,
                ATOMIC_HEADER_SIZE,
                (count as usize * std::mem::size_of::<PageAllocation>()) as u64,
            )?;

            self.allocator.vma.unmap_memory(&mut alloc);
        }

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &VsmConfig {
        &self.config
    }

    /// Destroy resources
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying VSM resources");

        // Destroy buffers
        if let Some(mut alloc) = self.metadata_buffer_alloc.take() {
            unsafe {
                self.allocator
                    .vma
                    .destroy_buffer(self.metadata_buffer, &mut alloc);
            }
        }

        if let Some(mut alloc) = self.allocation_buffer_alloc.take() {
            unsafe {
                self.allocator
                    .vma
                    .destroy_buffer(self.allocation_buffer, &mut alloc);
            }
        }

        for rb in &mut self.request_buffers {
            if rb.buffer != vk::Buffer::null() {
                unsafe {
                    self.allocator.destroy_buffer(rb.buffer, &mut rb.allocation);
                }
                rb.buffer = vk::Buffer::null();
            }
        }

        // Destroy page table
        if self.page_table_sampler != vk::Sampler::null() {
            unsafe {
                self.device.destroy_sampler(self.page_table_sampler, None);
            }
            self.page_table_sampler = vk::Sampler::null();
        }
        if self.page_table_view != vk::ImageView::null() {
            unsafe {
                self.device.destroy_image_view(self.page_table_view, None);
            }
            self.page_table_view = vk::ImageView::null();
        }
        if let Some(mut alloc) = self.page_table_alloc.take() {
            unsafe {
                self.allocator
                    .vma
                    .destroy_image(self.page_table, &mut alloc);
            }
            self.page_table = vk::Image::null();
        }

        // Destroy physical cache
        if self.physical_cache_view != vk::ImageView::null() {
            unsafe {
                self.device
                    .destroy_image_view(self.physical_cache_view, None);
            }
            self.physical_cache_view = vk::ImageView::null();
        }
        if self.physical_cache_sampler != vk::Sampler::null() {
            unsafe {
                self.device
                    .destroy_sampler(self.physical_cache_sampler, None);
            }
            self.physical_cache_sampler = vk::Sampler::null();
        }
        if let Some(mut alloc) = self.physical_cache_alloc.take() {
            unsafe {
                self.allocator
                    .vma
                    .destroy_image(self.physical_cache, &mut alloc);
            }
            self.physical_cache = vk::Image::null();
        }

        // Destroy physical depth
        if self.physical_depth_view != vk::ImageView::null() {
            unsafe {
                self.device
                    .destroy_image_view(self.physical_depth_view, None);
            }
            self.physical_depth_view = vk::ImageView::null();
        }
        if let Some(mut alloc) = self.physical_depth_image_alloc.take() {
            unsafe {
                self.allocator
                    .vma
                    .destroy_image(self.physical_depth_image, &mut alloc);
            }
            self.physical_depth_image = vk::Image::null();
        }

        log::debug!("VSM resources destroyed");
    }
}

impl Drop for VsmResources {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
