//! VSM GPU Resources - Physical Cache, Page Table, and Buffers

use ash::vk;
use std::sync::Arc;

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
    pub max_requests_per_frame: u32,
    /// Enable debug visualization
    pub debug_mode: bool,
    /// Number of clipmap levels for directional lights (0 = disabled, typical: 8)
    pub clipmap_levels: u32,
    /// World-space extent of clipmap level 0 (in meters, e.g., 100.0)
    pub clipmap_base_extent: f32,
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

/// VSM GPU Resources
pub struct VsmResources {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,
    config: VsmConfig,

    /// Physical cache texture (R32_FLOAT depth values)
    pub physical_cache: vk::Image,
    physical_cache_alloc: Option<vk_mem::Allocation>,
    pub physical_cache_view: vk::ImageView,
    pub physical_cache_sampler: vk::Sampler,

    /// Page table texture (R32_UINT - packed physical coordinates)
    pub page_table: vk::Image,
    page_table_alloc: Option<vk_mem::Allocation>,
    pub page_table_view: vk::ImageView,
    pub page_table_sampler: vk::Sampler,

    /// Request buffer (SSBO)
    pub request_buffer: vk::Buffer,
    request_buffer_alloc: Option<vk_mem::Allocation>,

    /// Allocation buffer (SSBO)
    pub allocation_buffer: vk::Buffer,
    allocation_buffer_alloc: Option<vk_mem::Allocation>,

    /// Metadata uniform buffer
    pub metadata_buffer: vk::Buffer,
    metadata_buffer_alloc: Option<vk_mem::Allocation>,
}

impl VsmResources {
    /// Create new VSM resources
    ///
    /// # Safety
    /// Device and allocator must remain valid for the lifetime of these resources.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Allocator>,
        config: VsmConfig,
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
            .format(vk::Format::R32_SFLOAT)
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
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (physical_cache, physical_cache_alloc) = allocator
            .create_image(&cache_info, vk_mem::MemoryUsage::AutoPreferDevice)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create physical cache: {e:?}"))
            })?;

        let cache_view_info = vk::ImageViewCreateInfo::default()
            .image(physical_cache)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let physical_cache_view = device
            .create_image_view(&cache_view_info, None)
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

        let physical_cache_sampler = device
            .create_sampler(&sampler_info, None)
            .map_err(|e| AshError::VulkanError(format!("Failed to create sampler: {e:?}")))?;

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

        let (page_table, page_table_alloc) = allocator
            .create_image(&table_info, vk_mem::MemoryUsage::AutoPreferDevice)
            .map_err(|e| AshError::VulkanError(format!("Failed to create page table: {e:?}")))?;

        let table_view_info = vk::ImageViewCreateInfo::default()
            .image(page_table)
            .view_type(if array_layers > 1 {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(vk::Format::R32_UINT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: array_layers,
            });

        let page_table_view = device
            .create_image_view(&table_view_info, None)
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

        let page_table_sampler = device
            .create_sampler(&page_table_sampler_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create page table sampler: {e:?}"))
            })?;

        // Create request buffer (SSBO)
        let request_size = (config.max_requests_per_frame as usize
            * std::mem::size_of::<PageRequest>()) as vk::DeviceSize;

        let (request_buffer, request_buffer_alloc) = allocator
            .create_buffer(
                request_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create request buffer: {e:?}"))
            })?;

        // Create allocation buffer (SSBO)
        let alloc_size = (config.physical_page_count() as usize
            * std::mem::size_of::<PageAllocation>()) as vk::DeviceSize;

        let (allocation_buffer, allocation_buffer_alloc) = allocator
            .create_buffer(
                alloc_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create allocation buffer: {e:?}"))
            })?;

        // Create metadata buffer (CPU-writable for per-frame updates)
        let metadata_size = std::mem::size_of::<VsmMetadata>() as vk::DeviceSize;

        let (metadata_buffer, metadata_buffer_alloc) = allocator
            .create_buffer_with_flags(
                metadata_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            )
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create metadata buffer: {e:?}"))
            })?;

        log::info!("VSM resources created successfully");

        Ok(Self {
            device,
            allocator,
            config,
            physical_cache,
            physical_cache_alloc: Some(physical_cache_alloc),
            physical_cache_view,
            physical_cache_sampler,
            page_table,
            page_table_alloc: Some(page_table_alloc),
            page_table_view,
            page_table_sampler,
            request_buffer,
            request_buffer_alloc: Some(request_buffer_alloc),
            allocation_buffer,
            allocation_buffer_alloc: Some(allocation_buffer_alloc),
            metadata_buffer,
            metadata_buffer_alloc: Some(metadata_buffer_alloc),
        })
    }

    /// Update metadata buffer with current configuration
    pub fn update_metadata(&self, frame_index: u32) -> Result<()> {
        let metadata = VsmMetadata {
            virtual_resolution: self.config.virtual_resolution,
            physical_resolution: self.config.physical_resolution,
            page_size: self.config.page_size,
            page_table_resolution: self.config.page_table_resolution(),
            physical_page_count: self.config.physical_page_count(),
            frame_index,
            debug_flags: if self.config.debug_mode { 1 } else { 0 },
            _padding: 0,
        };

        unsafe {
            let mut alloc = self.metadata_buffer_alloc.as_ref().unwrap().clone();
            let ptr = self.allocator.vma.map_memory(&mut alloc).map_err(|e| {
                AshError::VulkanError(format!("Failed to map metadata buffer: {e:?}"))
            })?;

            std::ptr::copy_nonoverlapping(
                &metadata as *const VsmMetadata as *const u8,
                ptr,
                std::mem::size_of::<VsmMetadata>(),
            );

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
            self.allocator
                .vma
                .destroy_buffer(self.metadata_buffer, &mut alloc);
        }

        if let Some(mut alloc) = self.allocation_buffer_alloc.take() {
            self.allocator
                .vma
                .destroy_buffer(self.allocation_buffer, &mut alloc);
        }

        if let Some(mut alloc) = self.request_buffer_alloc.take() {
            self.allocator
                .vma
                .destroy_buffer(self.request_buffer, &mut alloc);
        }

        // Destroy page table
        if self.page_table_sampler != vk::Sampler::null() {
            self.device.destroy_sampler(self.page_table_sampler, None);
            self.page_table_sampler = vk::Sampler::null();
        }
        if self.page_table_view != vk::ImageView::null() {
            self.device.destroy_image_view(self.page_table_view, None);
            self.page_table_view = vk::ImageView::null();
        }
        if let Some(mut alloc) = self.page_table_alloc.take() {
            self.allocator
                .vma
                .destroy_image(self.page_table, &mut alloc);
            self.page_table = vk::Image::null();
        }

        // Destroy physical cache
        if self.physical_cache_view != vk::ImageView::null() {
            self.device
                .destroy_image_view(self.physical_cache_view, None);
            self.physical_cache_view = vk::ImageView::null();
        }
        if self.physical_cache_sampler != vk::Sampler::null() {
            self.device
                .destroy_sampler(self.physical_cache_sampler, None);
            self.physical_cache_sampler = vk::Sampler::null();
        }
        if let Some(mut alloc) = self.physical_cache_alloc.take() {
            self.allocator
                .vma
                .destroy_image(self.physical_cache, &mut alloc);
            self.physical_cache = vk::Image::null();
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
