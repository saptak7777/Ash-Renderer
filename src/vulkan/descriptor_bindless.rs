use ash::vk;
use std::sync::Arc;

use crate::{AshError, Result};

use super::descriptor_allocator::DescriptorAllocator;
use super::descriptor_layout::{DescriptorSetLayout, DescriptorSetLayoutBuilder};
use super::descriptor_set::DescriptorSet;

#[derive(Clone)]
struct RegisteredResource {
    index: u32,
    binding: u32,
    info: ResourceInfo,
}

#[derive(Clone)]
enum ResourceInfo {
    Image {
        view: vk::ImageView,
        sampler: vk::Sampler,
    },
    Buffer {
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    },
}

/// Manages bindless descriptor resources (textures, page tables, cubemaps, buffers).
pub struct BindlessManager {
    layout: DescriptorSetLayout,
    descriptor_set: DescriptorSet,
    max_images: u32,
    max_page_tables: u32,
    max_cubemaps: u32,
    max_storage_images: u32,
    max_buffers: u32,
    next_image_index: u32,
    next_page_table_index: u32,
    next_cubemap_index: u32,
    next_storage_image_index: u32,
    next_buffer_index: u32,
    // Resource tracking for recreation
    resources: Vec<RegisteredResource>,
}

impl BindlessManager {
    pub const DEFAULT_MAX_TEXTURES: u32 = 16384;
    pub const DEFAULT_MAX_PAGE_TABLES: u32 = 1024;
    pub const DEFAULT_MAX_CUBEMAPS: u32 = 1024;
    pub const DEFAULT_MAX_STORAGE_IMAGES: u32 = 1024;
    pub const DEFAULT_MAX_BUFFERS: u32 = 1024;

    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: Arc<ash::Device>,
        allocator: &mut DescriptorAllocator,
        max_images: u32,
        max_page_tables: u32,
        max_cubemaps: u32,
        max_storage_images: u32,
        mut max_buffers: u32,
    ) -> Result<Self> {
        // Hardware Validation: Clamp buffers to hardware limits to prevent DEVICE_LOST
        unsafe {
            let mut v12_props = vk::PhysicalDeviceVulkan12Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut v12_props);
            instance.get_physical_device_properties2(physical_device, &mut props2);

            let hw_max_buffers = v12_props.max_descriptor_set_update_after_bind_storage_buffers;
            if max_buffers > hw_max_buffers {
                log::warn!(
                    "Requested {} bindless storage buffers, but hardware only supports {}. Clamping.",
                    max_buffers,
                    hw_max_buffers
                );
                max_buffers = hw_max_buffers;
            }
        }

        let layout = DescriptorSetLayoutBuilder::new()
            .add_bindless_binding(
                0, // global_textures
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_images,
            )
            .add_bindless_binding(
                1, // global_page_tables
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_page_tables,
            )
            .add_bindless_binding(
                2, // global_cubemaps
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_cubemaps,
            )
            .add_bindless_binding(
                3, // global_storage_images
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_storage_images,
            )
            .add_bindless_binding(
                4, // global_buffers
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE,
                max_buffers,
            )
            .build(Arc::clone(&device))?;

        // Bindless descriptors must be allocated from a pool with UPDATE_AFTER_BIND bit
        // The last binding (buffers) uses the variable descriptor count
        let descriptor_set =
            allocator.allocate_bindless_set(layout.handle(), layout.bindings(), max_buffers)?;

        Ok(Self {
            layout,
            descriptor_set,
            max_images,
            max_page_tables,
            max_cubemaps,
            max_storage_images,
            max_buffers,
            next_image_index: 0,
            next_page_table_index: 0,
            next_cubemap_index: 0,
            next_storage_image_index: 0,
            next_buffer_index: 0,
            resources: Vec::new(),
        })
    }

    /// Recreate the descriptor set (e.g., after swapchain resize)
    pub fn recreate(&mut self, allocator: &mut DescriptorAllocator) -> Result<()> {
        log::info!("Recreating bindless descriptor set...");

        let new_descriptor_set = allocator.allocate_bindless_set(
            self.layout.handle(),
            self.layout.bindings(),
            self.max_buffers,
        )?;

        // Free the old descriptor set
        allocator.free_bindless_set(self.descriptor_set.handle())?;
        self.descriptor_set = new_descriptor_set;

        // Re-register all resources
        self.re_register_all()?;
        Ok(())
    }

    fn re_register_all(&mut self) -> Result<()> {
        for res in &self.resources {
            match res.info {
                ResourceInfo::Image { view, sampler } => {
                    let (descriptor_type, image_layout) = if res.binding == 3 {
                        (vk::DescriptorType::STORAGE_IMAGE, vk::ImageLayout::GENERAL)
                    } else {
                        (
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        )
                    };

                    let info = vk::DescriptorImageInfo {
                        sampler,
                        image_view: view,
                        image_layout,
                    };
                    self.descriptor_set.update_image_at(
                        res.binding,
                        res.index,
                        info,
                        descriptor_type,
                    )?;
                }
                ResourceInfo::Buffer {
                    buffer,
                    offset,
                    range,
                } => {
                    self.descriptor_set.update_buffer_at(
                        res.binding,
                        res.index,
                        buffer,
                        offset,
                        range,
                        vk::DescriptorType::STORAGE_BUFFER,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.layout.handle()
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set.handle()
    }

    /// Update an existing sampled image at a given index.
    /// Used for transient resources like HDR buffers that are recreated on resize.
    pub fn update_sampled_image(
        &mut self,
        index: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<()> {
        // Find the tracked resource and update its info
        let res = self
            .resources
            .iter_mut()
            .find(|r| r.index == index && r.binding == 0)
            .ok_or_else(|| {
                AshError::VulkanError(format!(
                    "Bindless index {} (binding 0) not found for update",
                    index
                ))
            })?;

        res.info = ResourceInfo::Image {
            view: image_view,
            sampler,
        };

        // Update the actual descriptor set
        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        self.descriptor_set.update_image_at(
            0,
            index,
            info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        Ok(())
    }

    pub fn add_sampled_image(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<u32> {
        let index = self.allocate_index(0)?;
        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        self.descriptor_set.update_image_at(
            0,
            index,
            info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        self.resources.push(RegisteredResource {
            index,
            binding: 0,
            info: ResourceInfo::Image {
                view: image_view,
                sampler,
            },
        });

        Ok(index)
    }

    pub fn add_page_table(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<u32> {
        let index = self.allocate_index(1)?;
        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        self.descriptor_set.update_image_at(
            1,
            index,
            info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        self.resources.push(RegisteredResource {
            index,
            binding: 1,
            info: ResourceInfo::Image {
                view: image_view,
                sampler,
            },
        });

        Ok(index)
    }

    pub fn add_cubemap(&mut self, image_view: vk::ImageView, sampler: vk::Sampler) -> Result<u32> {
        let index = self.allocate_index(2)?;
        let info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        self.descriptor_set.update_image_at(
            2,
            index,
            info,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        self.resources.push(RegisteredResource {
            index,
            binding: 2,
            info: ResourceInfo::Image {
                view: image_view,
                sampler,
            },
        });

        Ok(index)
    }

    pub fn add_storage_image(&mut self, image_view: vk::ImageView) -> Result<u32> {
        let index = self.allocate_index(3)?;
        let info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        self.descriptor_set
            .update_image_at(3, index, info, vk::DescriptorType::STORAGE_IMAGE)?;

        self.resources.push(RegisteredResource {
            index,
            binding: 3,
            info: ResourceInfo::Image {
                view: image_view,
                sampler: vk::Sampler::null(),
            },
        });

        Ok(index)
    }

    pub fn add_storage_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Result<u32> {
        let index = self.allocate_index(4)?;
        self.descriptor_set.update_buffer_at(
            4,
            index,
            buffer,
            offset,
            range,
            vk::DescriptorType::STORAGE_BUFFER,
        )?;

        self.resources.push(RegisteredResource {
            index,
            binding: 4,
            info: ResourceInfo::Buffer {
                buffer,
                offset,
                range,
            },
        });

        Ok(index)
    }

    pub fn validate_index(&self, index: u32) -> Result<()> {
        if index >= self.max_images && index >= self.max_buffers {
            return Err(AshError::VulkanError(format!(
                "Invalid bindless index: {} (max_images: {}, max_buffers: {})",
                index, self.max_images, self.max_buffers
            )));
        }
        Ok(())
    }

    fn allocate_index(&mut self, binding: u32) -> Result<u32> {
        match binding {
            0 => {
                if self.next_image_index >= self.max_images {
                    return Err(AshError::VulkanError("Exceeded max images".into()));
                }
                let idx = self.next_image_index;
                self.next_image_index += 1;
                Ok(idx)
            }
            1 => {
                if self.next_page_table_index >= self.max_page_tables {
                    return Err(AshError::VulkanError("Exceeded max page tables".into()));
                }
                let idx = self.next_page_table_index;
                self.next_page_table_index += 1;
                Ok(idx)
            }
            2 => {
                if self.next_cubemap_index >= self.max_cubemaps {
                    return Err(AshError::VulkanError("Exceeded max cubemaps".into()));
                }
                let idx = self.next_cubemap_index;
                self.next_cubemap_index += 1;
                Ok(idx)
            }
            3 => {
                if self.next_storage_image_index >= self.max_storage_images {
                    return Err(AshError::VulkanError("Exceeded max storage images".into()));
                }
                let idx = self.next_storage_image_index;
                self.next_storage_image_index += 1;
                Ok(idx)
            }
            4 => {
                if self.next_buffer_index >= self.max_buffers {
                    return Err(AshError::VulkanError("Exceeded max buffers".into()));
                }
                let idx = self.next_buffer_index;
                self.next_buffer_index += 1;
                Ok(idx)
            }
            _ => Err(AshError::VulkanError("Invalid bindless binding".into())),
        }
    }
}
