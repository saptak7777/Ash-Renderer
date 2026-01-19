//! VSM Compute Pipelines - Analysis and Allocation

use ash::vk;
use std::sync::Arc;

use crate::vulkan::{Allocator, ComputePipeline};
use crate::{AshError, Result};

use super::resources::VsmResources;

/// VSM compute pipeline manager
pub struct VsmComputePipelines {
    device: Arc<ash::Device>,

    /// Analysis pipeline (determines needed pages)
    pub analysis_pipeline: ComputePipeline,
    pub analysis_layout: vk::PipelineLayout,
    pub analysis_descriptor_set_layout: vk::DescriptorSetLayout,

    /// Allocator pipeline (assigns physical pages)
    pub allocator_pipeline: ComputePipeline,
    pub allocator_layout: vk::PipelineLayout,
    pub allocator_descriptor_set_layout: vk::DescriptorSetLayout,
}

impl VsmComputePipelines {
    /// Create VSM compute pipelines
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of these pipelines.
    pub unsafe fn new(device: Arc<ash::Device>, _allocator: Arc<Allocator>) -> Result<Self> {
        log::info!("Creating VSM compute pipelines");

        // Create analysis descriptor set layout
        let analysis_bindings = [
            // Binding 0: Depth buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: VSM metadata
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 2: Camera data
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 3: Light data
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 4: Page table (storage image)
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 5: Request buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 6: Request data
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let analysis_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&analysis_bindings);

        let analysis_descriptor_set_layout = device
            .create_descriptor_set_layout(&analysis_layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!(
                    "Failed to create analysis descriptor layout: {e:?}"
                ))
            })?;

        // Create analysis pipeline layout
        let analysis_layouts = [analysis_descriptor_set_layout];
        let analysis_pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&analysis_layouts);

        let analysis_layout = device
            .create_pipeline_layout(&analysis_pipeline_layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create analysis pipeline layout: {e:?}"))
            })?;

        // Create analysis pipeline
        let analysis_shader_code = std::fs::read("shaders/vsm/analyze.comp.spv")
            .map_err(|e| AshError::VulkanError(format!("Failed to read analysis shader: {e:?}")))?;

        let analysis_shader_info =
            vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(&analysis_shader_code));

        let analysis_shader_module = device
            .create_shader_module(&analysis_shader_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create analysis shader module: {e:?}"))
            })?;

        let entry_point = std::ffi::CString::new("main").unwrap();
        let analysis_pipeline = ComputePipeline::new(
            Arc::clone(&device),
            analysis_layout,
            analysis_shader_module,
            &entry_point,
        )?;

        // Clean up shader module
        device.destroy_shader_module(analysis_shader_module, None);

        // Create allocator descriptor set layout
        let allocator_bindings = [
            // Binding 0: VSM metadata
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 1: Request buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 2: Request data
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 3: Page table (storage image)
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 4: Allocation buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 5: Allocation data
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Binding 6: Free page pool
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let allocator_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&allocator_bindings);

        let allocator_descriptor_set_layout = device
            .create_descriptor_set_layout(&allocator_layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!(
                    "Failed to create allocator descriptor layout: {e:?}"
                ))
            })?;

        // Create allocator pipeline layout
        let allocator_layouts = [allocator_descriptor_set_layout];
        let allocator_pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&allocator_layouts);

        let allocator_layout = device
            .create_pipeline_layout(&allocator_pipeline_layout_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create allocator pipeline layout: {e:?}"))
            })?;

        // Create allocator pipeline
        let allocator_shader_code =
            std::fs::read("shaders/vsm/allocator.comp.spv").map_err(|e| {
                AshError::VulkanError(format!("Failed to read allocator shader: {e:?}"))
            })?;

        let allocator_shader_info = vk::ShaderModuleCreateInfo::default()
            .code(bytemuck::cast_slice(&allocator_shader_code));

        let allocator_shader_module = device
            .create_shader_module(&allocator_shader_info, None)
            .map_err(|e| {
                AshError::VulkanError(format!("Failed to create allocator shader module: {e:?}"))
            })?;

        let entry_point = std::ffi::CString::new("main").unwrap();
        let allocator_pipeline = ComputePipeline::new(
            Arc::clone(&device),
            allocator_layout,
            allocator_shader_module,
            &entry_point,
        )?;

        // Clean up shader module
        device.destroy_shader_module(allocator_shader_module, None);

        log::info!("VSM compute pipelines created successfully");

        Ok(Self {
            device,
            analysis_pipeline,
            analysis_layout,
            analysis_descriptor_set_layout,
            allocator_pipeline,
            allocator_layout,
            allocator_descriptor_set_layout,
        })
    }

    /// Dispatch analysis pass
    ///
    /// # Safety
    /// Command buffer must be in recording state. Descriptor set must be valid.
    pub unsafe fn dispatch_analysis(
        &self,
        cmd: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        screen_width: u32,
        screen_height: u32,
    ) {
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.analysis_pipeline.handle(),
        );

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.analysis_layout,
            0,
            &[descriptor_set],
            &[],
        );

        // Dispatch with 8x8 local size
        let group_count_x = (screen_width + 7) / 8;
        let group_count_y = (screen_height + 7) / 8;

        self.device
            .cmd_dispatch(cmd, group_count_x, group_count_y, 1);
    }

    /// Dispatch allocator pass
    ///
    /// # Safety
    /// Command buffer must be in recording state. Descriptor set must be valid.
    pub unsafe fn dispatch_allocator(
        &self,
        cmd: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        request_count: u32,
    ) {
        self.device.cmd_bind_pipeline(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.allocator_pipeline.handle(),
        );

        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.allocator_layout,
            0,
            &[descriptor_set],
            &[],
        );

        // Dispatch with 64 threads per group
        let group_count = (request_count + 63) / 64;

        self.device.cmd_dispatch(cmd, group_count, 1, 1);
    }

    /// Destroy pipelines
    ///
    /// # Safety
    /// Must be called before device is destroyed. Resources must not be in use.
    pub unsafe fn destroy(&mut self) {
        log::debug!("Destroying VSM compute pipelines");

        // Pipelines are destroyed by Drop (ComputePipeline struct)

        if self.analysis_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.analysis_layout, None);
            self.analysis_layout = vk::PipelineLayout::null();
        }
        if self.analysis_descriptor_set_layout != vk::DescriptorSetLayout::null() {
            self.device
                .destroy_descriptor_set_layout(self.analysis_descriptor_set_layout, None);
            self.analysis_descriptor_set_layout = vk::DescriptorSetLayout::null();
        }

        if self.allocator_layout != vk::PipelineLayout::null() {
            self.device
                .destroy_pipeline_layout(self.allocator_layout, None);
            self.allocator_layout = vk::PipelineLayout::null();
        }
        if self.allocator_descriptor_set_layout != vk::DescriptorSetLayout::null() {
            self.device
                .destroy_descriptor_set_layout(self.allocator_descriptor_set_layout, None);
            self.allocator_descriptor_set_layout = vk::DescriptorSetLayout::null();
        }

        log::debug!("VSM compute pipelines destroyed");
    }
}

impl Drop for VsmComputePipelines {
    fn drop(&mut self) {
        unsafe {
            self.destroy();
        }
    }
}
