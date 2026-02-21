//! VSM Compute Pipelines - Analysis and Allocation

use ash::vk;
use std::sync::Arc;

use crate::Result;
use crate::vulkan::ComputePipeline;

/// VSM compute pipeline manager
pub struct VsmComputePipelines {
    /// Pipeline for clearing physical pages
    pub clear: ComputePipeline,
    /// Pipeline for analyzing scene depth and generating requests
    pub analyze: ComputePipeline,
    /// Pipeline for allocating physical pages on GPU
    pub allocate: ComputePipeline,
}

impl VsmComputePipelines {
    /// Create VSM compute pipelines
    ///
    /// # Safety
    /// Device must remain valid for the lifetime of these pipelines.
    pub unsafe fn new(
        device: Arc<ash::Device>,
        descriptor_layout: vk::DescriptorSetLayout,
        max_requests: u32,
    ) -> Result<Self> {
        log::info!("Creating VSM compute pipelines");

        // 1. Clear Pipeline
        let clear_shader = include_bytes!(concat!(env!("OUT_DIR"), "/clear.glsl.spv"));
        let clear_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(bytemuck::cast_slice(clear_shader)),
                    None,
                )
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to create clear shader: {e}"))
                })?
        };

        let clear = unsafe {
            ComputePipeline::builder(Arc::clone(&device))
                .add_set_layout(descriptor_layout)
                .with_shader(clear_module)
                .with_entry_point("main")
                .build()?
        };
        unsafe { device.destroy_shader_module(clear_module, None) };

        // 2. Analyze Pipeline
        let analyze_shader = include_bytes!(concat!(env!("OUT_DIR"), "/analyze.glsl.spv"));
        let analyze_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default()
                        .code(bytemuck::cast_slice(analyze_shader)),
                    None,
                )
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to create analyze shader: {e}"))
                })?
        };

        let spec_data = max_requests.to_ne_bytes();
        let spec_map = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(0)
            .size(4)];
        let spec_info = vk::SpecializationInfo::default()
            .map_entries(&spec_map)
            .data(&spec_data);

        let analyze = unsafe {
            ComputePipeline::builder(Arc::clone(&device))
                .add_set_layout(descriptor_layout)
                .with_shader(analyze_module)
                .with_entry_point("main")
                .with_specialization(spec_info)
                .build()?
        };
        unsafe { device.destroy_shader_module(analyze_module, None) };

        // 3. Allocate Pipeline
        let allocate_shader = include_bytes!(concat!(env!("OUT_DIR"), "/allocate.glsl.spv"));
        let allocate_module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default()
                        .code(bytemuck::cast_slice(allocate_shader)),
                    None,
                )
                .map_err(|e| {
                    crate::AshError::VulkanError(format!("Failed to create allocate shader: {e}"))
                })?
        };

        let allocate = unsafe {
            ComputePipeline::builder(Arc::clone(&device))
                .add_set_layout(descriptor_layout)
                .with_shader(allocate_module)
                .with_entry_point("main")
                .build()?
        };
        unsafe { device.destroy_shader_module(allocate_module, None) };

        log::info!("VSM compute pipelines loaded successfully");

        Ok(Self {
            clear,
            analyze,
            allocate,
        })
    }

    /// Destroy pipelines (handled by Drop)
    ///
    /// # Safety
    /// This should only be called when the GPU is no longer using these pipelines.
    pub unsafe fn destroy(&mut self) {
        // ComputePipeline implements Drop
    }
}
