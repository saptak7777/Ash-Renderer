use crate::renderer::vcgs::IndirectDrawCommand;
use crate::vulkan::Allocator;
use crate::Result;
use ash::vk;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowCullPushConstants {
    pub view_proj: [[f32; 4]; 4],
    pub object_count: u32,
    pub base_index: u32,
    pub indirect_start: u32,
    pub _padding: u32,
    pub object_buffer_ptr: u64,
    pub indirect_buffer_ptr: u64,
    pub count_buffer_ptr: u64,
}

pub struct ShadowCullPass {
    device: Arc<ash::Device>,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,

    destroyed: bool,

    // Per-frame resources
    pub indirect_buffers: Vec<vk::Buffer>,
    pub indirect_allocs: Vec<vk_mem::Allocation>,
    pub count_buffers: Vec<vk::Buffer>,
    pub count_allocs: Vec<vk_mem::Allocation>,

    max_commands: u32,
    clipmap_levels: u32,
}

impl ShadowCullPass {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: &Arc<Allocator>,
        frame_count: u32,
        max_objects: u32,
        clipmap_levels: u32,
    ) -> Result<Self> {
        if clipmap_levels == 0 {
            return Err(crate::AshError::VulkanError(
                "clipmap_levels must be > 0".into(),
            ));
        }
        log::info!(
            "Creating ShadowCullPass (max_objects={}, clipmap_levels={})",
            max_objects,
            clipmap_levels
        );

        let max_commands = max_objects * clipmap_levels;

        // Load shader module
        let shader_code = include_bytes!(concat!(env!("OUT_DIR"), "/shadow_cull.comp.spv"));
        let code = ash::util::read_spv(&mut std::io::Cursor::new(shader_code))
            .map_err(|e| crate::AshError::VulkanError(e.to_string()))?;

        let shader_module_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { device.create_shader_module(&shader_module_info, None)? };

        // Pipeline layout
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<ShadowCullPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        // Pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap());

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };
        let pipeline = pipelines[0];

        unsafe { device.destroy_shader_module(shader_module, None) };

        let mut pass = Self {
            device,
            pipeline,
            layout,
            indirect_buffers: Vec::new(),
            indirect_allocs: Vec::new(),
            count_buffers: Vec::new(),
            count_allocs: Vec::new(),
            max_commands,
            clipmap_levels,
            destroyed: false,
        };

        unsafe { pass.init_buffers(allocator, frame_count, clipmap_levels)? };

        Ok(pass)
    }

    unsafe fn init_buffers(
        &mut self,
        allocator: &Arc<Allocator>,
        frame_count: u32,
        clipmap_levels: u32,
    ) -> Result<()> {
        let command_buffer_size = (self.max_commands as usize
            * std::mem::size_of::<IndirectDrawCommand>())
            as vk::DeviceSize;
        let count_buffer_size = (clipmap_levels as usize * 4) as vk::DeviceSize;

        for i in 0..frame_count {
            let (indirect_buffer, indirect_alloc) = allocator.create_buffer_with_flags_and_name(
                command_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
                vk_mem::AllocationCreateFlags::empty(),
                Some(format!("Shadow Indirect Buffer {}", i)),
            )?;

            let (count_buffer, count_alloc) = allocator.create_buffer_with_flags_and_name(
                count_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferDevice,
                vk_mem::AllocationCreateFlags::empty(),
                Some(format!("Shadow Count Buffer {}", i)),
            )?;

            self.indirect_buffers.push(indirect_buffer);
            self.indirect_allocs.push(indirect_alloc);
            self.count_buffers.push(count_buffer);
            self.count_allocs.push(count_alloc);
        }

        Ok(())
    }

    pub unsafe fn cull_shadows(
        &self,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        view_proj: glam::Mat4,
        object_count: u32,
        base_index: u32,
        clipmap_level: u32,
        object_buffer_ptr: u64,
    ) {
        let indirect_buffer = self.indirect_buffers[frame_index];
        let count_buffer = self.count_buffers[frame_index];

        let indirect_ptr = self.device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(indirect_buffer),
        );
        let count_ptr = self.device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(count_buffer),
        );

        let level_count_ptr = count_ptr + (clipmap_level as u64 * 4);
        let max_objects_per_level = self.max_commands / self.clipmap_levels;

        let push_constants = ShadowCullPushConstants {
            view_proj: view_proj.to_cols_array_2d(),
            object_count,
            base_index,
            indirect_start: clipmap_level * max_objects_per_level,
            _padding: 0,
            object_buffer_ptr,
            indirect_buffer_ptr: indirect_ptr,
            count_buffer_ptr: level_count_ptr,
        };

        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        self.device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        let group_count = (object_count + 63) / 64;
        if group_count > 0 {
            self.device.cmd_dispatch(cmd, group_count, 1, 1);
        }
    }

    pub unsafe fn destroy(&mut self, allocator: &Arc<Allocator>) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;

        for mut alloc in self.indirect_allocs.drain(..) {
            let buffer = self.indirect_buffers.remove(0);
            allocator.destroy_buffer(buffer, &mut alloc);
        }
        for mut alloc in self.count_allocs.drain(..) {
            let buffer = self.count_buffers.remove(0);
            allocator.destroy_buffer(buffer, &mut alloc);
        }
        if self.pipeline != vk::Pipeline::null() {
            self.device.destroy_pipeline(self.pipeline, None);
        }
        if self.layout != vk::PipelineLayout::null() {
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
