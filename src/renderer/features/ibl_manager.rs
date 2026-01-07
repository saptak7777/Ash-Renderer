use crate::renderer::resources::ImageHandle;
use crate::vulkan::{Allocator, Framebuffer, Pipeline, VulkanDevice};
use crate::Result;
use ash::vk;
use glam::{Mat4, Vec3};
use std::sync::Arc;

const CUBE_VERTICES: [f32; 108] = [
    -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0,
];

pub struct IblManager {
    device: Arc<ash::Device>,
    allocator: Arc<Allocator>,
    cube_buffer: Option<crate::renderer::resources::BufferHandle>,
    descriptor_layout: Option<crate::vulkan::descriptor_layout::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    render_pass: Option<crate::vulkan::RenderPass>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct IblPushConstants {
    view: [f32; 16],
    projection: [f32; 16],
    roughness: f32,
    _padding: [f32; 3],
}

struct IblRenderPassParams<'a> {
    target: &'a ImageHandle,
    source_view: vk::ImageView,
    source_sampler: vk::Sampler,
    shader_bytes: (&'a [u8], &'a [u8]),
    mip: u32,
    roughness: f32,
}

impl IblManager {
    pub fn new(device: Arc<ash::Device>, allocator: Arc<Allocator>) -> Self {
        Self {
            device,
            allocator,
            cube_buffer: None,
            descriptor_layout: None,
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: None,
        }
    }

    fn ensure_cube_buffer(&mut self) -> Result<vk::Buffer> {
        if let Some(ref buffer) = self.cube_buffer {
            return Ok(buffer.handle());
        }

        let mut buffer = unsafe {
            crate::renderer::resources::BufferHandle::new(
                Arc::clone(&self.allocator),
                std::mem::size_of_val(&CUBE_VERTICES) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk_mem::MemoryUsage::AutoPreferHost,
                Some("IBL_Cube_Buffer".to_string()),
            )?
        };

        unsafe {
            let allocation = buffer.allocation_mut();
            let ptr = self.allocator.vma.map_memory(allocation)?;
            std::ptr::copy_nonoverlapping(CUBE_VERTICES.as_ptr(), ptr.cast(), CUBE_VERTICES.len());
            self.allocator.vma.unmap_memory(allocation);
        }

        let handle = buffer.handle();
        self.cube_buffer = Some(buffer);
        Ok(handle)
    }

    fn ensure_resources(&mut self) -> Result<()> {
        if self.render_pass.is_some() {
            return Ok(());
        }

        // 1. Create Render Pass
        let render_pass = crate::vulkan::RenderPass::builder(Arc::clone(&self.device))
            .with_color_attachment(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
            .build()?;

        // 2. Descriptor Set Layout
        let desc_layout = crate::vulkan::descriptor_layout::DescriptorSetLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ShaderStageFlags::FRAGMENT,
                1,
            )
            .build(Arc::clone(&self.device))?;

        // 3. Pipeline Layout
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            offset: 0,
            size: std::mem::size_of::<IblPushConstants>() as u32,
        };
        let layouts = [desc_layout.handle()];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        self.render_pass = Some(render_pass);
        self.descriptor_layout = Some(desc_layout);
        self.pipeline_layout = pipeline_layout;

        Ok(())
    }

    /// Generic helper to render to each face of a cubemap.
    fn render_to_cubemap(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        params: IblRenderPassParams,
    ) -> Result<()> {
        let IblRenderPassParams {
            target,
            source_view,
            source_sampler,
            shader_bytes,
            mip,
            roughness,
        } = params;
        self.ensure_resources()?;
        let cube_buffer = self.ensure_cube_buffer()?;
        let resolution = target.extent().width >> mip;

        let render_pass = self.render_pass.as_ref().unwrap();
        let desc_layout = self.descriptor_layout.as_ref().unwrap();

        // 1. Descriptor Set
        let mut desc_alloc = crate::vulkan::descriptor_allocator::DescriptorAllocator::new(
            Arc::clone(&self.device),
            1,
            None,
        )?;
        let desc_set =
            desc_alloc.allocate_static_set(&desc_layout.handle(), desc_layout.bindings())?;
        desc_set.update_image(
            0,
            source_view,
            source_sampler,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        )?;

        // 2. Pipeline
        let pipeline = Pipeline::builder(Arc::clone(&self.device))
            .with_layout(self.pipeline_layout)
            .with_render_pass(render_pass.handle())
            .with_extent(vk::Extent2D {
                width: resolution,
                height: resolution,
            })
            .add_shader_from_bytes(shader_bytes.0, vk::ShaderStageFlags::VERTEX, "main")?
            .add_shader_from_bytes(shader_bytes.1, vk::ShaderStageFlags::FRAGMENT, "main")?
            .with_vertex_input(
                vec![vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: 12,
                    input_rate: vk::VertexInputRate::VERTEX,
                }],
                vec![vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                }],
            )
            .with_cull_mode(vk::CullModeFlags::NONE)
            .build()?;

        // 3. Render to each face
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 10.0);
        let views = [
            Mat4::look_at_rh(Vec3::ZERO, Vec3::X, -Vec3::Y),
            Mat4::look_at_rh(Vec3::ZERO, -Vec3::X, -Vec3::Y),
            Mat4::look_at_rh(Vec3::ZERO, Vec3::Y, Vec3::Z),
            Mat4::look_at_rh(Vec3::ZERO, -Vec3::Y, -Vec3::Z),
            Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, -Vec3::Y),
            Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, -Vec3::Y),
        ];

        for (i, view_mat) in views.iter().enumerate() {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(target.handle())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(target.format())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: i as u32,
                    layer_count: 1,
                });

            let face_view = unsafe { self.device.create_image_view(&view_info, None)? };
            let fb = Framebuffer::new(
                Arc::clone(&self.device),
                render_pass.handle(),
                &[face_view],
                vk::Extent2D {
                    width: resolution,
                    height: resolution,
                },
            )?;

            vulkan_device.execute_single_use(command_pool, |cmd| {
                let clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];

                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: resolution as f32,
                    height: resolution as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: resolution,
                        height: resolution,
                    },
                };

                unsafe {
                    self.device.cmd_set_viewport(cmd, 0, &[viewport]);
                    self.device.cmd_set_scissor(cmd, 0, &[scissor]);

                    let begin_info = vk::RenderPassBeginInfo::default()
                        .render_pass(render_pass.handle())
                        .framebuffer(fb.handle())
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: resolution,
                                height: resolution,
                            },
                        })
                        .clear_values(&clear_values);

                    self.device.cmd_begin_render_pass(
                        cmd,
                        &begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    self.device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[desc_set.handle()],
                        &[],
                    );

                    let push = IblPushConstants {
                        view: view_mat.to_cols_array(),
                        projection: proj.to_cols_array(),
                        roughness,
                        _padding: [0.0; 3],
                    };
                    self.device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );

                    self.device
                        .cmd_bind_vertex_buffers(cmd, 0, &[cube_buffer], &[0]);
                    self.device.cmd_draw(cmd, 36, 1, 0, 0);
                    self.device.cmd_end_render_pass(cmd);
                }
            })?;

            unsafe {
                self.device.destroy_image_view(face_view, None);
            }
        }

        Ok(())
    }

    /// Converts an equirectangular environment map to a cubemap.
    pub fn create_cubemap_from_equirect(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        equirect_view: vk::ImageView,
        equirect_sampler: vk::Sampler,
        resolution: u32,
    ) -> Result<ImageHandle> {
        let env_cubemap = ImageHandle::create_cubemap(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            Some("EnvironmentCubemap".to_string()),
        )?;

        self.render_to_cubemap(
            vulkan_device,
            command_pool,
            IblRenderPassParams {
                target: &env_cubemap,
                source_view: equirect_view,
                source_sampler: equirect_sampler,
                shader_bytes: (
                    include_bytes!(concat!(env!("OUT_DIR"), "/skybox.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/equirect_to_cube.spv")),
                ),
                mip: 0,
                roughness: 0.0,
            },
        )?;

        Ok(env_cubemap)
    }

    /// Generates an irradiance map from an environment cubemap.
    pub fn generate_irradiance(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        env_view: vk::ImageView,
        env_sampler: vk::Sampler,
    ) -> Result<ImageHandle> {
        let resolution = 32;
        let irradiance_map = ImageHandle::create_cubemap(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            Some("IrradianceMap".to_string()),
        )?;

        self.render_to_cubemap(
            vulkan_device,
            command_pool,
            IblRenderPassParams {
                target: &irradiance_map,
                source_view: env_view,
                source_sampler: env_sampler,
                shader_bytes: (
                    include_bytes!(concat!(env!("OUT_DIR"), "/skybox.spv")),
                    include_bytes!(concat!(env!("OUT_DIR"), "/irradiance.spv")),
                ),
                mip: 0,
                roughness: 0.0,
            },
        )?;

        Ok(irradiance_map)
    }

    /// Generates a pre-filtered environment map for specular IBL.
    pub fn generate_prefiltered(
        &mut self,
        vulkan_device: &VulkanDevice,
        command_pool: vk::CommandPool,
        env_view: vk::ImageView,
        env_sampler: vk::Sampler,
    ) -> Result<ImageHandle> {
        let resolution = 128;
        let max_mips = 5;

        let prefiltered_map = ImageHandle::create_cubemap(
            Arc::clone(&self.device),
            Arc::clone(&self.allocator),
            resolution,
            max_mips,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            Some("PrefilteredMap".to_string()),
        )?;

        for mip in 0..max_mips {
            let roughness = mip as f32 / (max_mips - 1) as f32;
            self.render_to_cubemap(
                vulkan_device,
                command_pool,
                IblRenderPassParams {
                    target: &prefiltered_map,
                    source_view: env_view,
                    source_sampler: env_sampler,
                    shader_bytes: (
                        include_bytes!(concat!(env!("OUT_DIR"), "/skybox.spv")),
                        include_bytes!(concat!(env!("OUT_DIR"), "/prefilter.spv")),
                    ),
                    mip,
                    roughness,
                },
            )?;
        }

        Ok(prefiltered_map)
    }

    pub fn destroy(&mut self) {
        self.render_pass = None;
        self.descriptor_layout = None;
        if self.pipeline_layout != vk::PipelineLayout::null() {
            unsafe {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            self.pipeline_layout = vk::PipelineLayout::null();
        }
        self.cube_buffer = None;
    }
}

impl Drop for IblManager {
    fn drop(&mut self) {
        self.destroy();
    }
}
