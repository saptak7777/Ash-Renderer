use crate::renderer::resources::{
    uniform::{StorageBuffer, UniformBuffer},
    DepthBuffer, Texture,
};
use crate::vulkan;
use ash::vk;

pub struct SwapchainData {
    pub swapchain: vulkan::SwapchainWrapper,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vulkan::Framebuffer>,
    pub depth_buffer: DepthBuffer,
}

pub struct FrameData {
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub frame_syncs: Vec<vulkan::FrameSync>,
    pub command_manager: vulkan::CommandBufferManager,
}

pub struct RendererResources {
    pub uniform_buffers: Vec<UniformBuffer>,
    pub default_texture: Texture,
    pub black_texture: Texture,
    pub white_texture: Texture,
    pub default_skybox: Texture, // Procedural skybox
    pub default_cube_black: Texture,
    pub material_storage_buffer:
        StorageBuffer<crate::renderer::resources::uniform::MaterialUniform>,
    pub instance_buffers: Vec<crate::renderer::resources::InstanceBuffer>,
    pub transform_arena: vk::Buffer,
    pub transform_arena_alloc: vk_mem::Allocation,
    pub post_sampler: vk::Sampler,
}
