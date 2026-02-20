use crate::renderer::resource_registry::ResourceId;
use crate::renderer::resources::{
    uniform::{StorageBuffer, UniformBuffer},
    DepthBuffer, Texture,
};
use crate::vulkan;
use ash::vk;
use std::sync::Arc;

pub struct SwapchainData {
    pub swapchain: vulkan::SwapchainWrapper,
    pub depth_buffer: DepthBuffer,
}

pub struct SwapchainDataWithIds {
    pub data: SwapchainData,
    pub swapchain_image_view_ids: Vec<ResourceId>,
    pub depth_buffer_id: ResourceId,
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
    pub dummy_black_cube: Texture,
    pub dummy_black_2d: Texture,
    pub material_storage_buffer:
        StorageBuffer<crate::renderer::resources::uniform::MaterialUniform>,
    pub instance_buffers: Vec<crate::renderer::resources::InstanceBuffer>,
    pub post_sampler: vk::Sampler,
}

pub struct CoreInfrastructure {
    pub buffer_pool: Arc<crate::renderer::resources::BufferPool>,
    pub geometry_buffer: Arc<crate::renderer::resources::DualHeapGeometryBuffer>,
    pub model_renderer: crate::renderer::model_renderer::ModelRenderer,
    pub bindless_manager: crate::vulkan::BindlessManager,
    pub descriptor_allocator: vulkan::DescriptorAllocator,
    pub renderer_resources: RendererResources,
}

pub struct PipelineData {
    pub layout: vulkan::PipelineLayout,
    pub layout_id: ResourceId,
    pub pipeline: vulkan::Pipeline,
    pub pipeline_id: ResourceId,
}

pub struct RenderingPasses {
    pub gbuffer: Option<crate::renderer::GBuffer>,
    pub gbuffer_indices: crate::renderer::types::GBufferIndices,
    pub hiz_pass: Option<crate::renderer::passes::hiz::HiZPass>,
    pub indirect_draw_pass: Option<crate::renderer::vcgs::IndirectDrawPass>,
    pub skybox_pass: Option<crate::renderer::passes::SkyboxPass>,
}

pub struct LightingSystem {
    pub forward_plus: crate::renderer::ForwardPlusIntegration,
    pub global_cluster_buffer: Arc<crate::renderer::resources::GlobalClusterBuffer>,
}

pub struct RenderQueueData {
    pub queue: crate::renderer::queue::RenderQueue,
}

pub struct PostProcessingSystem {
    pub post_process: crate::renderer::systems::post_process::PostProcessSystem,
}
