use ash::vk;

pub mod allocator;
pub mod buffer_builder;
pub mod buffer_types;
pub mod command;
pub mod command_manager;
pub mod compute_pipeline;
pub mod deletion_queue;
pub mod descriptor_allocator;
pub mod descriptor_bindless;
pub mod descriptor_layout;
pub mod descriptor_set;
pub mod device;
pub mod framebuffer;
pub mod instance;
pub mod light_culling_pipeline;
#[cfg(feature = "parallel")]
pub mod parallel_command;
pub mod parallel_command_recorder;
pub mod pipeline;
pub mod pipeline_layout;
pub mod pipeline_state;
pub mod renderpass;
pub mod shader;
pub mod surface_provider;
pub mod swapchain;
pub mod sync;
pub mod transfer_context;
pub mod utils;

pub use allocator::Allocator;
pub use buffer_builder::BufferBuilder;
pub use buffer_types::BufferDescriptor;
pub use command::CommandPool;
pub use command_manager::{CommandBufferContext, CommandBufferManager};
pub use compute_pipeline::{ComputePipeline, ComputePipelineBuilder};
pub use descriptor_allocator::DescriptorAllocator;
pub use descriptor_bindless::{BindlessConfig, BindlessManager};
pub use descriptor_layout::{DescriptorSetLayout, DescriptorSetLayoutBuilder};
pub use descriptor_set::DescriptorSet;
pub use device::VulkanDevice;
pub use framebuffer::Framebuffer;
pub use instance::VulkanInstance;
pub use parallel_command_recorder::ParallelCommandRecorder;
pub use pipeline::{MultisampleConfig, Pipeline, PipelineBuilder};
pub use pipeline_layout::{PipelineLayout, PipelineLayoutBuilder};
pub use pipeline_state::PipelineState;
pub use renderpass::{RenderPass, RenderPassBuilder};
pub use shader::{ShaderModule, ShaderReflection};
pub use surface_provider::{HeadlessSurfaceProvider, SurfaceProvider, WindowSurfaceProvider};
pub use swapchain::SwapchainWrapper;
pub use sync::FrameSync;
pub use transfer_context::TransferContext;

#[inline]
pub fn set_debug_object_name<T: vk::Handle>(
    loader: Option<&ash::ext::debug_utils::Device>,
    handle: T,
    object_type: vk::ObjectType,
    name: &str,
) {
    #[cfg(debug_assertions)]
    if let Some(loader) = loader {
        let c_name = std::ffi::CString::new(name).unwrap_or_default();
        let info = vk::DebugUtilsObjectNameInfoEXT {
            object_type,
            object_handle: vk::Handle::as_raw(handle),
            p_object_name: c_name.as_ptr(),
            ..Default::default()
        };
        unsafe {
            let _ = loader.set_debug_utils_object_name(&info);
        }
    }
}
