use crate::renderer::queue::RenderQueue;
use crate::{
    AshError, Result,
    renderer::{initialization, resource_registry::ResourceRegistry},
    vulkan::{self, Allocator},
};
use std::sync::Arc;

/// # Safety
///
/// **DO NOT IMPLEMENT `Drop`**.
/// Manually destroying the `device` here will cause crashes in `alloc` and `resources`
/// which likely get dropped *after* this struct or its fields.
/// Rely on OS cleanup for the Device/Instance.
///
/// Vulkan core infrastructure handles.
/// Grouped to ensure proper LIFO destruction order.
pub struct Context {
    // High-level dependents (Drop FIRST)
    pub resources: Arc<ResourceRegistry>,
    pub queue: RenderQueue,
    pub alloc: Arc<Allocator>,

    // Low-level foundations (Drop LAST)
    pub device: vulkan::VulkanDevice,
    pub instance: Arc<vulkan::VulkanInstance>,
}

impl Context {
    /// Initializes the core Vulkan context.
    pub fn new<S: vulkan::SurfaceProvider>(surface_provider: &S) -> Result<Self> {
        log::info!("Context::new: Creating Vulkan Foundation (Instance, Device, Allocator)");

        unsafe {
            let instance = Arc::new(vulkan::VulkanInstance::new(
                surface_provider,
                cfg!(debug_assertions),
            )?);

            let device =
                vulkan::VulkanDevice::new(Arc::clone(&instance), surface_provider.is_headless())?;

            let alloc = Arc::new(vulkan::Allocator::new(&device)?);
            let resources = Arc::new(ResourceRegistry::new(Arc::clone(&device.device)));

            let queue_data = initialization::init_render_queue(&device)?;
            let queue = queue_data.queue;

            Ok(Self {
                resources,
                queue,
                alloc,
                device,
                instance,
            })
        }
    }

    /// Waits for all GPU operations to complete.
    pub fn wait_for_idle(&self) -> Result<()> {
        unsafe {
            self.device
                .device
                .device_wait_idle()
                .map_err(|e| AshError::VulkanError(e.to_string()))
        }
    }
}
