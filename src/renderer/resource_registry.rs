use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::Instant;

use ash::{vk, Device};
use log::{error, info, trace, warn};
use thiserror::Error;
use uuid::Uuid;
use vk_mem::Allocation;

use crate::renderer::cleanup_traits::VulkanResourceCleanup;
use crate::vulkan::Allocator;

/// Unique identifier for a tracked Vulkan resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId(Uuid);

impl ResourceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}
impl Default for ResourceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResourceId({})", self.0)
    }
}

/// Errors that can occur while tracking and cleaning up resources.
#[derive(Debug, Error)]
pub enum ResourceError {
    #[error("Resource not found: {0}")]
    NotFound(ResourceId),
    #[error("Resource already exists: {0}")]
    AlreadyExists(ResourceId),
    #[error("Cleanup error for {0}: {1}")]
    CleanupFailed(ResourceId, String),
    #[error("Dependency cycle detected: {0}")]
    DependencyCycle(String),
    #[error("Resource is already cleaned up: {0}")]
    AlreadyCleanedUp(ResourceId),
    #[error("Invalid dependency: {0}")]
    InvalidDependency(String),
}

/// Trait implemented by tracked resources.
pub trait VulkanResource: VulkanResourceCleanup + Send + Sync {
    /// Perform cleanup with the Vulkan device.
    fn cleanup(&mut self, device: &Device) -> Result<(), String> {
        self.cleanup_with_device(device)
    }

    /// Whether this resource has already been cleaned up.
    fn is_cleaned_up(&self) -> bool {
        false
    }

    /// Resource dependencies that must be cleaned first.
    fn dependencies(&self) -> Vec<ResourceId> {
        Vec::new()
    }
}

type ResourceEntry = Arc<RwLock<dyn VulkanResource>>;

/// Dependency-aware resource registry that guarantees cleanup order.
pub struct ResourceRegistry {
    resources: RwLock<HashMap<ResourceId, ResourceEntry>>,
    dependencies: RwLock<HashMap<ResourceId, HashSet<ResourceId>>>,
    reverse_dependencies: RwLock<HashMap<ResourceId, HashSet<ResourceId>>>,
    device: Weak<Device>,
    cleaned_up: AtomicBool,
}

impl ResourceRegistry {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            resources: RwLock::new(HashMap::new()),
            dependencies: RwLock::new(HashMap::new()),
            reverse_dependencies: RwLock::new(HashMap::new()),
            device: Arc::downgrade(&device),
            cleaned_up: AtomicBool::new(false),
        }
    }

    /// Explicitly clean up all resources honoring dependencies.
    pub fn cleanup(&self) -> Result<(), String> {
        // Mark as cleaned up to prevent double cleanup in Drop
        if self.cleaned_up.swap(true, Ordering::SeqCst) {
            trace!("Registry already cleaned up, skipping");
            return Ok(());
        }

        match self.cleanup_all_resources() {
            Ok(()) => Ok(()),
            Err(errors) => {
                if let Some(first) = errors.first() {
                    Err(format!(
                        "{} errors occurred during cleanup. First error: {first}",
                        errors.len()
                    ))
                } else {
                    Err("Unknown cleanup error".to_string())
                }
            }
        }
    }

    /// Register a framebuffer for cleanup.
    pub fn register_framebuffer(
        &self,
        framebuffer: vk::Framebuffer,
        dependencies: &[ResourceId],
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(FramebufferResource::new(framebuffer, dependencies))
    }

    /// Register a render pass for cleanup.
    pub fn register_render_pass(
        &self,
        render_pass: vk::RenderPass,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(RenderPassResource::new(render_pass))
    }

    /// Register a depth buffer (image + view) for cleanup.
    pub fn register_depth_buffer(
        &self,
        image: vk::Image,
        view: vk::ImageView,
        allocation: Allocation,
        allocator: Arc<Allocator>,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(DepthBufferResource::new(image, view, allocation, allocator))
    }

    /// Register an image view for cleanup.
    pub fn register_image_view(
        &self,
        image_view: vk::ImageView,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(ImageViewResource::new(image_view))
    }

    /// Register a command pool for cleanup.
    pub fn register_command_pool(
        &self,
        pool: vk::CommandPool,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(CommandPoolResource::new(pool))
    }

    /// Register a semaphore for cleanup.
    pub fn register_semaphore(
        &self,
        semaphore: vk::Semaphore,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(SemaphoreResource::new(semaphore))
    }

    /// Register a fence for cleanup.
    pub fn register_fence(&self, fence: vk::Fence) -> Result<ResourceId, ResourceError> {
        self.add_resource(FenceResource::new(fence))
    }

    /// Register a pipeline layout.
    pub fn register_pipeline_layout(
        &self,
        layout: vk::PipelineLayout,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(PipelineLayoutResource::new(layout))
    }

    /// Register a pipeline and declare dependencies (e.g., pipeline layout).
    pub fn register_pipeline(
        &self,
        pipeline: vk::Pipeline,
        dependencies: &[ResourceId],
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(PipelineResource::new(pipeline, dependencies))
    }

    /// Register a descriptor pool for cleanup.
    pub fn register_descriptor_pool(
        &self,
        pool: vk::DescriptorPool,
    ) -> Result<ResourceId, ResourceError> {
        self.add_resource(DescriptorPoolResource::new(pool))
    }

    /// Immediately cleans up a specific resource, honoring dependency constraints.
    pub fn cleanup_resource(&self, id: ResourceId) -> Result<(), ResourceError> {
        self.remove_resource(id)
    }

    /// Removes all resources, honoring dependency relationships.
    fn cleanup_all_resources(&self) -> Result<(), Vec<ResourceError>> {
        let start = Instant::now();
        info!("Starting cleanup of tracked resources...");

        let ids: Vec<ResourceId> = match self.resources.read() {
            Ok(resources) => resources.keys().copied().collect(),
            Err(poisoned) => {
                warn!("Resource registry lock poisoned during cleanup; continuing");
                poisoned.into_inner().keys().copied().collect()
            }
        };

        if ids.is_empty() {
            trace!("No resources to clean up");
            return Ok(());
        }

        let order = self.get_cleanup_order().ok();
        let cleanup_ids = order.map(|o| o.into_iter().rev().collect()).unwrap_or(ids);

        let mut errors = Vec::new();
        let mut cleaned = 0;
        for id in cleanup_ids {
            if let Err(err) = self.remove_resource(id) {
                errors.push(err);
            } else {
                cleaned += 1;
            }
        }

        if errors.is_empty() {
            info!("Cleaned up {cleaned} resources in {:.2?}", start.elapsed());
            Ok(())
        } else {
            error!(
                "Cleanup completed with {} errors in {:.2?} ({cleaned} cleaned)",
                errors.len(),
                start.elapsed()
            );
            Err(errors)
        }
    }

    fn add_resource<T: VulkanResource + 'static>(
        &self,
        resource: T,
    ) -> Result<ResourceId, ResourceError> {
        let id = ResourceId::new();
        self.add_resource_with_id(id, resource)
    }

    fn add_resource_with_id<T: VulkanResource + 'static>(
        &self,
        id: ResourceId,
        resource: T,
    ) -> Result<ResourceId, ResourceError> {
        let deps = resource.dependencies();
        if let Some(cycle) = self.detect_cycle(id, &deps) {
            return Err(ResourceError::DependencyCycle(cycle));
        }

        let mut resources = self.resources.write().unwrap();
        if resources.contains_key(&id) {
            return Err(ResourceError::AlreadyExists(id));
        }

        let deps_set: HashSet<_> = deps.into_iter().collect();
        resources.insert(id, Arc::new(RwLock::new(resource)));

        self.dependencies
            .write()
            .unwrap()
            .insert(id, deps_set.clone());
        let mut reverse = self.reverse_dependencies.write().unwrap();
        for dep in deps_set {
            reverse.entry(dep).or_default().insert(id);
        }

        Ok(id)
    }

    fn detect_cycle(&self, new_id: ResourceId, deps: &[ResourceId]) -> Option<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![new_id];

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                return Some(format!("Cycle detected involving {current}"));
            }

            if deps.contains(&current) {
                if let Some(child_deps) = self.dependencies.read().unwrap().get(&current) {
                    stack.extend(child_deps.iter().copied());
                }
            }
        }

        None
    }

    fn remove_resource(&self, id: ResourceId) -> Result<(), ResourceError> {
        let device = self
            .device
            .upgrade()
            .ok_or_else(|| ResourceError::CleanupFailed(id, "Device has been dropped".into()))?;

        if let Some(dependents) = self.reverse_dependencies.read().unwrap().get(&id) {
            if !dependents.is_empty() {
                return Err(ResourceError::InvalidDependency(format!(
                    "Cannot remove resource {id}: {} dependents exist",
                    dependents.len()
                )));
            }
        }

        if let Some(deps) = self.dependencies.write().unwrap().remove(&id) {
            let mut reverse = self.reverse_dependencies.write().unwrap();
            for dep in deps {
                if let Some(entries) = reverse.get_mut(&dep) {
                    entries.remove(&id);
                }
            }
        }

        self.reverse_dependencies.write().unwrap().remove(&id);

        let entry = self
            .resources
            .write()
            .unwrap()
            .remove(&id)
            .ok_or(ResourceError::NotFound(id))?;

        if let Ok(mut resource) = entry.write() {
            if resource.is_cleaned_up() {
                return Err(ResourceError::AlreadyCleanedUp(id));
            }
            resource
                .cleanup(&device)
                .map_err(|e| ResourceError::CleanupFailed(id, e))?
        }

        Ok(())
    }

    fn get_cleanup_order(&self) -> Result<Vec<ResourceId>, ResourceError> {
        let resources: Vec<ResourceId> = self
            .resources
            .read()
            .map_err(|_| {
                ResourceError::CleanupFailed(
                    ResourceId::default(),
                    "Failed to read resources".into(),
                )
            })?
            .keys()
            .copied()
            .collect();

        let mut visited = HashSet::new();
        let mut temp = HashSet::new();
        let mut order = Vec::new();

        for id in resources {
            if !visited.contains(&id) {
                self.visit(id, &mut visited, &mut temp, &mut order)?;
            }
        }

        Ok(order)
    }

    fn visit(
        &self,
        id: ResourceId,
        visited: &mut HashSet<ResourceId>,
        temp: &mut HashSet<ResourceId>,
        order: &mut Vec<ResourceId>,
    ) -> Result<(), ResourceError> {
        if temp.contains(&id) {
            return Err(ResourceError::DependencyCycle(format!(
                "Circular dependency involving {id}"
            )));
        }

        if visited.contains(&id) {
            return Ok(());
        }

        temp.insert(id);
        if let Some(deps) = self.dependencies.read().unwrap().get(&id) {
            for dep in deps.iter().copied() {
                self.visit(dep, visited, temp, order)?;
            }
        }
        temp.remove(&id);
        visited.insert(id);
        order.push(id);
        Ok(())
    }
}

impl Drop for ResourceRegistry {
    fn drop(&mut self) {
        // Skip if already cleaned up via explicit cleanup() call
        if self.cleaned_up.load(Ordering::SeqCst) {
            trace!("Registry already cleaned up; skipping Drop cleanup");
            return;
        }

        if self.device.strong_count() == 0 {
            trace!("Device already dropped; skipping registry cleanup");
            return;
        }

        if let Err(err) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.cleanup_all_resources()
        })) {
            if let Some(msg) = err.downcast_ref::<String>() {
                error!("Panic during resource registry cleanup: {msg}");
            } else if let Some(msg) = err.downcast_ref::<&str>() {
                error!("Panic during resource registry cleanup: {msg}");
            } else {
                error!("Unknown panic during resource registry cleanup");
            }
        }
    }
}

/// Framebuffer resource wrapper.
struct FramebufferResource {
    framebuffer: vk::Framebuffer,
    cleaned: bool,
    deps: Vec<ResourceId>,
}

impl FramebufferResource {
    fn new(framebuffer: vk::Framebuffer, dependencies: &[ResourceId]) -> Self {
        Self {
            framebuffer,
            cleaned: false,
            deps: dependencies.to_vec(),
        }
    }
}

impl VulkanResourceCleanup for FramebufferResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.framebuffer == vk::Framebuffer::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "Framebuffer"
    }
}

impl VulkanResource for FramebufferResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }

    fn dependencies(&self) -> Vec<ResourceId> {
        self.deps.clone()
    }
}

/// Depth buffer resource wrapper (image + view + allocation).
struct DepthBufferResource {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    allocator: Arc<Allocator>,
    cleaned: bool,
}

impl DepthBufferResource {
    fn new(
        image: vk::Image,
        view: vk::ImageView,
        allocation: Allocation,
        allocator: Arc<Allocator>,
    ) -> Self {
        Self {
            image,
            view,
            allocation: Some(allocation),
            allocator,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for DepthBufferResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned {
            return Ok(());
        }

        unsafe {
            if self.view != vk::ImageView::null() {
                device.destroy_image_view(self.view, None);
            }
        }

        if let Some(mut allocation) = self.allocation.take() {
            unsafe {
                self.allocator
                    .vma
                    .destroy_image(self.image, &mut allocation);
            }
        }

        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "DepthBuffer"
    }
}

impl VulkanResource for DepthBufferResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Descriptor pool resource wrapper.
struct DescriptorPoolResource {
    pool: vk::DescriptorPool,
    cleaned: bool,
}

impl DescriptorPoolResource {
    fn new(pool: vk::DescriptorPool) -> Self {
        Self {
            pool,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for DescriptorPoolResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.pool == vk::DescriptorPool::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "DescriptorPool"
    }
}

impl VulkanResource for DescriptorPoolResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Image view resource wrapper.
struct ImageViewResource {
    view: vk::ImageView,
    cleaned: bool,
}

impl ImageViewResource {
    fn new(view: vk::ImageView) -> Self {
        Self {
            view,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for ImageViewResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.view == vk::ImageView::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_image_view(self.view, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "ImageView"
    }
}

impl VulkanResource for ImageViewResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Render pass resource wrapper.
struct RenderPassResource {
    render_pass: vk::RenderPass,
    cleaned: bool,
}

impl RenderPassResource {
    fn new(render_pass: vk::RenderPass) -> Self {
        Self {
            render_pass,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for RenderPassResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.render_pass == vk::RenderPass::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_render_pass(self.render_pass, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "RenderPass"
    }
}

impl VulkanResource for RenderPassResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Command pool resource wrapper.
struct CommandPoolResource {
    pool: vk::CommandPool,
    cleaned: bool,
}

impl CommandPoolResource {
    fn new(pool: vk::CommandPool) -> Self {
        Self {
            pool,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for CommandPoolResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.pool == vk::CommandPool::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "CommandPool"
    }
}

impl VulkanResource for CommandPoolResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Semaphore resource wrapper.
struct SemaphoreResource {
    semaphore: vk::Semaphore,
    cleaned: bool,
}

impl SemaphoreResource {
    fn new(semaphore: vk::Semaphore) -> Self {
        Self {
            semaphore,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for SemaphoreResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.semaphore == vk::Semaphore::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_semaphore(self.semaphore, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "Semaphore"
    }
}

impl VulkanResource for SemaphoreResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Fence resource wrapper.
struct FenceResource {
    fence: vk::Fence,
    cleaned: bool,
}

impl FenceResource {
    fn new(fence: vk::Fence) -> Self {
        Self {
            fence,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for FenceResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.fence == vk::Fence::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_fence(self.fence, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "Fence"
    }
}

impl VulkanResource for FenceResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Pipeline layout resource wrapper.
struct PipelineLayoutResource {
    layout: vk::PipelineLayout,
    cleaned: bool,
}

impl PipelineLayoutResource {
    fn new(layout: vk::PipelineLayout) -> Self {
        Self {
            layout,
            cleaned: false,
        }
    }
}

impl VulkanResourceCleanup for PipelineLayoutResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.layout == vk::PipelineLayout::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "PipelineLayout"
    }
}

impl VulkanResource for PipelineLayoutResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }
}

/// Pipeline resource wrapper with dependencies.
struct PipelineResource {
    pipeline: vk::Pipeline,
    cleaned: bool,
    deps: Vec<ResourceId>,
}

impl PipelineResource {
    fn new(pipeline: vk::Pipeline, deps: &[ResourceId]) -> Self {
        Self {
            pipeline,
            cleaned: false,
            deps: deps.to_vec(),
        }
    }
}

impl VulkanResourceCleanup for PipelineResource {
    fn cleanup_with_device(&mut self, device: &Device) -> Result<(), String> {
        if self.cleaned || self.pipeline == vk::Pipeline::null() {
            return Ok(());
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
        self.cleaned = true;
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "Pipeline"
    }
}

impl VulkanResource for PipelineResource {
    fn is_cleaned_up(&self) -> bool {
        self.cleaned
    }

    fn dependencies(&self) -> Vec<ResourceId> {
        self.deps.clone()
    }
}
