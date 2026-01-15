use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

/// Handle to a render graph resource (image or buffer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle(u32);

impl ResourceHandle {
    fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Resource access mode for dependency tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAccess {
    Read,
    Write,
}

/// Resource usage information for barrier generation.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub handle: ResourceHandle,
    pub access: ResourceAccess,
    pub image: Option<vk::Image>,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
}

/// A single pass in the render graph.
pub struct PassNode {
    name: String,
    reads: Vec<ResourceUsage>,
    writes: Vec<ResourceUsage>,
    execute: Box<dyn Fn(vk::CommandBuffer) + Send + Sync>,
}

impl PassNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            execute: Box::new(|_| {}),
        }
    }

    pub fn read(mut self, usage: ResourceUsage) -> Self {
        self.reads.push(usage);
        self
    }

    pub fn write(mut self, usage: ResourceUsage) -> Self {
        self.writes.push(usage);
        self
    }

    pub fn execute<F>(mut self, f: F) -> Self
    where
        F: Fn(vk::CommandBuffer) + Send + Sync + 'static,
    {
        self.execute = Box::new(f);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn reads(&self) -> &[ResourceUsage] {
        &self.reads
    }

    pub fn writes(&self) -> &[ResourceUsage] {
        &self.writes
    }
}

/// Simple render graph for automatic barrier injection.
///
/// This is a minimal implementation focused on correctness, not performance.
/// It executes passes linearly and injects barriers between dependent passes.
pub struct RenderGraph {
    device: Arc<ash::Device>,
    passes: Vec<PassNode>,
    resource_counter: u32,
    resource_states: HashMap<ResourceHandle, (vk::ImageLayout, vk::AccessFlags)>,
}

impl RenderGraph {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            passes: Vec::new(),
            resource_counter: 0,
            resource_states: HashMap::new(),
        }
    }

    /// Registers a new resource and returns its handle.
    pub fn register_resource(&mut self) -> ResourceHandle {
        let handle = ResourceHandle::new(self.resource_counter);
        self.resource_counter += 1;
        handle
    }

    /// Adds a pass to the graph.
    pub fn add_pass(&mut self, pass: PassNode) {
        log::debug!("RenderGraph: Adding pass '{}'", pass.name());
        self.passes.push(pass);
    }

    /// Executes the graph, automatically injecting barriers.
    ///
    /// # Safety
    ///
    /// The command buffer must be in the recording state.
    pub unsafe fn execute(&mut self, cmd: vk::CommandBuffer) -> crate::Result<()> {
        log::info!("RenderGraph: Executing {} passes", self.passes.len());

        for (i, pass) in self.passes.iter().enumerate() {
            log::debug!("RenderGraph: Executing pass '{}'", pass.name());

            // Inject barriers before this pass
            if i > 0 {
                self.inject_barriers(cmd, pass)?;
            }

            // Execute the pass
            (pass.execute)(cmd);

            // Update resource states
            for write in &pass.writes {
                self.resource_states.insert(
                    write.handle,
                    (write.new_layout, vk::AccessFlags::SHADER_WRITE),
                );
            }
        }

        Ok(())
    }

    /// Injects barriers for resources accessed by this pass.
    unsafe fn inject_barriers(&self, cmd: vk::CommandBuffer, pass: &PassNode) -> crate::Result<()> {
        let mut barriers = Vec::new();

        // Generate barriers for read dependencies
        for read in &pass.reads {
            if let Some(image) = read.image {
                if let Some(&(old_layout, src_access)) = self.resource_states.get(&read.handle) {
                    if old_layout != read.old_layout {
                        barriers.push(
                            vk::ImageMemoryBarrier::default()
                                .src_access_mask(src_access)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                                .old_layout(old_layout)
                                .new_layout(read.old_layout)
                                .image(image)
                                .subresource_range(
                                    vk::ImageSubresourceRange::default()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .level_count(1)
                                        .layer_count(1),
                                ),
                        );
                    }
                }
            }
        }

        // Generate barriers for write dependencies
        for write in &pass.writes {
            if let Some(image) = write.image {
                barriers.push(
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_READ)
                        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .old_layout(write.old_layout)
                        .new_layout(write.new_layout)
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                );
            }
        }

        if !barriers.is_empty() {
            log::debug!(
                "RenderGraph: Injecting {} barriers for pass '{}'",
                barriers.len(),
                pass.name()
            );
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        Ok(())
    }

    /// Clears all passes (for next frame).
    pub fn clear(&mut self) {
        self.passes.clear();
        self.resource_states.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_handle_creation() {
        let h1 = ResourceHandle::new(0);
        let h2 = ResourceHandle::new(1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_pass_builder() {
        let pass = PassNode::new("TestPass")
            .read(ResourceUsage {
                handle: ResourceHandle::new(0),
                access: ResourceAccess::Read,
                image: None,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .execute(|_cmd| {
                // Test execution
            });

        assert_eq!(pass.name(), "TestPass");
        assert_eq!(pass.reads().len(), 1);
    }
}
