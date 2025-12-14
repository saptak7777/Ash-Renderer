use ash::{vk, Device};
use log::{debug, trace};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::renderer::Transform;
use crate::vulkan::DescriptorManager;

pub struct FeatureFrameContext<'a> {
    pub device: &'a Device,
    pub descriptor_manager: Option<&'a DescriptorManager>,
    pub transform: &'a mut Transform,
    pub auto_rotate: bool,
    pub elapsed_seconds: f32,
}

pub struct FeatureRenderContext<'a> {
    pub device: &'a Device,
    pub descriptor_manager: Option<&'a DescriptorManager>,
    pub command_buffer: vk::CommandBuffer,
    pub transform: &'a Transform,
}

pub trait RenderFeature: Send + Any {
    fn name(&self) -> &'static str;
    fn on_added(&mut self, _device: &Device) {}
    fn before_frame(&mut self, _ctx: &mut FeatureFrameContext<'_>) {}
    /// # Safety
    /// The caller must ensure the command buffer in the render context is in the recording state
    /// and remains valid for the duration of the call.
    unsafe fn render(&self, _ctx: &FeatureRenderContext<'_>) {}
    fn on_removed(&mut self, _device: &Device) {}
}

pub struct FeatureManager {
    device: Option<Arc<Device>>,
    features: HashMap<TypeId, Box<dyn RenderFeature>>,
    render_order: Vec<TypeId>,
}

impl FeatureManager {
    pub fn new() -> Self {
        Self {
            device: None,
            features: HashMap::new(),
            render_order: Vec::new(),
        }
    }

    pub fn set_device(&mut self, device: Arc<Device>) {
        if self.device.is_none() {
            self.device = Some(device);
        }
    }

    pub fn add_feature<F: RenderFeature + 'static>(&mut self, mut feature: F) {
        let type_id = TypeId::of::<F>();
        if let Some(device) = &self.device {
            feature.on_added(device);
        }
        trace!("feature added: {}", feature.name());
        self.render_order.push(type_id);
        self.features.insert(type_id, Box::new(feature));
    }

    pub fn before_frame(&mut self, ctx: &mut FeatureFrameContext<'_>) {
        for type_id in &self.render_order {
            if let Some(feature) = self.features.get_mut(type_id) {
                feature.before_frame(ctx);
            }
        }
    }

    /// # Safety
    /// The caller must ensure the command buffer in the provided context is valid and in the
    /// recording state, and that all referenced resources outlive the render pass.
    pub unsafe fn render(&self, ctx: &FeatureRenderContext<'_>) {
        for type_id in &self.render_order {
            if let Some(feature) = self.features.get(type_id) {
                feature.render(ctx);
            }
        }
    }

    pub fn cleanup(&mut self) {
        if let Some(device) = &self.device {
            for type_id in &self.render_order {
                if let Some(feature) = self.features.get_mut(type_id) {
                    feature.on_removed(device);
                    debug!("feature removed: {}", feature.name());
                }
            }
        }
        self.features.clear();
        self.render_order.clear();
    }
}

impl Default for FeatureManager {
    fn default() -> Self {
        Self::new()
    }
}
