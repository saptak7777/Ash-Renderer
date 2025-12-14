use ash::{khr::swapchain, vk, Device};
use std::collections::HashSet;
use std::ffi::CStr;
use std::sync::Arc;

use crate::{AshError, Result};

pub struct VulkanDevice {
    pub instance: Arc<crate::vulkan::VulkanInstance>,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub present_queue_family: u32,
    /// Timestamp period in nanoseconds (for GPU timing queries)
    pub timestamp_period_ns: f32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VulkanDevice {
    /// Create a logical device for the provided Vulkan instance.
    pub fn new(instance: Arc<crate::vulkan::VulkanInstance>) -> Result<Self> {
        unsafe {
            let vk_instance = instance.instance();

            let physical_devices = vk_instance.enumerate_physical_devices().map_err(|e| {
                AshError::DeviceInitFailed(format!("Failed to enumerate devices: {e:?}"))
            })?;

            if physical_devices.is_empty() {
                return Err(AshError::DeviceInitFailed(
                    "No Vulkan-capable GPU found".to_string(),
                ));
            }

            let mut selected = None;
            for &candidate in &physical_devices {
                if let Some((graphics, present)) = Self::find_queue_families(&instance, candidate) {
                    selected = Some((candidate, graphics, present));
                    break;
                }
            }

            let (physical_device, graphics_queue_family, present_queue_family) = selected
                .ok_or_else(|| {
                    AshError::DeviceInitFailed(
                        "No GPU found with graphics+present support".to_string(),
                    )
                })?;

            let device_properties = vk_instance.get_physical_device_properties(physical_device);
            let memory_properties =
                vk_instance.get_physical_device_memory_properties(physical_device);
            let device_name = CStr::from_ptr(device_properties.device_name.as_ptr());
            let timestamp_period_ns = device_properties.limits.timestamp_period;
            log::info!(
                "Selected GPU: {device_name:?} (timestamp period: {timestamp_period_ns:.3}ns)"
            );

            let queue_priorities = [1.0f32];
            let mut unique_families = HashSet::new();
            unique_families.insert(graphics_queue_family);
            unique_families.insert(present_queue_family);

            let queue_infos: Vec<_> = unique_families
                .iter()
                .map(|family| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*family)
                        .queue_priorities(&queue_priorities)
                })
                .collect();

            let device_extension_names = [swapchain::NAME.as_ptr()];
            let device_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

            let mut vulnerability_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(false)
                .descriptor_indexing(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .runtime_descriptor_array(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_sampled_image_update_after_bind(true);

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .features(device_features)
                .push_next(&mut vulnerability_features);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extension_names)
                .push_next(&mut features2);

            let logical_device = vk_instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| {
                    AshError::DeviceInitFailed(format!("Failed to create device: {e:?}"))
                })?;

            let device = Arc::new(logical_device);
            let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
            let present_queue = device.get_device_queue(present_queue_family, 0);

            Ok(Self {
                instance,
                physical_device,
                device,
                graphics_queue,
                present_queue,
                graphics_queue_family,
                present_queue_family,
                timestamp_period_ns,
                memory_properties,
            })
        }
    }

    fn find_queue_families(
        instance: &Arc<crate::vulkan::VulkanInstance>,
        physical_device: vk::PhysicalDevice,
    ) -> Option<(u32, u32)> {
        let vk_instance = instance.instance();
        let surface_loader = instance.surface_loader();
        let surface = instance.surface();
        let queue_families =
            unsafe { vk_instance.get_physical_device_queue_family_properties(physical_device) };

        let mut graphics_family = None;
        let mut present_family = None;

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(index as u32);
            }

            let present_support = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )
            }
            .unwrap_or(false);

            if present_support {
                present_family = Some(index as u32);
            }

            if graphics_family.is_some() && present_family.is_some() {
                break;
            }
        }

        match (graphics_family, present_family) {
            (Some(graphics), Some(present)) => Some((graphics, present)),
            _ => None,
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            log::info!("Vulkan device destroyed");
        }
    }
}
