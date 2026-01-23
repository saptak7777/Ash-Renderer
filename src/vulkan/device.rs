use ash::{khr::swapchain, vk, Device};
use std::collections::HashSet;
use std::ffi::CStr;
use std::sync::Arc;

use crate::{AshError, Result};

pub struct VulkanDevice {
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub graphics_queue_family: u32,
    pub present_queue_family: u32,
    /// Timestamp period in nanoseconds (for GPU timing queries)
    pub timestamp_period_ns: f32,
    pub sample_rate_shading_supported: bool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub headless: bool,
    pub instance: Arc<crate::vulkan::VulkanInstance>,
}

impl VulkanDevice {
    /// Create a logical device for the provided Vulkan instance.
    pub fn new(instance: Arc<crate::vulkan::VulkanInstance>, headless: bool) -> Result<Self> {
        unsafe {
            let vk_instance = instance.instance();

            let physical_devices = vk_instance.enumerate_physical_devices().map_err(|e| {
                AshError::DeviceInitFailed(format!("Failed to enumerate devices: {e:?}"))
            })?;

            log::info!("Found {} physical device(s)", physical_devices.len());

            if physical_devices.is_empty() {
                return Err(AshError::DeviceInitFailed(
                    "No Vulkan-capable GPU found".to_string(),
                ));
            }

            // Log all available devices
            for (idx, &device) in physical_devices.iter().enumerate() {
                let props = vk_instance.get_physical_device_properties(device);
                let device_name = CStr::from_ptr(props.device_name.as_ptr());
                let device_type = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
                    vk::PhysicalDeviceType::CPU => "CPU",
                    _ => "Other",
                };
                log::info!("  [{}] {device_name:?} ({device_type})", idx);
            }

            let mut selected = None;
            for &candidate in &physical_devices {
                if let Some((graphics, present)) =
                    Self::find_queue_families(&instance, candidate, headless)
                {
                    selected = Some((candidate, graphics, present));
                    break;
                }
            }

            let (physical_device, graphics_queue_family, present_queue_family) = selected
                .ok_or_else(|| {
                    log::error!("No suitable GPU found. All devices were rejected due to missing graphics or present queue support.");
                    AshError::DeviceInitFailed(
                        "No GPU found with graphics+present support".to_string(),
                    )
                })?;

            let device_properties = vk_instance.get_physical_device_properties(physical_device);
            let device_features_supported =
                vk_instance.get_physical_device_features(physical_device);
            let sample_rate_shading_supported =
                device_features_supported.sample_rate_shading == vk::TRUE;

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

            let mut device_extension_names = Vec::new();
            if !headless {
                device_extension_names.push(swapchain::NAME.as_ptr());
            }
            device_extension_names.push(ash::ext::memory_budget::NAME.as_ptr());

            let device_features = vk::PhysicalDeviceFeatures::default()
                .sampler_anisotropy(true)
                .multi_draw_indirect(true)
                .sample_rate_shading(sample_rate_shading_supported);

            let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .descriptor_indexing(true)
                .draw_indirect_count(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .runtime_descriptor_array(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .scalar_block_layout(true); // CRITICAL: Required for BDA vertex pulling with scalar layout

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .features(device_features)
                .push_next(&mut vulkan12_features);

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
                sample_rate_shading_supported,
                memory_properties,
                headless,
            })
        }
    }

    fn find_queue_families(
        instance: &Arc<crate::vulkan::VulkanInstance>,
        physical_device: vk::PhysicalDevice,
        headless: bool,
    ) -> Option<(u32, u32)> {
        let vk_instance = instance.instance();
        let surface_loader = instance.surface_loader();
        let surface = instance.surface();
        let queue_families =
            unsafe { vk_instance.get_physical_device_queue_family_properties(physical_device) };

        // Log device being evaluated
        unsafe {
            let props = vk_instance.get_physical_device_properties(physical_device);
            let device_name = CStr::from_ptr(props.device_name.as_ptr());
            log::debug!("Evaluating queue families for {device_name:?}");
            log::debug!("  Found {} queue families", queue_families.len());
        }

        let mut graphics_family = None;
        let mut present_family = None;

        for (index, queue_family) in queue_families.iter().enumerate() {
            log::debug!(
                "    Family {}: flags={:?}, count={}",
                index,
                queue_family.queue_flags,
                queue_family.queue_count
            );

            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(index as u32);
                log::debug!("      -> Graphics support found");
            }

            if headless {
                // In headless mode, we can always \"present\" to our offscreen images
                // using the graphics queue.
                present_family = graphics_family;
            } else if surface != vk::SurfaceKHR::null() {
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
                    log::debug!("      -> Present support found");
                } else {
                    log::debug!("      -> No present support");
                }
            }

            if graphics_family.is_some() && present_family.is_some() {
                break;
            }
        }

        match (graphics_family, present_family) {
            (Some(graphics), Some(present)) => {
                log::debug!(
                    "  ✓ Device suitable: graphics={}, present={}",
                    graphics,
                    present
                );
                Some((graphics, present))
            }
            _ => {
                log::debug!(
                    "  ✗ Device rejected: graphics={:?}, present={:?}",
                    graphics_family,
                    present_family
                );
                None
            }
        }
    }

    /// Helper to execute a single-use command buffer on the graphics queue.
    pub fn execute_single_use<F>(&self, command_pool: vk::CommandPool, recorder: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer),
    {
        crate::vulkan::utils::execute_single_use(
            &self.device,
            command_pool,
            self.graphics_queue,
            recorder,
        )
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
