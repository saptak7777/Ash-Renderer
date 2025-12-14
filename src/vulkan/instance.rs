use ash::{
    ext::{debug_utils, validation_features},
    khr::surface,
    vk, Entry, Instance,
};
use log::{debug, warn};
use std::ffi::CStr;

use crate::{AshError, Result};

/// Vulkan instance wrapper that owns the global instance, optional validation
/// layers, and the window surface.
pub struct VulkanInstance {
    entry: Entry,
    instance: Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    debug_utils: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl VulkanInstance {
    /// Create a new Vulkan instance configured for the provided surface provider.
    pub fn new<S: crate::vulkan::SurfaceProvider>(
        surface_provider: &S,
        enable_validation: bool,
    ) -> Result<Self> {
        unsafe {
            let entry = Entry::load().map_err(|e| {
                AshError::DeviceInitFailed(format!("Failed to load Vulkan entry: {e:?}"))
            })?;

            let validation_layers = if enable_validation {
                Self::query_validation_layers(&entry)?
            } else {
                Vec::new()
            };

            let mut extensions = surface_provider.required_extensions();
            if enable_validation {
                extensions.push(debug_utils::NAME.as_ptr());

                // Check if validation features extension is supported
                let available_extensions = entry
                    .enumerate_instance_extension_properties(None)
                    .map_err(|e| {
                        AshError::DeviceInitFailed(format!(
                            "Failed to enumerate instance extensions: {e:?}"
                        ))
                    })?;

                let validation_features_name = CStr::from_ptr(validation_features::NAME.as_ptr());
                let has_validation_features = available_extensions.iter().any(|ext| {
                    CStr::from_ptr(ext.extension_name.as_ptr()) == validation_features_name
                });

                if has_validation_features {
                    extensions.push(validation_features::NAME.as_ptr());
                } else {
                    warn!("VK_EXT_validation_features not supported, GPU-assisted validation disabled");
                }
            }

            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Ash Renderer")
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(c"Ash Renderer")
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::API_VERSION_1_3);

            let mut create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&validation_layers);

            let mut debug_create_info =
                enable_validation.then_some(Self::debug_messenger_create_info());
            if let Some(ref mut info) = debug_create_info {
                create_info = create_info.push_next(info);
            }

            // Enable GPU-Assisted Validation and Best Practices if validation is requested
            // AND the extension was successfully enabled
            let mut validation_features = vk::ValidationFeaturesEXT::default()
                .enabled_validation_features(&[
                    vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
                    vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
                ]);

            let validation_features_enabled = extensions.iter().any(|&ext| {
                CStr::from_ptr(ext) == CStr::from_ptr(validation_features::NAME.as_ptr())
            });

            if enable_validation && validation_features_enabled {
                create_info = create_info.push_next(&mut validation_features);
            }

            let instance = entry.create_instance(&create_info, None).map_err(|e| {
                AshError::DeviceInitFailed(format!("Failed to create Vulkan instance: {e:?}"))
            })?;

            let debug_utils_loader =
                enable_validation.then(|| debug_utils::Instance::new(&entry, &instance));

            let debug_messenger = if let Some(ref utils) = debug_utils_loader {
                let create_info = Self::debug_messenger_create_info();
                Some(
                    utils
                        .create_debug_utils_messenger(&create_info, None)
                        .map_err(|e| {
                            AshError::DeviceInitFailed(format!(
                                "Failed to create debug messenger: {e:?}"
                            ))
                        })?,
                )
            } else {
                None
            };

            let surface = surface_provider.create_surface(&entry, &instance)?;

            let surface_loader = surface::Instance::new(&entry, &instance);

            Ok(Self {
                entry,
                instance,
                surface_loader,
                surface,
                debug_utils: debug_utils_loader,
                debug_messenger,
            })
        }
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface_loader(&self) -> &surface::Instance {
        &self.surface_loader
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    fn query_validation_layers(entry: &Entry) -> Result<Vec<*const i8>> {
        unsafe {
            let available_layers = entry.enumerate_instance_layer_properties().map_err(|e| {
                AshError::DeviceInitFailed(format!(
                    "Failed to enumerate instance layer properties: {e:?}"
                ))
            })?;

            let desired = [c"VK_LAYER_KHRONOS_VALIDATION".as_ptr()];
            let mut enabled = Vec::new();

            for &layer_name in &desired {
                let desired_name = CStr::from_ptr(layer_name);
                let found = available_layers
                    .iter()
                    .any(|layer| CStr::from_ptr(layer.layer_name.as_ptr()) == desired_name);

                if found {
                    enabled.push(layer_name);
                } else {
                    warn!("Validation layer {desired_name:?} not available");
                }
            }

            Ok(enabled)
        }
    }

    fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger) {
                utils.destroy_debug_utils_messenger(messenger, None);
            }

            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
                self.surface = vk::SurfaceKHR::null();
            }

            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = if !callback_data.is_null() {
        CStr::from_ptr((*callback_data).p_message)
            .to_string_lossy()
            .into_owned()
    } else {
        String::from("<null>")
    };

    debug!(
        target: "vulkan",
        "[{message_types:?}][{message_severity:?}] {message}"
    );

    vk::FALSE
}
