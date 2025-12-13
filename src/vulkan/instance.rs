use ash::{ext::debug_utils, khr::surface, vk, Entry, Instance};
use log::{debug, warn};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::ffi::CStr;
use winit::window::Window;

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
    /// Create a new Vulkan instance configured for the provided window.
    pub fn new(window: &Window, enable_validation: bool) -> Result<Self> {
        unsafe {
            let entry = Entry::load().map_err(|e| {
                AshError::DeviceInitFailed(format!("Failed to load Vulkan entry: {e:?}"))
            })?;

            let validation_layers = if enable_validation {
                Self::query_validation_layers(&entry)?
            } else {
                Vec::new()
            };

            let extensions = Self::required_extensions(enable_validation);

            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Ash Renderer")
                .application_version(vk::make_api_version(0, 0, 1, 0))
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

            let surface = create_surface(&entry, &instance, window)?;

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

    /// Create a new Vulkan instance using a SurfaceProvider.
    /// This is the preferred method for new code as it decouples windowing.
    pub fn from_surface<S: super::surface::SurfaceProvider>(
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

            // Combine standard extensions with surface provider's requirements
            let mut extensions = Self::required_extensions(enable_validation);
            for ext in surface_provider.required_extensions() {
                if !extensions.contains(&ext) {
                    extensions.push(ext);
                }
            }

            let app_info = vk::ApplicationInfo::default()
                .application_name(c"Ash Renderer")
                .application_version(vk::make_api_version(0, 0, 1, 0))
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

            // Create surface using the provider
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

    fn required_extensions(enable_validation: bool) -> Vec<*const i8> {
        let mut extensions = vec![surface::NAME.as_ptr()];

        #[cfg(target_os = "windows")]
        {
            extensions.push(ash::khr::win32_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "linux")]
        {
            extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "macos")]
        {
            extensions.push(ash::ext::metal_surface::NAME.as_ptr());
        }

        if enable_validation {
            extensions.push(debug_utils::NAME.as_ptr());
        }

        extensions
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

#[cfg(target_os = "windows")]
unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR> {
    use ash::khr::win32_surface;

    let win32_surface_loader = win32_surface::Instance::new(entry, instance);

    match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Win32(handle)) => {
            let hwnd = handle.hwnd.get();
            let hinstance = handle.hinstance.map(|h| h.get()).unwrap_or(0);

            let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(hwnd as vk::HWND)
                .hinstance(hinstance as vk::HINSTANCE);

            win32_surface_loader
                .create_win32_surface(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
    }
}

#[cfg(target_os = "linux")]
unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR> {
    use ash::khr::{wayland_surface, xlib_surface};

    match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Wayland(handle)) => {
            let wayland_surface_loader = wayland_surface::Instance::new(entry, instance);
            let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(handle.display.as_ptr())
                .surface(handle.surface.as_ptr());
            wayland_surface_loader
                .create_wayland_surface(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        Ok(RawWindowHandle::Xlib(handle)) => {
            let xlib_surface_loader = xlib_surface::Instance::new(entry, instance);
            let create_info = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(
                    handle
                        .display
                        .map(|d| d.as_ptr())
                        .unwrap_or(std::ptr::null_mut()) as *mut _,
                )
                .window(handle.window);
            xlib_surface_loader
                .create_xlib_surface(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
    }
}

#[cfg(target_os = "macos")]
unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR> {
    use ash::ext::metal_surface;
    use raw_window_handle::RawDisplayHandle;

    let metal_surface_loader = metal_surface::Instance::new(entry, instance);

    match (
        window.window_handle().map(|h| h.as_raw()),
        window.display_handle().map(|h| h.as_raw()),
    ) {
        (Ok(RawWindowHandle::AppKit(handle)), Ok(RawDisplayHandle::AppKit(_display))) => {
            let view = handle.ns_view.as_ptr() as *mut objc::runtime::Object;
            let layer: *mut objc::runtime::Object = objc::msg_send![view, layer];
            let layer_ptr = layer as *const vk::CAMetalLayer;

            let create_info = vk::MetalSurfaceCreateInfoEXT::default().layer(layer_ptr);
            metal_surface_loader
                .create_metal_surface(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
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
