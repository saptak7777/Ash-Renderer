use ash::{
    extensions::{ext::DebugUtils, khr::Surface},
    vk, Entry, Instance,
};
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
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    debug_utils: Option<DebugUtils>,
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

            let app_info = vk::ApplicationInfo::builder()
                .application_name(c"REDE")
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(c"REDE")
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::API_VERSION_1_3);

            let mut create_info_builder = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&extensions)
                .enabled_layer_names(&validation_layers);

            let mut debug_create_info =
                enable_validation.then_some(Self::debug_messenger_create_info());
            if let Some(ref mut info) = debug_create_info {
                create_info_builder = create_info_builder.push_next(info);
            }

            let instance_create_info = create_info_builder.build();

            let instance = entry
                .create_instance(&instance_create_info, None)
                .map_err(|e| {
                    AshError::DeviceInitFailed(format!("Failed to create Vulkan instance: {e:?}"))
                })?;

            let debug_utils = enable_validation.then(|| DebugUtils::new(&entry, &instance));

            let debug_messenger = if let Some(ref utils) = debug_utils {
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

            let surface_loader = Surface::new(&entry, &instance);

            Ok(Self {
                entry,
                instance,
                surface_loader,
                surface,
                debug_utils,
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

    pub fn surface_loader(&self) -> &Surface {
        &self.surface_loader
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    fn required_extensions(enable_validation: bool) -> Vec<*const i8> {
        let mut extensions = vec![Surface::name().as_ptr()];

        #[cfg(target_os = "windows")]
        {
            extensions.push(ash::extensions::khr::Win32Surface::name().as_ptr());
        }

        #[cfg(target_os = "linux")]
        {
            extensions.push(ash::extensions::khr::XlibSurface::name().as_ptr());
        }

        #[cfg(target_os = "macos")]
        {
            extensions.push(ash::extensions::ext::MetalSurface::name().as_ptr());
        }

        if enable_validation {
            extensions.push(DebugUtils::name().as_ptr());
        }

        extensions
    }

    fn query_validation_layers(entry: &Entry) -> Result<Vec<*const i8>> {
        let available_layers = entry.enumerate_instance_layer_properties().map_err(|e| {
            AshError::DeviceInitFailed(format!(
                "Failed to enumerate instance layer properties: {e:?}"
            ))
        })?;

        let desired = [c"VK_LAYER_KHRONOS_VALIDATION".as_ptr()];
        let mut enabled = Vec::new();

        for &layer_name in &desired {
            let desired_name = unsafe { CStr::from_ptr(layer_name) };
            let found = available_layers
                .iter()
                .any(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) == desired_name });

            if found {
                enabled.push(layer_name);
            } else {
                warn!("Validation layer {desired_name:?} not available");
            }
        }

        Ok(enabled)
    }

    fn debug_messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
        vk::DebugUtilsMessengerCreateInfoEXT::builder()
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
            .build()
    }
}

#[cfg(target_os = "windows")]
unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR> {
    use ash::extensions::khr::Win32Surface;

    let win32_surface = Win32Surface::new(entry, instance);

    match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Win32(handle)) => {
            let hwnd = handle.hwnd.get() as *mut _;
            let hinstance = handle
                .hinstance
                .map(|h| h.get() as *mut _)
                .unwrap_or(std::ptr::null_mut());

            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hwnd(hwnd)
                .hinstance(hinstance);

            win32_surface
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
    use ash::extensions::khr::{WaylandSurface, XlibSurface};

    match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Wayland(handle)) => {
            let wayland_surface = WaylandSurface::new(entry, instance);
            let create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
                .display(handle.display)
                .surface(handle.surface);
            wayland_surface
                .create_wayland_surface(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        Ok(RawWindowHandle::Xlib(handle)) => {
            let xlib_surface = XlibSurface::new(entry, instance);
            let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                .dpy(handle.display as *mut _)
                .window(handle.window);
            xlib_surface
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
    use ash::extensions::ext::MetalSurface;
    use raw_window_handle::RawDisplayHandle;

    let metal_surface = MetalSurface::new(entry, instance);

    match (
        window.window_handle().map(|h| h.as_raw()),
        window.display_handle().map(|h| h.as_raw()),
    ) {
        (Ok(RawWindowHandle::AppKit(handle)), Ok(RawDisplayHandle::AppKit(_display))) => {
            let view = handle.ns_view.as_ptr() as *mut objc::runtime::Object;
            let layer: *mut objc::runtime::Object = objc::msg_send![view, layer];
            let layer_ptr = layer as *const vk::CAMetalLayer;

            let create_info = vk::MetalSurfaceCreateInfoEXT::builder().layer(layer_ptr);
            metal_surface
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
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
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
