use ash::{Entry, Instance, vk};
#[cfg(target_os = "macos")]
use raw_window_handle::{HasDisplayHandle, RawDisplayHandle};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

use crate::{AshError, Result};

/// Trait for platform-agnostic surface creation.
/// This allows the renderer to run on windows, off-screen (headless), or other custom surfaces.
pub trait SurfaceProvider {
    /// Create a Vulkan surface for the instance.
    ///
    /// # Safety
    /// The surface must be valid for the lifetime of the instance.
    unsafe fn create_surface(&self, entry: &Entry, instance: &Instance) -> Result<vk::SurfaceKHR>;

    /// Get required instance extensions for this surface type.
    fn required_extensions(&self) -> Vec<*const i8>;

    /// Get the physical size of the surface in pixels.
    fn physical_size(&self) -> (u32, u32);

    /// Returns true if the surface is headless (off-screen).
    fn is_headless(&self) -> bool {
        false
    }
}

/// Headless surface provider for off-screen rendering
pub struct HeadlessSurfaceProvider {
    width: u32,
    height: u32,
}

impl HeadlessSurfaceProvider {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl SurfaceProvider for HeadlessSurfaceProvider {
    fn required_extensions(&self) -> Vec<*const i8> {
        vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::ext::headless_surface::NAME.as_ptr(),
        ]
    }

    unsafe fn create_surface(&self, entry: &Entry, instance: &Instance) -> Result<vk::SurfaceKHR> {
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(|e| {
            AshError::DeviceInitFailed(format!("Failed to enumerate instance extensions: {e:?}"))
        })?;

        let has_headless = available_extensions.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == ash::ext::headless_surface::NAME
        });

        if !has_headless {
            log::warn!(
                "VK_EXT_headless_surface not available, falling back to surface-less operation."
            );
            return Ok(vk::SurfaceKHR::null());
        }

        let headless_loader = ash::ext::headless_surface::Instance::new(entry, instance);
        let create_info = vk::HeadlessSurfaceCreateInfoEXT::default();

        unsafe { headless_loader.create_headless_surface(&create_info, None) }.map_err(|e| {
            AshError::DeviceInitFailed(format!("Failed to create headless surface: {e:?}"))
        })
    }

    fn physical_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn is_headless(&self) -> bool {
        true
    }
}

/// Standard window-based surface provider using `winit`.
pub struct WindowSurfaceProvider<'a> {
    window: &'a Window,
}

impl<'a> WindowSurfaceProvider<'a> {
    pub fn new(window: &'a Window) -> Self {
        Self { window }
    }
}

impl<'a> SurfaceProvider for WindowSurfaceProvider<'a> {
    fn required_extensions(&self) -> Vec<*const i8> {
        let mut extensions = vec![ash::khr::surface::NAME.as_ptr()];

        #[cfg(target_os = "windows")]
        {
            extensions.push(ash::khr::win32_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "linux")]
        {
            extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
            extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "macos")]
        {
            extensions.push(ash::ext::metal_surface::NAME.as_ptr());
        }

        extensions
    }

    unsafe fn create_surface(&self, entry: &Entry, instance: &Instance) -> Result<vk::SurfaceKHR> {
        unsafe { create_surface_impl(entry, instance, self.window) }
    }

    fn physical_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }
}

#[cfg(target_os = "windows")]
unsafe fn create_surface_impl(
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

            unsafe { win32_surface_loader.create_win32_surface(&create_info, None) }
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
    }
}

#[cfg(target_os = "linux")]
unsafe fn create_surface_impl(
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
            unsafe { wayland_surface_loader.create_wayland_surface(&create_info, None) }
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
            unsafe { xlib_surface_loader.create_xlib_surface(&create_info, None) }
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
    }
}

#[cfg(target_os = "macos")]
unsafe fn create_surface_impl(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR> {
    use ash::ext::metal_surface;

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
            unsafe { metal_surface_loader.create_metal_surface(&create_info, None) }
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
    }
}
