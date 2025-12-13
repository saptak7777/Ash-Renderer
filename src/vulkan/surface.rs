//! Surface provider trait for decoupling windowing from renderer initialization.
//!
//! This enables:
//! - Headless rendering for CI/benchmarks
//! - Platform-agnostic surface creation
//! - Testing without a display

use ash::{khr::surface, vk, Entry, Instance};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use raw_window_handle::RawDisplayHandle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

use crate::{AshError, Result};

/// Trait for providing a Vulkan surface to the renderer.
///
/// Implementations can provide surfaces from various sources:
/// - Window systems (winit, SDL, GLFW)
/// - Headless/offscreen rendering
/// - Virtual displays for testing
pub trait SurfaceProvider {
    /// Create a Vulkan surface using the provided entry and instance.
    ///
    /// # Safety
    /// The caller must ensure the entry and instance are valid.
    unsafe fn create_surface(&self, entry: &Entry, instance: &Instance) -> Result<vk::SurfaceKHR>;

    /// Get the current extent (size) of the surface.
    fn extent(&self) -> vk::Extent2D;

    /// Get required instance extensions for this surface type.
    fn required_extensions(&self) -> Vec<*const i8> {
        let mut extensions = vec![surface::NAME.as_ptr()];

        #[cfg(target_os = "windows")]
        extensions.push(ash::khr::win32_surface::NAME.as_ptr());

        #[cfg(target_os = "linux")]
        {
            extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
            extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "macos")]
        extensions.push(ash::ext::metal_surface::NAME.as_ptr());

        extensions
    }
}

/// Window-based surface provider wrapping any type that implements
/// `HasWindowHandle` and `HasDisplayHandle` (e.g., winit::Window).
pub struct WindowSurfaceProvider<W> {
    window: W,
    width: u32,
    height: u32,
}

impl<W> WindowSurfaceProvider<W> {
    /// Create a new window surface provider.
    pub fn new(window: W, width: u32, height: u32) -> Self {
        Self {
            window,
            width,
            height,
        }
    }

    /// Update the extent (e.g., after window resize).
    pub fn set_extent(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Get a reference to the underlying window.
    pub fn window(&self) -> &W {
        &self.window
    }
}

impl<W: HasWindowHandle + HasDisplayHandle> SurfaceProvider for WindowSurfaceProvider<W> {
    unsafe fn create_surface(&self, entry: &Entry, instance: &Instance) -> Result<vk::SurfaceKHR> {
        create_surface_from_handles(entry, instance, &self.window)
    }

    fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

/// Headless surface provider for CI/benchmarks (no actual surface).
///
/// This creates a null surface, suitable for compute-only workloads
/// or when you need to initialize Vulkan without a display.
pub struct HeadlessSurfaceProvider {
    width: u32,
    height: u32,
}

impl HeadlessSurfaceProvider {
    /// Create a new headless surface provider with the given virtual extent.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl SurfaceProvider for HeadlessSurfaceProvider {
    unsafe fn create_surface(
        &self,
        _entry: &Entry,
        _instance: &Instance,
    ) -> Result<vk::SurfaceKHR> {
        // Return null surface - caller must handle compute-only path
        Ok(vk::SurfaceKHR::null())
    }

    fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }

    fn required_extensions(&self) -> Vec<*const i8> {
        // Headless doesn't need surface extensions
        Vec::new()
    }
}

// Platform-specific surface creation from raw handles
#[cfg(target_os = "windows")]
unsafe fn create_surface_from_handles<W: HasWindowHandle>(
    entry: &Entry,
    instance: &Instance,
    window: &W,
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
unsafe fn create_surface_from_handles<W: HasWindowHandle + HasDisplayHandle>(
    entry: &Entry,
    instance: &Instance,
    window: &W,
) -> Result<vk::SurfaceKHR> {
    use ash::khr::{wayland_surface, xlib_surface};

    match window.window_handle().map(|h| h.as_raw()) {
        Ok(RawWindowHandle::Wayland(handle)) => {
            let display_handle = window.display_handle().map_err(|e| {
                AshError::DeviceInitFailed(format!("Failed to get display handle: {e:?}"))
            })?;
            let display = match display_handle.as_raw() {
                RawDisplayHandle::Wayland(d) => d.display.as_ptr(),
                _ => {
                    return Err(AshError::DeviceInitFailed(
                        "Invalid display handle".to_string(),
                    ))
                }
            };

            let wayland_surface_loader = wayland_surface::Instance::new(entry, instance);
            let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display)
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
unsafe fn create_surface_from_handles<W: HasWindowHandle + HasDisplayHandle>(
    entry: &Entry,
    instance: &Instance,
    window: &W,
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
            metal_surface_loader
                .create_metal_surface(&create_info, None)
                .map_err(|e| AshError::VulkanError(format!("{e:?}")))
        }
        _ => Err(AshError::DeviceInitFailed(
            "Invalid window handle".to_string(),
        )),
    }
}

// Fallback for other platforms
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
unsafe fn create_surface_from_handles<W: HasWindowHandle + HasDisplayHandle>(
    _entry: &Entry,
    _instance: &Instance,
    _window: &W,
) -> Result<vk::SurfaceKHR> {
    Err(AshError::DeviceInitFailed(
        "Platform not supported for surface creation".to_string(),
    ))
}
