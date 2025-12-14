//! Error types for the ASH Renderer.
//!
//! This module provides a unified error type [`AshError`] and a convenient [`Result`] alias.

use std::fmt;

/// Main error type for the renderer.
///
/// All fallible operations in the renderer return this error type, providing
/// detailed context about what went wrong.
#[derive(Debug)]
pub enum AshError {
    /// A Vulkan API call failed.
    VulkanError(String),
    /// An I/O operation failed (file loading, etc.).
    IoError(std::io::Error),
    /// Device initialization failed.
    DeviceInitFailed(String),
    /// Swapchain creation failed.
    SwapchainCreationFailed(String),
    /// Failed to acquire next swapchain image.
    FrameAcquisitionFailed(String),
    /// Swapchain is out of date (window resized).
    SwapchainOutOfDate(String),
    /// Resource not found in registry.
    ResourceNotFound(String),
    /// Feature not initialized.
    FeatureNotInitialized(String),
}

impl fmt::Display for AshError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::VulkanError(msg) => write!(f, "Vulkan error: {msg}"),
            Self::IoError(err) => write!(f, "IO error: {err}"),
            Self::DeviceInitFailed(msg) => write!(f, "Device init failed: {msg}"),
            Self::SwapchainCreationFailed(msg) => write!(f, "Swapchain creation failed: {msg}"),
            Self::FrameAcquisitionFailed(msg) => write!(f, "Frame acquisition failed: {msg}"),
            Self::SwapchainOutOfDate(msg) => write!(f, "Swapchain out of date: {msg}"),
            Self::ResourceNotFound(msg) => write!(f, "Resource not found: {msg}"),
            Self::FeatureNotInitialized(msg) => write!(f, "Feature not initialized: {msg}"),
        }
    }
}

impl std::error::Error for AshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(err) => Some(err),
            _ => None,
        }
    }
}

/// Convenient Result type alias for renderer operations.
pub type Result<T> = std::result::Result<T, AshError>;

impl From<std::io::Error> for AshError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<ash::vk::Result> for AshError {
    fn from(result: ash::vk::Result) -> Self {
        Self::VulkanError(format!("{result:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AshError::VulkanError("test".to_string());
        assert!(err.to_string().contains("Vulkan error"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AshError = io_err.into();
        assert!(matches!(err, AshError::IoError(_)));
    }
}
