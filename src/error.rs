//! Error types for the ASH Renderer.
//!
//! This module provides a unified error type [`AshError`] and a convenient [`Result`] alias.

use thiserror::Error;

/// Main error type for the renderer.
///
/// All fallible operations in the renderer return this error type, providing
/// detailed context about what went wrong.
#[derive(Debug, Error)]
pub enum AshError {
    /// A Vulkan API call failed.
    #[error("Vulkan error: {0}")]
    VulkanError(String),
    /// An I/O operation failed (file loading, etc.).
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    /// Device initialization failed.
    #[error("Device init failed: {0}")]
    DeviceInitFailed(String),
    /// Swapchain creation failed.
    #[error("Swapchain creation failed: {0}")]
    SwapchainCreationFailed(String),
    /// Failed to acquire next swapchain image.
    #[error("Frame acquisition failed: {0}")]
    FrameAcquisitionFailed(String),
    /// Swapchain is out of date (window resized).
    #[error("Swapchain out of date: {0}")]
    SwapchainOutOfDate(String),
    /// Resource not found in registry.
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    /// Feature not initialized.
    #[error("Feature not initialized: {0}")]
    FeatureNotInitialized(String),
    /// GPU memory budget exceeded.
    #[error("VRAM exhausted: requested {requested} bytes, but only {available} bytes available in budget")]
    VramExhausted {
        requested: ash::vk::DeviceSize,
        available: ash::vk::DeviceSize,
        recommendation: &'static str,
    },
    /// Render pass not found or not initialized.
    #[error("Render pass missing: {0}")]
    RenderPassMissing(String),
    /// Pipeline not found or not initialized.
    #[error("Pipeline missing: {0}")]
    PipelineMissing(String),
    /// Swapchain not initialized.
    #[error("Swapchain missing: {0}")]
    SwapchainMissing(String),
    /// Texture not found or invalid.
    #[error("Texture not found: {0}")]
    TextureNotFound(String),
    /// Failed to bind texture to descriptor set.
    #[error("Texture binding failed: {0}")]
    TextureBindingFailed(String),
    /// Material handle not found in registry.
    #[error("Material not found: {0}")]
    MaterialNotFound(u32),
    /// Mesh handle not found in registry.
    #[error("Mesh not found: {0}")]
    MeshNotFound(u32),
    /// Resource registration failed.
    #[error("Resource registration failed: {0}")]
    ResourceRegistrationFailed(String),
    /// Required hardware capability missing.
    #[error("Hardware capability missing: {0}")]
    HardwareCapabilityMissing(String),
    /// Transform arena overflow.
    #[error("Transform arena overflow (1MB limit reached)")]
    TransformArenaOverflow,
}

impl AshError {
    /// Helper to create a Vulkan error from a message.
    pub fn vulkan<S: Into<String>>(msg: S) -> Self {
        Self::VulkanError(msg.into())
    }
}

/// Convenient result type for the renderer.
pub type Result<T, E = AshError> = std::result::Result<T, E>;

impl From<ash::vk::Result> for AshError {
    fn from(result: ash::vk::Result) -> Self {
        Self::VulkanError(format!("{result:?}"))
    }
}

impl From<crate::renderer::resource_registry::ResourceError> for AshError {
    fn from(err: crate::renderer::resource_registry::ResourceError) -> Self {
        Self::ResourceRegistrationFailed(err.to_string())
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
