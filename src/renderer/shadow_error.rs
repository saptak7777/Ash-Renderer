//! Shadow map allocation errors with detailed information
use ash::vk;
use thiserror::Error;

/// Detailed shadow map allocation error
#[derive(Error, Debug)]
pub enum ShadowMapError {
    /// No suitable memory type was found for the depth image
    #[error("No suitable memory type found: required flags {required_flags}")]
    NoSuitableMemoryType {
        required_flags: String,
        available_types: Vec<u32>,
    },

    /// The GPU ran out of memory for the requested allocation
    #[error("Out of GPU memory: needed {needed_mb:.1} MB, resolution {resolution}x{resolution}")]
    OutOfMemory {
        needed_mb: f32,
        available_mb: f32,
        resolution: u32,
    },

    /// A raw Vulkan error occurred during allocation
    #[error("Vulkan error: {0}")]
    VulkanError(#[from] vk::Result),

    /// Shadow map creation failed after trying all fallback resolutions
    #[error("Failed at all resolutions: {attempts:?}")]
    AllResolutionsFailed {
        attempts: Vec<u32>,
        final_error: Box<ShadowMapError>,
    },
}

impl ShadowMapError {
    /// User-friendly message for display in UI or logs
    pub fn user_message(&self) -> String {
        match self {
            Self::NoSuitableMemoryType { .. } => {
                "Your GPU doesn't support the required memory type for shadow maps. \
                This can happen on very old hardware or with outdated drivers."
                    .to_string()
            }
            Self::OutOfMemory { needed_mb, .. } => {
                format!(
                    "Out of GPU memory! Shadow map needs {needed_mb:.1} MB. \
                    Try closing other high-performance applications or reducing graphics settings."
                )
            }
            Self::AllResolutionsFailed { attempts, .. } => {
                format!(
                    "Failed to create shadow map at resolutions: {}. \
                    Your GPU may not have enough memory or supports for target resolutions.",
                    attempts
                        .iter()
                        .map(|r| format!("{r}x{r}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Self::VulkanError(e) => {
                format!("Internal graphics error: {e:?}")
            }
        }
    }

    /// Suggested actions for the user to resolve the issue
    pub fn suggested_actions(&self) -> Vec<String> {
        match self {
            Self::OutOfMemory { .. } | Self::AllResolutionsFailed { .. } => vec![
                "Close other applications to free GPU memory".to_string(),
                "Reduce shadow resolution in settings".to_string(),
            ],
            Self::NoSuitableMemoryType { .. } => vec![
                "Update your graphics drivers".to_string(),
                "Check if your GPU is Vulkan 1.1+ compatible".to_string(),
            ],
            _ => vec!["Restart the application".to_string()],
        }
    }
}
