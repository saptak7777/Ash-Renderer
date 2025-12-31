use crate::{AshError, Result};

/// Validates bindless texture indices before they are sent to the GPU.
pub struct BindlessValidator;

impl BindlessValidator {
    /// Validates a single index against the current bindless manager state.
    pub fn validate_index(index: i32, max_resources: u32) -> Result<()> {
        if index < -1 {
            return Err(AshError::VulkanError(format!(
                "Invalid bindless index: {index} (must be >= -1)"
            )));
        }

        if index >= max_resources as i32 {
            return Err(AshError::VulkanError(format!(
                "Bindless index out of bounds: {index} (max: {max_resources})"
            )));
        }

        Ok(())
    }

    /// Validates a set of texture indices.
    pub fn validate_indices(indices: &[i32], max_resources: u32) -> Result<()> {
        for &index in indices {
            Self::validate_index(index, max_resources)?;
        }
        Ok(())
    }
}
