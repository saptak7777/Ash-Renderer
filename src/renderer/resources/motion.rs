//! Motion vector data structures for Temporal Super-Resolution (TSR)
//!
//! Provides GPU-compatible structures for storing current and previous frame
//! transformation matrices, enabling accurate motion vector calculation for
//! temporal upscaling and anti-aliasing.

use glam::Mat4;

/// Per-object motion data for TSR (GPU layout)
///
/// Stores both current and previous frame MVP matrices to enable
/// motion vector calculation in shaders. Follows Unreal Engine 5's
/// approach of maintaining transform history for temporal coherence.
///
/// # Memory Layout
/// - 128 bytes total (GPU-aligned)
/// - 64 bytes: current frame MVP
/// - 64 bytes: previous frame MVP
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectMotionData {
    /// Current frame Model-View-Projection matrix
    pub current_mvp: [[f32; 4]; 4],
    /// Previous frame Model-View-Projection matrix
    pub previous_mvp: [[f32; 4]; 4],
}

// Compile-time size verification (GPU expects exactly 128 bytes)
const _: () = assert!(std::mem::size_of::<ObjectMotionData>() == 128);
const _: () = assert!(std::mem::align_of::<ObjectMotionData>() == 16);

impl ObjectMotionData {
    /// Create motion data from current and previous MVP matrices
    pub fn new(current_mvp: Mat4, previous_mvp: Mat4) -> Self {
        Self {
            current_mvp: current_mvp.to_cols_array_2d(),
            previous_mvp: previous_mvp.to_cols_array_2d(),
        }
    }

    /// Create motion data with identity previous matrix (for first frame)
    pub fn from_current(current_mvp: Mat4) -> Self {
        Self::new(current_mvp, Mat4::IDENTITY)
    }
}

impl Default for ObjectMotionData {
    fn default() -> Self {
        Self {
            current_mvp: Mat4::IDENTITY.to_cols_array_2d(),
            previous_mvp: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_data_size() {
        assert_eq!(std::mem::size_of::<ObjectMotionData>(), 128);
    }

    #[test]
    fn test_motion_data_alignment() {
        assert_eq!(std::mem::align_of::<ObjectMotionData>(), 16);
    }

    #[test]
    fn test_motion_data_default() {
        let data = ObjectMotionData::default();
        assert_eq!(data.current_mvp, Mat4::IDENTITY.to_cols_array_2d());
        assert_eq!(data.previous_mvp, Mat4::IDENTITY.to_cols_array_2d());
    }

    #[test]
    fn test_motion_data_from_current() {
        let mvp = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let data = ObjectMotionData::from_current(mvp);

        assert_eq!(data.current_mvp, mvp.to_cols_array_2d());
        assert_eq!(data.previous_mvp, Mat4::IDENTITY.to_cols_array_2d());
    }
}
