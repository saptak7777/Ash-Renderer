use glam::{EulerRot, Mat4, Quat, Vec3};

/// 3D Transform with position, rotation, scale
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    /// Creates identity transform
    pub fn identity() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Converts to 4x4 model matrix
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_translation(self.position)
            * Mat4::from_quat(self.rotation)
            * Mat4::from_scale(self.scale)
    }

    /// Sets rotation from euler angles (radians)
    pub fn set_rotation(&mut self, euler: Vec3) {
        self.rotation = Quat::from_euler(EulerRot::XYZ, euler.x, euler.y, euler.z);
    }

    /// Rotates by adding euler angles
    pub fn rotate(&mut self, euler: Vec3) {
        self.rotation *= Quat::from_euler(EulerRot::XYZ, euler.x, euler.y, euler.z);
    }
}

/// MVP matrices for rendering
#[derive(Debug, Clone, Copy)]
pub struct MVP {
    pub model: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
}

impl MVP {
    /// Creates MVP matrices
    pub fn new(model: Mat4, view: Mat4, projection: Mat4) -> Self {
        Self {
            model,
            view,
            projection,
        }
    }

    /// Calculates combined matrix (projection * view * model)
    pub fn combined(&self) -> Mat4 {
        self.projection * self.view * self.model
    }
}

/// Simple perspective camera
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// Creates a default camera looking at origin
    pub fn default(aspect: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov: 45.0,
            aspect,
            near: 0.5,
            far: 100.0,
        }
    }

    /// Creates a camera with custom parameters
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        Self {
            position,
            target,
            up: Vec3::Y,
            fov: 45.0,
            aspect,
            near: 0.5,
            far: 100.0,
        }
    }

    /// Calculates view matrix (lookAt)
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Calculates projection matrix (perspective)
    /// Note: Vulkan has Y pointing down in NDC, so we flip Y
    pub fn projection_matrix(&self) -> Mat4 {
        let mut proj =
            Mat4::perspective_rh(self.fov.to_radians(), self.aspect, self.near, self.far);
        // Flip Y for Vulkan's coordinate system (Y points down in NDC)
        proj.y_axis.y *= -1.0;
        proj
    }
}
