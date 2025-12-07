use glam::Vec3;
use std::collections::HashSet;
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

/// First-person camera controller
pub struct CameraController {
    pub position: Vec3,
    pub yaw: f32,   // Horizontal rotation (radians)
    pub pitch: f32, // Vertical rotation (radians)

    // Input state
    keys_pressed: HashSet<KeyCode>,
    mouse_delta: (f32, f32),

    // Movement parameters
    pub speed: f32,
    pub mouse_sensitivity: f32,
}

impl CameraController {
    /// Create new camera controller
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            keys_pressed: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            speed: 5.0, // units per second
            mouse_sensitivity: 0.01,
        }
    }

    /// Update camera based on input and elapsed time
    pub fn update(&mut self, delta_time: f32) {
        // Handle keyboard movement
        self.handle_movement(delta_time);

        // Handle mouse look
        self.handle_mouse_look();
    }

    /// Process keyboard input
    pub fn handle_key_event(&mut self, input: KeyEvent) {
        let key_code = match input.physical_key {
            PhysicalKey::Code(code) => code,
            _ => return,
        };

        match input.state {
            winit::event::ElementState::Pressed => {
                self.keys_pressed.insert(key_code);
            }
            winit::event::ElementState::Released => {
                self.keys_pressed.remove(&key_code);
            }
        }
    }

    /// Process mouse movement
    pub fn handle_mouse_motion(&mut self, delta: (f32, f32)) {
        self.mouse_delta = delta;
    }

    /// Internal: Handle keyboard-based movement
    fn handle_movement(&mut self, delta_time: f32) {
        let mut direction = Vec3::ZERO;

        // WASD movement
        for &key in &self.keys_pressed {
            match key {
                KeyCode::KeyW => direction.z -= 1.0,  // Forward
                KeyCode::KeyS => direction.z += 1.0,  // Backward
                KeyCode::KeyA => direction.x -= 1.0,  // Left
                KeyCode::KeyD => direction.x += 1.0,  // Right
                KeyCode::Space => direction.y += 1.0, // Up
                KeyCode::ShiftLeft | KeyCode::ShiftRight => direction.y -= 1.0, // Down
                _ => {}
            }
        }

        // Normalize and apply direction relative to camera rotation
        if direction.length() > 0.0 {
            direction = direction.normalize();

            // Create rotation matrix from yaw/pitch
            let forward = Vec3::new(
                self.yaw.sin() * self.pitch.cos(),
                self.pitch.sin(),
                self.yaw.cos() * self.pitch.cos(),
            );

            let right = Vec3::new(
                (self.yaw - std::f32::consts::PI / 2.0).sin(),
                0.0,
                (self.yaw - std::f32::consts::PI / 2.0).cos(),
            );

            let up = Vec3::Y;

            // Calculate movement in world space
            let movement = (forward * direction.z + right * direction.x + up * direction.y)
                * self.speed
                * delta_time;

            self.position += movement;
        }
    }

    /// Internal: Handle mouse-based look
    fn handle_mouse_look(&mut self) {
        let (delta_x, delta_y) = self.mouse_delta;

        if delta_x != 0.0 || delta_y != 0.0 {
            self.yaw -= delta_x * self.mouse_sensitivity;
            self.pitch -= delta_y * self.mouse_sensitivity;

            // Clamp pitch to prevent camera flip
            let max_pitch = std::f32::consts::PI / 2.0 - 0.1;
            self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

            // Normalize yaw
            self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
        }

        // Reset mouse delta for next frame
        self.mouse_delta = (0.0, 0.0);
    }

    /// Get forward direction
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
    }

    /// Get right direction
    pub fn right(&self) -> Vec3 {
        Vec3::new(
            (self.yaw - std::f32::consts::PI / 2.0).sin(),
            0.0,
            (self.yaw - std::f32::consts::PI / 2.0).cos(),
        )
    }

    /// Get up direction
    pub fn up(&self) -> Vec3 {
        Vec3::Y
    }
}
