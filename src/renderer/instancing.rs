//! GPU Instancing System
//!
//! Provides efficient batched rendering of many identical objects.
//! Combines objects sharing the same mesh/material into single draw calls.
//!
//! # Features
//! - Automatic instance batching
//! - Per-instance data (transform, color, custom)
//! - Frustum culling of instances
//! - Statistics tracking

use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;

/// Maximum instances per draw call
pub const MAX_INSTANCES_PER_BATCH: usize = 65536;

/// Per-instance data (GPU layout)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    /// Model matrix row 0
    pub model_row0: [f32; 4],
    /// Model matrix row 1
    pub model_row1: [f32; 4],
    /// Model matrix row 2
    pub model_row2: [f32; 4],
    /// Model matrix row 3
    pub model_row3: [f32; 4],
    /// Instance color multiplier (RGBA)
    pub color: [f32; 4],
    /// Custom data (user-defined, e.g. animation frame, variation ID)
    pub custom: [f32; 4],
}

impl InstanceData {
    /// Create from transform matrix
    pub fn from_matrix(model: Mat4) -> Self {
        let cols = model.to_cols_array_2d();
        Self {
            model_row0: cols[0],
            model_row1: cols[1],
            model_row2: cols[2],
            model_row3: cols[3],
            color: [1.0, 1.0, 1.0, 1.0],
            custom: [0.0; 4],
        }
    }

    /// Create with transform and color
    pub fn new(model: Mat4, color: Vec4) -> Self {
        let cols = model.to_cols_array_2d();
        Self {
            model_row0: cols[0],
            model_row1: cols[1],
            model_row2: cols[2],
            model_row3: cols[3],
            color: color.to_array(),
            custom: [0.0; 4],
        }
    }

    /// Set custom data
    pub fn with_custom(mut self, custom: [f32; 4]) -> Self {
        self.custom = custom;
        self
    }

    /// Get position from matrix
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.model_row3[0], self.model_row3[1], self.model_row3[2])
    }
}

/// Batch key for grouping instances
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchKey {
    /// Mesh identifier
    pub mesh_id: u32,
    /// Material identifier
    pub material_id: u32,
}

impl BatchKey {
    pub fn new(mesh_id: u32, material_id: u32) -> Self {
        Self {
            mesh_id,
            material_id,
        }
    }
}

/// Instance batch (single draw call)
#[derive(Debug, Clone)]
pub struct InstanceBatch {
    /// Batch key
    pub key: BatchKey,
    /// Instance data for this batch
    pub instances: Vec<InstanceData>,
    /// Bounding sphere center (for frustum culling)
    pub bounds_center: Vec3,
    /// Bounding sphere radius
    pub bounds_radius: f32,
}

impl InstanceBatch {
    pub fn new(key: BatchKey) -> Self {
        Self {
            key,
            instances: Vec::new(),
            bounds_center: Vec3::ZERO,
            bounds_radius: 0.0,
        }
    }

    /// Add instance to batch
    pub fn add(&mut self, instance: InstanceData) {
        self.instances.push(instance);
    }

    /// Number of instances
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Is batch empty?
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Clear instances
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    /// Calculate bounding sphere from instances
    pub fn calculate_bounds(&mut self, mesh_radius: f32) {
        if self.instances.is_empty() {
            return;
        }

        // Calculate center as average of positions
        let sum: Vec3 = self.instances.iter().map(|i| i.position()).sum();
        self.bounds_center = sum / self.instances.len() as f32;

        // Find maximum distance from center
        let max_dist = self
            .instances
            .iter()
            .map(|i| (i.position() - self.bounds_center).length())
            .fold(0.0f32, f32::max);

        self.bounds_radius = max_dist + mesh_radius;
    }
}

/// Instancing statistics
#[derive(Debug, Clone, Default)]
pub struct InstancingStats {
    /// Total instances submitted
    pub total_instances: u32,
    /// Number of batches (draw calls)
    pub batch_count: u32,
    /// Instances culled
    pub instances_culled: u32,
    /// Average instances per batch
    pub avg_instances_per_batch: f32,
}

impl InstancingStats {
    /// Calculate efficiency (higher = better batching)
    pub fn efficiency(&self) -> f32 {
        if self.batch_count == 0 {
            0.0
        } else {
            self.avg_instances_per_batch / MAX_INSTANCES_PER_BATCH as f32
        }
    }

    /// Format as summary string
    pub fn format(&self) -> String {
        format!(
            "Instancing: {} instances in {} batches (avg {:.1}), {} culled",
            self.total_instances,
            self.batch_count,
            self.avg_instances_per_batch,
            self.instances_culled
        )
    }
}

/// Instancing manager
pub struct InstancingManager {
    /// Batches by key
    batches: HashMap<BatchKey, InstanceBatch>,
    /// Statistics
    stats: InstancingStats,
    /// Enable frustum culling of instances
    frustum_cull: bool,
}

impl InstancingManager {
    /// Create a new instancing manager
    pub fn new() -> Self {
        Self {
            batches: HashMap::new(),
            stats: InstancingStats::default(),
            frustum_cull: true,
        }
    }

    /// Begin new frame
    pub fn begin_frame(&mut self) {
        for batch in self.batches.values_mut() {
            batch.clear();
        }
        self.stats = InstancingStats::default();
    }

    /// Add an instance
    pub fn add_instance(&mut self, key: BatchKey, instance: InstanceData) {
        let batch = self
            .batches
            .entry(key.clone())
            .or_insert_with(|| InstanceBatch::new(key));

        if batch.count() < MAX_INSTANCES_PER_BATCH {
            batch.add(instance);
            self.stats.total_instances += 1;
        }
    }

    /// Add many instances at once
    pub fn add_instances(
        &mut self,
        key: BatchKey,
        instances: impl IntoIterator<Item = InstanceData>,
    ) {
        let batch = self
            .batches
            .entry(key.clone())
            .or_insert_with(|| InstanceBatch::new(key));

        for instance in instances {
            if batch.count() < MAX_INSTANCES_PER_BATCH {
                batch.add(instance);
                self.stats.total_instances += 1;
            }
        }
    }

    /// Finalize batches (call before rendering)
    pub fn finalize(&mut self) {
        // Remove empty batches
        self.batches.retain(|_, batch| !batch.is_empty());

        // Calculate stats
        self.stats.batch_count = self.batches.len() as u32;
        if self.stats.batch_count > 0 {
            self.stats.avg_instances_per_batch =
                self.stats.total_instances as f32 / self.stats.batch_count as f32;
        }
    }

    /// Get all batches for rendering
    pub fn batches(&self) -> impl Iterator<Item = &InstanceBatch> {
        self.batches.values()
    }

    /// Get batch by key
    pub fn get_batch(&self, key: &BatchKey) -> Option<&InstanceBatch> {
        self.batches.get(key)
    }

    /// Get current statistics
    pub fn stats(&self) -> &InstancingStats {
        &self.stats
    }

    /// Enable/disable frustum culling
    pub fn set_frustum_cull(&mut self, enabled: bool) {
        self.frustum_cull = enabled;
    }
}

impl Default for InstancingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_data() {
        let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let instance = InstanceData::from_matrix(model);
        let pos = instance.position();
        assert_eq!(pos, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_batching() {
        let mut manager = InstancingManager::new();
        manager.begin_frame();

        let key = BatchKey::new(1, 1);
        for i in 0..100 {
            let model = Mat4::from_translation(Vec3::new(i as f32, 0.0, 0.0));
            manager.add_instance(key.clone(), InstanceData::from_matrix(model));
        }

        manager.finalize();
        assert_eq!(manager.stats().total_instances, 100);
        assert_eq!(manager.stats().batch_count, 1);
    }

    #[test]
    fn test_multiple_batches() {
        let mut manager = InstancingManager::new();
        manager.begin_frame();

        // Different mesh IDs = different batches
        for mesh_id in 0..5 {
            let key = BatchKey::new(mesh_id, 0);
            manager.add_instance(key, InstanceData::default());
        }

        manager.finalize();
        assert_eq!(manager.stats().batch_count, 5);
    }
}
