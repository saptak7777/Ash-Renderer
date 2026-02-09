//! GPU Instancing System
//!
//! Provides efficient batched rendering of many identical objects.
//! Combines objects sharing the same mesh/material into single draw calls.
//!
//! # Features
//! - Automatic instance batching
//! - Per-instance data (transform, color, custom)
//! - Statistics tracking

use crate::renderer::resources::material::MaterialHandle;
use crate::renderer::vcgs::CullObjectData;
use ahash::AHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Maximum instances per draw call
pub const MAX_INSTANCES_PER_BATCH: usize = 65536;

pub type InstanceData = CullObjectData;

/// Batch key for grouping instances
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchKey {
    /// Mesh identifier
    pub mesh_id: u32,
    /// Material identifier
    pub material_id: MaterialHandle,
}

impl BatchKey {
    pub fn new(mesh_id: u32, material_id: MaterialHandle) -> Self {
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
    /// Hashes of instances in this batch (for duplicate detection)
    pub instance_hashes: HashSet<u64>,
}

impl InstanceBatch {
    pub fn new(key: BatchKey) -> Self {
        Self {
            key,
            instances: Vec::new(),
            instance_hashes: HashSet::new(),
        }
    }

    /// Add instance to batch
    pub fn add(&mut self, instance: InstanceData) {
        self.instances.push(instance);
    }

    /// Add instance uniquely (returns true if added, false if duplicate)
    pub fn add_unique(&mut self, instance: InstanceData) -> bool {
        let mut hasher = AHasher::default();
        // CullObjectData is Pod, so we can hash it as bytes
        let bytes = bytemuck::bytes_of(&instance);
        hasher.write(bytes);
        let hash = hasher.finish();

        if self.instance_hashes.contains(&hash) {
            false
        } else {
            self.instance_hashes.insert(hash);
            self.instances.push(instance);
            true
        }
    }

    /// Number of instances
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Is batch empty?
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Check if batch contains any transparent instances
    pub fn is_transparent(&self) -> bool {
        self.instances.iter().any(|i| i.is_transparent())
    }

    /// Check if batch contains any shadow-casting instances
    pub fn casts_shadows(&self) -> bool {
        self.instances
            .iter()
            .any(|i| i.has_flag(crate::renderer::vcgs::CULL_FLAG_CAST_SHADOWS))
    }

    /// Check if batch contains any shadow-receiving instances
    pub fn receives_shadows(&self) -> bool {
        self.instances
            .iter()
            .any(|i| i.has_flag(crate::renderer::vcgs::CULL_FLAG_RECEIVE_SHADOWS))
    }

    /// Check if batch is shadow-only (casts shadows but doesn't receive them, used for Rage Engine optimization)
    pub fn is_shadow_only(&self) -> bool {
        self.casts_shadows() && !self.receives_shadows()
    }

    /// Clear instances
    pub fn clear(&mut self) {
        self.instances.clear();
        self.instance_hashes.clear();
    }
}

/// Instancing statistics
#[derive(Debug, Clone, Default)]
pub struct InstancingStats {
    /// Total instances submitted (before duplicate prevention)
    pub total_instances: u32,
    /// Number of batches (draw calls)
    pub batch_count: u32,
    /// Instances culled
    pub instances_culled: u32,
    /// Instances prevented (duplicates)
    pub duplicates_prevented: u32,
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
            "Instancing: {} instances in {} batches (avg {:.1}), {} culled, {} duplicates prevented",
            self.total_instances,
            self.batch_count,
            self.avg_instances_per_batch,
            self.instances_culled,
            self.duplicates_prevented
        )
    }
}

/// Instancing manager
pub struct InstancingManager {
    /// Batches by key
    batches: HashMap<BatchKey, InstanceBatch>,
    /// Statistics
    stats: InstancingStats,
    /// Enable duplicate prevention
    duplicate_prevention: bool,
}

impl InstancingManager {
    /// Create a new instancing manager
    pub fn new() -> Self {
        Self {
            batches: HashMap::new(),
            stats: InstancingStats::default(),
            duplicate_prevention: true,
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
            if self.duplicate_prevention {
                if batch.add_unique(instance) {
                    self.stats.total_instances += 1;
                } else {
                    self.stats.duplicates_prevented += 1;
                }
            } else {
                batch.add(instance);
                self.stats.total_instances += 1;
            }
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
                if self.duplicate_prevention {
                    if batch.add_unique(instance) {
                        self.stats.total_instances += 1;
                    } else {
                        self.stats.duplicates_prevented += 1;
                    }
                } else {
                    batch.add(instance);
                    self.stats.total_instances += 1;
                }
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

    /// Get visible opaque batches
    pub fn visible_batches(&self) -> impl Iterator<Item = &InstanceBatch> {
        self.batches.values().filter(|b| !b.is_transparent())
    }

    /// Get opaque batches
    pub fn opaque_batches(&self) -> impl Iterator<Item = &InstanceBatch> {
        self.batches.values().filter(|b| !b.is_transparent())
    }

    /// Get transparent batches
    pub fn transparent_batches(&self) -> impl Iterator<Item = &InstanceBatch> {
        self.batches.values().filter(|b| b.is_transparent())
    }

    /// Get batch by key
    pub fn get_batch(&self, key: &BatchKey) -> Option<&InstanceBatch> {
        self.batches.get(key)
    }

    /// Get current statistics
    pub fn stats(&self) -> &InstancingStats {
        &self.stats
    }

    /// Enable/disable duplicate prevention
    pub fn set_duplicate_prevention(&mut self, enabled: bool) {
        self.duplicate_prevention = enabled;
    }

    /// Is duplicate prevention enabled?
    pub fn is_duplicate_prevention_enabled(&self) -> bool {
        self.duplicate_prevention
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
    use glam::{Mat4, Vec3};

    #[test]
    fn test_instance_data() {
        let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let instance = InstanceData::from_matrix(model);
        let pos = instance.position();
        assert_eq!(pos, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_duplicate_prevention() {
        let mut manager = InstancingManager::new();
        let key = BatchKey::new(1, MaterialHandle { index: 1 });
        let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let instance = InstanceData::from_matrix(model);

        // Add same instance twice
        manager.add_instance(key.clone(), instance);
        manager.add_instance(key.clone(), instance);

        manager.finalize();
        let stats = manager.stats();

        assert_eq!(stats.total_instances, 1);
        assert_eq!(stats.duplicates_prevented, 1);
        assert_eq!(manager.get_batch(&key).unwrap().count(), 1);
    }

    #[test]
    fn test_duplicate_prevention_disabled() {
        let mut manager = InstancingManager::new();
        manager.set_duplicate_prevention(false);
        let key = BatchKey::new(1, MaterialHandle { index: 1 });
        let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let instance = InstanceData::from_matrix(model);

        // Add same instance twice
        manager.add_instance(key.clone(), instance);
        manager.add_instance(key.clone(), instance);

        manager.finalize();
        let stats = manager.stats();

        assert_eq!(stats.total_instances, 2);
        assert_eq!(stats.duplicates_prevented, 0);
        assert_eq!(manager.get_batch(&key).unwrap().count(), 2);
    }

    #[test]
    fn test_batching() {
        let mut manager = InstancingManager::new();
        manager.begin_frame();

        let key = BatchKey::new(1, MaterialHandle { index: 1 });
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
            let key = BatchKey::new(mesh_id, MaterialHandle::null());
            manager.add_instance(key, InstanceData::default());
        }

        manager.finalize();
        assert_eq!(manager.stats().batch_count, 5);
    }
}
