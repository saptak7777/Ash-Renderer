use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

use crate::renderer::resource_registry::ResourceRegistry;
use crate::{AshError, Result};

use super::descriptor_set::DescriptorSet;

/// Upper bound for bindless resources per descriptor type
pub const MAX_BINDLESS_RESOURCES: u32 = 1024 * 128; // 128k entries per type by default

const FRAMES_IN_FLIGHT: usize = 3;
/// Maximum number of frame pools to prevent unbounded memory growth.
/// Beyond this limit, the oldest (already-recycled) pool is reused.
const MAX_FRAME_POOLS: usize = 16;

struct FramePool {
    pool: vk::DescriptorPool,
    used_sets: u32,
    frame_number: u64,
    sets: Vec<vk::DescriptorSet>,
}

/// Ring-buffered descriptor allocator with optional bindless support
pub struct DescriptorAllocator {
    device: Arc<ash::Device>,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    sets_per_pool: u32,
    frame_pools: Vec<FramePool>,
    current_frame: u64,
    bindless_pool: Option<vk::DescriptorPool>,
    /// Static pool for long-lived descriptors (textures) that should never be reset
    static_pool: vk::DescriptorPool,
    static_pool_used: u32,
    descriptor_set_cache: HashMap<vk::DescriptorSet, vk::DescriptorPool>,
    resource_registry: Option<Arc<ResourceRegistry>>,
    managed_pools: HashMap<vk::DescriptorPool, bool>,
}

impl DescriptorAllocator {
    pub fn new(
        device: Arc<ash::Device>,
        sets_per_pool: u32,
        resource_registry: Option<Arc<ResourceRegistry>>,
    ) -> Result<Self> {
        let pool_sizes = Self::default_pool_sizes(sets_per_pool);

        // Create static pool for long-lived descriptors (textures)
        let static_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(sets_per_pool * 8) // Larger capacity for textures
            .pool_sizes(&pool_sizes);
        let static_pool = unsafe {
            device
                .create_descriptor_pool(&static_pool_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create static descriptor pool: {e}"))
                })?
        };

        let mut allocator = Self {
            device,
            pool_sizes,
            sets_per_pool: sets_per_pool.max(8),
            frame_pools: Vec::with_capacity(FRAMES_IN_FLIGHT * 2),
            current_frame: 0,
            bindless_pool: None,
            static_pool,
            static_pool_used: 0,
            descriptor_set_cache: HashMap::new(),
            resource_registry,
            managed_pools: HashMap::new(),
        };

        for _ in 0..FRAMES_IN_FLIGHT {
            allocator.create_pool()?;
        }

        Ok(allocator)
    }

    fn default_pool_sizes(sets_per_pool: u32) -> Vec<vk::DescriptorPoolSize> {
        let scale = sets_per_pool.max(1);
        vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: scale * 4,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: scale * 8,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: scale * 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: scale * 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                descriptor_count: scale,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                descriptor_count: scale,
            },
        ]
    }

    fn create_pool(&mut self) -> Result<&mut FramePool> {
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.sets_per_pool)
            .pool_sizes(&self.pool_sizes);

        let pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create descriptor pool: {e}"))
                })?
        };

        let pool_entry = FramePool {
            pool,
            used_sets: 0,
            frame_number: self.current_frame,
            sets: Vec::new(),
        };
        if let Some(registry) = &self.resource_registry {
            let _ = registry.register_descriptor_pool(pool);
            self.managed_pools.insert(pool, true);
        }
        self.frame_pools.push(pool_entry);

        Ok(self
            .frame_pools
            .last_mut()
            .expect("just pushed descriptor pool"))
    }

    pub fn next_frame(&mut self) {
        self.current_frame += 1;
        let threshold = self.current_frame.saturating_sub(FRAMES_IN_FLIGHT as u64);

        for pool in &mut self.frame_pools {
            if pool.frame_number < threshold {
                unsafe {
                    let _ = self
                        .device
                        .reset_descriptor_pool(pool.pool, vk::DescriptorPoolResetFlags::empty());
                }
                pool.used_sets = 0;
                pool.frame_number = self.current_frame;
                pool.sets.clear();
            }
        }

        // O(N) cleanup using HashSet instead of O(N*M) nested loop
        let active_pools: std::collections::HashSet<_> =
            self.frame_pools.iter().map(|p| p.pool).collect();
        self.descriptor_set_cache
            .retain(|_, pool| active_pools.contains(pool));
    }

    pub fn allocate_set(
        &mut self,
        layout: &vk::DescriptorSetLayout,
        bindings: &[vk::DescriptorSetLayoutBinding<'static>],
    ) -> Result<DescriptorSet> {
        let (set, pool) = self.allocate_raw_set(layout)?;
        self.descriptor_set_cache.insert(set, pool);
        DescriptorSet::new(Arc::clone(&self.device), set, *layout, bindings)
    }

    /// Allocate a descriptor set from the static pool (for long-lived resources like textures).
    /// These sets are NEVER reset by `next_frame()`.
    pub fn allocate_static_set(
        &mut self,
        layout: &vk::DescriptorSetLayout,
        bindings: &[vk::DescriptorSetLayoutBinding<'static>],
    ) -> Result<DescriptorSet> {
        let layouts = [*layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.static_pool)
            .set_layouts(&layouts);

        let set = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to allocate static descriptor set: {e}"))
                })?
        }[0];

        self.static_pool_used += 1;
        DescriptorSet::new(Arc::clone(&self.device), set, *layout, bindings)
    }

    fn allocate_raw_set(
        &mut self,
        layout: &vk::DescriptorSetLayout,
    ) -> Result<(vk::DescriptorSet, vk::DescriptorPool)> {
        // First try existing pools with available capacity
        for pool in &mut self.frame_pools {
            if pool.used_sets >= self.sets_per_pool {
                continue;
            }

            let layouts = [*layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool.pool)
                .set_layouts(&layouts);

            match unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
                Ok(sets) => {
                    let set = sets[0];
                    pool.used_sets += 1;
                    pool.sets.push(set);
                    return Ok((set, pool.pool));
                }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => continue,
                Err(e) => {
                    return Err(AshError::VulkanError(format!(
                        "Failed to allocate descriptor set: {e}"
                    )))
                }
            }
        }

        // If at max pool limit, force-reset and reuse the oldest pool
        if self.frame_pools.len() >= MAX_FRAME_POOLS {
            log::warn!("Descriptor pool limit reached ({MAX_FRAME_POOLS}), recycling oldest pool");
            // Find oldest pool (lowest frame_number)
            if let Some(oldest) = self.frame_pools.iter_mut().min_by_key(|p| p.frame_number) {
                unsafe {
                    let _ = self
                        .device
                        .reset_descriptor_pool(oldest.pool, vk::DescriptorPoolResetFlags::empty());
                }
                oldest.used_sets = 0;
                oldest.frame_number = self.current_frame;
                oldest.sets.clear();

                let layouts = [*layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(oldest.pool)
                    .set_layouts(&layouts);

                let sets = unsafe {
                    self.device
                        .allocate_descriptor_sets(&alloc_info)
                        .map_err(|e| {
                            AshError::VulkanError(format!("Failed to allocate descriptor set: {e}"))
                        })?
                };

                let set = sets[0];
                oldest.used_sets += 1;
                oldest.sets.push(set);
                return Ok((set, oldest.pool));
            }
        }

        // Create new pool (under limit)
        let pool_index = self.frame_pools.len();
        self.create_pool()?;
        if let Some(pool) = self.frame_pools.get_mut(pool_index) {
            let layouts = [*layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool.pool)
                .set_layouts(&layouts);

            let sets = unsafe {
                self.device
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| {
                        AshError::VulkanError(format!("Failed to allocate descriptor set: {e}"))
                    })?
            };

            let set = sets[0];
            pool.used_sets += 1;
            pool.sets.push(set);
            Ok((set, pool.pool))
        } else {
            Err(AshError::VulkanError(
                "Failed to access newly created descriptor pool".into(),
            ))
        }
    }

    pub fn allocate_bindless_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
        bindings: &[vk::DescriptorSetLayoutBinding<'static>],
        max_count: u32,
    ) -> Result<DescriptorSet> {
        if self.bindless_pool.is_none() {
            self.create_bindless_pool()?;
        }

        let pool = self
            .bindless_pool
            .ok_or_else(|| AshError::VulkanError("Bindless pool not initialized".into()))?;

        let counts = [max_count];
        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&counts);

        let layouts = [layout];
        let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        alloc_info = alloc_info.push_next(&mut variable_info);

        let set = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| {
                    AshError::VulkanError(format!(
                        "Failed to allocate bindless descriptor set: {e}"
                    ))
                })?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    AshError::VulkanError("Bindless descriptor allocation returned no sets".into())
                })?
        };

        DescriptorSet::new(Arc::clone(&self.device), set, layout, bindings)
    }

    fn create_bindless_pool(&mut self) -> Result<()> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: MAX_BINDLESS_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: MAX_BINDLESS_RESOURCES,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: MAX_BINDLESS_RESOURCES,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .flags(
                vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET
                    | vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            );

        let pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    AshError::VulkanError(format!("Failed to create bindless descriptor pool: {e}"))
                })?
        };

        if let Some(registry) = &self.resource_registry {
            let _ = registry.register_descriptor_pool(pool);
            self.managed_pools.insert(pool, true);
        }

        self.bindless_pool = Some(pool);
        Ok(())
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        unsafe {
            // Destroy static pool (never managed by registry)
            self.device.destroy_descriptor_pool(self.static_pool, None);

            if let Some(pool) = self.bindless_pool {
                if !self.managed_pools.contains_key(&pool) {
                    self.device.destroy_descriptor_pool(pool, None);
                }
            }
            for pool in &self.frame_pools {
                if !self.managed_pools.contains_key(&pool.pool) {
                    self.device.destroy_descriptor_pool(pool.pool, None);
                }
            }
        }
    }
}
