//! Light Manager
//!
//! Coordinates GPU light data and tile indices for Forward+ rendering.
//! Assumes the renderer handles frame synchronization (e.g., waiting on fences)
//! before calling upload_lights or create_buffers.

use ash::vk;
use vk_mem::Alloc;

use super::light_culling::{GpuLight, LightCullingConfig, LightCullingPass, MAX_LIGHTS};
use super::lighting::{DirectionalLight, PointLight, SpotLight};

/// GPU buffer info for lights
pub struct LightBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub size: u64,
}

/// GPU buffer info for tile indices (output of light culling compute)
pub struct TileBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub size: u64,
}

/// Forward+ info UBO (matches shader ForwardPlusInfo)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForwardPlusInfo {
    /// Number of tiles in x and y
    pub num_tiles: [u32; 2],
    /// Tile size in pixels
    pub tile_size: u32,
    /// Padding
    pub _padding: u32,
}

/// Manages high-level GPU resources for the light culling compute pass.
/// Note: This implementation currently doesn't double-buffer; it assumes
/// the caller ensures GPU execution is complete before overwriting.
pub struct LightManager {
    /// CPU-side culling logic
    culling_pass: LightCullingPass,
    /// GPU light buffer (to be created)
    light_buffer: Option<LightBuffer>,
    /// GPU tile buffer (compute output)
    tile_buffer: Option<TileBuffer>,
    /// Forward+ info for shaders
    fp_info: ForwardPlusInfo,
    /// Whether Forward+ is enabled
    enabled: bool,
    /// Whether buffers need recreation
    dirty: bool,
}

impl LightManager {
    /// Create a new light manager
    pub fn new() -> Self {
        Self {
            culling_pass: LightCullingPass::new(),
            light_buffer: None,
            tile_buffer: None,
            fp_info: ForwardPlusInfo::default(),
            enabled: true,
            dirty: true,
        }
    }

    /// Create with custom culling config
    pub fn with_config(config: LightCullingConfig) -> Result<Self, crate::error::AshError> {
        if config.debug_tiles && !config.enabled {
            return Err(crate::error::AshError::HardwareCapabilityMissing(
                "LightManager: debug_tiles requires culling to be enabled".to_string(),
            ));
        }

        Ok(Self {
            culling_pass: LightCullingPass::with_config(config),
            ..Self::new()
        })
    }

    pub fn update_lights(
        &mut self,
        point_lights: &[PointLight],
        directional_lights: &[DirectionalLight],
        spot_lights: &[SpotLight],
    ) {
        // We assume the incoming slices contain valid, world-space lighting data.
        // The culling pass handles its own internal capacity limits and sanitization.
        self.culling_pass
            .update_lights(point_lights, directional_lights, spot_lights);
        self.dirty = true;
    }

    /// Update for screen resize
    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.culling_pass.calculate_tiles(width, height);

        let (tiles_x, tiles_y, _) = self.culling_pass.get_dispatch_dimensions();
        self.fp_info = ForwardPlusInfo {
            num_tiles: [tiles_x, tiles_y],
            tile_size: super::light_culling::TILE_SIZE,
            _padding: 0,
        };

        // CRITICAL: Mark buffers as needing recreation
        // Tile buffer size depends on screen resolution
        self.dirty = true;

        log::debug!(
            "LightManager::on_resize: {width}x{height} -> {tiles_x}x{tiles_y} tiles (buffer needs recreation)",
        );
    }

    /// Get the light buffer data for upload
    pub fn get_light_buffer_data(&self) -> &[GpuLight] {
        self.culling_pass.get_light_buffer_data()
    }

    /// Get light count
    pub fn light_count(&self) -> usize {
        self.culling_pass.light_count()
    }

    /// Get tile buffer size in bytes
    pub fn get_tile_buffer_size(&self) -> usize {
        self.culling_pass.get_tile_buffer_size()
    }

    /// Get dispatch dimensions for compute shader
    pub fn get_dispatch_dimensions(&self) -> (u32, u32, u32) {
        self.culling_pass.get_dispatch_dimensions()
    }

    /// Get push constants for light culling shader
    pub fn get_culling_push_constants(
        &self,
        width: u32,
        height: u32,
    ) -> super::light_culling::LightCullingPushConstants {
        self.culling_pass.get_push_constants(width, height)
    }

    /// Get Forward+ info for fragment shader
    pub fn get_forward_plus_info(&self) -> ForwardPlusInfo {
        self.fp_info
    }

    /// Is Forward+ enabled and has lights?
    pub fn is_enabled(&self) -> bool {
        self.enabled && self.culling_pass.is_enabled()
    }

    /// Enable/disable Forward+
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Access culling pass config
    pub fn culling_config(&self) -> &LightCullingConfig {
        self.culling_pass.config()
    }

    /// Access culling pass config mutably
    pub fn culling_config_mut(&mut self) -> &mut LightCullingConfig {
        self.culling_pass.config_mut()
    }

    /// Check if buffers need recreation
    pub fn needs_buffer_update(&self) -> bool {
        self.dirty
    }

    /// Mark buffers as up-to-date
    pub fn mark_buffers_updated(&mut self) {
        self.dirty = false;
    }

    // =========================================================================
    // GPU Resource Management (implemented via VMA)
    // =========================================================================

    /// Create GPU buffers for lights and tiles.
    ///
    /// # Safety
    /// Caller must ensure no GPU commands are pending that reference the old buffers
    /// if this is called as a recreation (e.g., during resize).
    pub unsafe fn create_buffers(&mut self, allocator: &vk_mem::Allocator) -> crate::Result<()> {
        // Light buffer: MAX_LIGHTS * sizeof(GpuLight)
        let light_buffer_size = (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64;

        let light_buffer_info = vk::BufferCreateInfo::default()
            .size(light_buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let light_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (light_buffer, light_allocation) = allocator
            .create_buffer(&light_buffer_info, &light_alloc_info)
            .expect("LightManager: Base buffer allocation failed during initialization");

        self.light_buffer = Some(LightBuffer {
            buffer: light_buffer,
            allocation: light_allocation,
            size: light_buffer_size,
        });

        // Tile buffer initialization based on tile grid dimensions.
        // Formula: tiles_x * tiles_y * (MAX_LIGHTS_PER_TILE + 1) * sizeof(u32)
        let tile_buffer_size = self.get_tile_buffer_size().max(1024) as u64; // Minimum 1KB

        // Create tile buffer with host-accessible memory for zero-initialization
        let tile_buffer_info = vk::BufferCreateInfo::default()
            .size(tile_buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let tile_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let (tile_buffer, tile_allocation) = allocator
            .create_buffer(&tile_buffer_info, &tile_alloc_info)
            .expect("LightManager: Tile buffer allocation failed");

        // Zero out the tile buffer to prevent garbage data
        let tile_mapped = allocator.get_allocation_info(&tile_allocation).mapped_data;
        if !tile_mapped.is_null() {
            unsafe {
                std::ptr::write_bytes(tile_mapped as *mut u8, 0, tile_buffer_size as usize);
            }
        }

        self.tile_buffer = Some(TileBuffer {
            buffer: tile_buffer,
            allocation: tile_allocation,
            size: tile_buffer_size,
        });

        log::info!(
            "LightManager: Created buffers (light: {}KB, tile: {}KB)",
            light_buffer_size / 1024,
            tile_buffer_size / 1024
        );

        Ok(())
    }

    /// Recreate tile buffer if screen size changed
    ///
    /// # Safety
    /// GPU must be idle or synchronized. Old buffer must not be in use.
    pub unsafe fn recreate_tile_buffer_if_needed(
        &mut self,
        allocator: &vk_mem::Allocator,
    ) -> crate::Result<()> {
        if !self.dirty {
            return Ok(());
        }

        let new_tile_buffer_size = self.get_tile_buffer_size().max(1024) as u64;

        // Check if buffer exists and size matches
        if let Some(ref tile_buffer) = self.tile_buffer {
            if tile_buffer.size == new_tile_buffer_size {
                // Size hasn't changed, no need to recreate
                return Ok(());
            }

            log::info!(
                "LightManager: Recreating tile buffer ({}KB -> {}KB)",
                tile_buffer.size / 1024,
                new_tile_buffer_size / 1024
            );
        }

        // Destroy old buffer if it exists
        if let Some(mut old_tile_buffer) = self.tile_buffer.take() {
            allocator.destroy_buffer(old_tile_buffer.buffer, &mut old_tile_buffer.allocation);
        }

        // Create new tile buffer with host-accessible memory for zero-initialization
        let tile_buffer_info = vk::BufferCreateInfo::default()
            .size(new_tile_buffer_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let tile_alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let (tile_buffer, tile_allocation) = allocator
            .create_buffer(&tile_buffer_info, &tile_alloc_info)
            .map_err(|e| {
                crate::AshError::VulkanError(format!(
                    "LightManager: Tile buffer recreation failed: {e:?}"
                ))
            })?;

        // Zero out the tile buffer to prevent garbage data
        let tile_mapped = allocator.get_allocation_info(&tile_allocation).mapped_data;
        if !tile_mapped.is_null() {
            unsafe {
                std::ptr::write_bytes(tile_mapped as *mut u8, 0, new_tile_buffer_size as usize);
            }
        }

        self.tile_buffer = Some(TileBuffer {
            buffer: tile_buffer,
            allocation: tile_allocation,
            size: new_tile_buffer_size,
        });

        log::info!(
            "LightManager: Tile buffer recreated ({}KB)",
            new_tile_buffer_size / 1024
        );

        Ok(())
    }

    /// Upload light data to GPU buffer
    ///
    /// # Safety
    /// Buffer must have been created and be valid.
    pub unsafe fn upload_lights(&mut self, allocator: &vk_mem::Allocator) -> crate::Result<()> {
        let Some(ref light_buffer) = self.light_buffer else {
            return Ok(()); // Light buffer not initialized
        };

        let lights = self.get_light_buffer_data();
        if lights.is_empty() {
            return Ok(());
        }

        let data_size = std::mem::size_of_val(lights);
        let allocation_info = allocator.get_allocation_info(&light_buffer.allocation);

        let mapped_ptr = allocation_info.mapped_data;
        if !mapped_ptr.is_null() {
            // Memory is mapped with HOST_ACCESS_SEQUENTIAL_WRITE.
            // copy_nonoverlapping is used here as we're initializing the entire buffer segment.
            std::ptr::copy_nonoverlapping(
                lights.as_ptr() as *const u8,
                mapped_ptr as *mut u8,
                data_size,
            );
        }

        self.dirty = false;
        Ok(())
    }

    /// Get light buffer handle for descriptor binding
    pub fn get_light_buffer(&self) -> Option<vk::Buffer> {
        self.light_buffer.as_ref().map(|b| b.buffer)
    }

    /// Get tile buffer handle for descriptor binding
    pub fn get_tile_buffer(&self) -> Option<vk::Buffer> {
        self.tile_buffer.as_ref().map(|b| b.buffer)
    }

    /// Destroy GPU buffers
    ///
    /// # Safety
    /// Buffers must not be in use by GPU.
    pub unsafe fn destroy_buffers(&mut self, allocator: &vk_mem::Allocator) {
        if let Some(mut light_buffer) = self.light_buffer.take() {
            allocator.destroy_buffer(light_buffer.buffer, &mut light_buffer.allocation);
        }
        if let Some(mut tile_buffer) = self.tile_buffer.take() {
            allocator.destroy_buffer(tile_buffer.buffer, &mut tile_buffer.allocation);
        }
        log::info!("LightManager: Destroyed buffers");
    }
}

impl Default for LightManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_light_manager_update() {
        let mut manager = LightManager::new();

        let point_lights = vec![PointLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            color: Vec3::ONE,
            intensity: 1.0,
            radius: 10.0,
        }];

        manager.update_lights(&point_lights, &[], &[]);
        assert_eq!(manager.light_count(), 1);
        assert!(manager.needs_buffer_update());
    }

    #[test]
    fn test_resize() {
        let mut manager = LightManager::new();
        manager.on_resize(1920, 1080);

        let info = manager.get_forward_plus_info();
        assert!(info.num_tiles[0] > 0);
        assert!(info.num_tiles[1] > 0);
        assert_eq!(info.tile_size, 16); // TILE_SIZE
    }
}
