//! Page Manager - Handles virtual page allocation and tracking

use std::collections::VecDeque;

use super::resources::{PageAllocation, PageRequest};

/// Page state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// Page is not allocated
    Free,
    /// Page is allocated and valid
    Allocated {
        physical_x: u32,
        physical_y: u32,
        last_used_frame: u32,
        dirty: bool,
    },
    /// Page is pending allocation
    Pending,
}

/// Manages virtual to physical page mapping
pub struct PageManager {
    /// Number of pages per axis in virtual space
    virtual_pages_per_axis: u32,
    /// Number of pages per axis in physical cache
    physical_pages_per_axis: u32,
    /// Page state for each virtual page (flattened: layer * (pages_per_axis^2) + y * pages_per_axis + x)
    page_states: Vec<PageState>,
    /// Number of layers managed
    _layer_count: u32,
    /// Free physical page pool (ring buffer)
    free_physical_pages: VecDeque<(u32, u32)>,
    /// Current frame index for LRU
    current_frame: u32,
    /// List of indices of dirty pages for efficient clearing
    dirty_pages_indices: Vec<usize>,
}

impl PageManager {
    /// Create a new page manager
    pub fn new(
        virtual_resolution: u32,
        physical_resolution: u32,
        page_size: u32,
        layer_count: u32,
    ) -> Self {
        let virtual_pages_per_axis = virtual_resolution / page_size;
        let physical_pages_per_axis = physical_resolution / page_size;

        let pages_per_layer = (virtual_pages_per_axis * virtual_pages_per_axis) as usize;
        let virtual_page_count = pages_per_layer * (layer_count as usize);

        // Initialize all virtual pages as free
        let page_states = vec![PageState::Free; virtual_page_count];

        // Initialize free physical page pool
        let mut free_physical_pages = VecDeque::new();
        for y in 0..physical_pages_per_axis {
            for x in 0..physical_pages_per_axis {
                free_physical_pages.push_back((x, y));
            }
        }

        log::info!(
            "PageManager initialized: {} virtual pages ({} layers), {} physical pages",
            virtual_page_count,
            layer_count,
            free_physical_pages.len()
        );

        Self {
            virtual_pages_per_axis,
            physical_pages_per_axis,
            page_states,
            _layer_count: layer_count,
            free_physical_pages,
            current_frame: 0,
            dirty_pages_indices: Vec::new(),
        }
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self, frame_index: u32) {
        self.current_frame = frame_index;

        // Clear dirty flags from previous frame
        for idx in self.dirty_pages_indices.drain(..) {
            if let PageState::Allocated {
                physical_x,
                physical_y,
                last_used_frame,
                ..
            } = self.page_states[idx]
            {
                self.page_states[idx] = PageState::Allocated {
                    physical_x,
                    physical_y,
                    last_used_frame,
                    dirty: false,
                };
            }
        }
    }

    /// Get virtual page index from coordinates and layer
    fn virtual_page_index(&self, x: u32, y: u32, layer: u32) -> usize {
        let pages_per_layer = (self.virtual_pages_per_axis * self.virtual_pages_per_axis) as usize;
        let layer_offset = (layer as usize) * pages_per_layer;
        layer_offset + (y * self.virtual_pages_per_axis + x) as usize
    }

    /// Invalidate a specific page (mark as dirty)
    pub fn invalidate_page(&mut self, virtual_x: u32, virtual_y: u32, layer: u32) {
        let idx = self.virtual_page_index(virtual_x, virtual_y, layer);

        if let PageState::Allocated { dirty, .. } = &mut self.page_states[idx] {
            if !*dirty {
                *dirty = true;
                self.dirty_pages_indices.push(idx);
            }
        }
    }

    /// Process page requests and allocate physical pages
    ///
    /// Returns allocations that need to be written to GPU
    pub fn process_requests(&mut self, requests: &[PageRequest]) -> Vec<PageAllocation> {
        let mut allocations = Vec::new();

        for request in requests {
            let idx = self.virtual_page_index(request.virtual_x, request.virtual_y, request.layer);

            match self.page_states[idx] {
                PageState::Free | PageState::Pending => {
                    // Try to allocate a physical page
                    if let Some((phys_x, phys_y)) = self.allocate_physical_page() {
                        self.page_states[idx] = PageState::Allocated {
                            physical_x: phys_x,
                            physical_y: phys_y,
                            last_used_frame: self.current_frame,
                            dirty: true,
                        };
                        self.dirty_pages_indices.push(idx);

                        allocations.push(PageAllocation {
                            virtual_x: request.virtual_x,
                            virtual_y: request.virtual_y,
                            physical_x: phys_x,
                            physical_y: phys_y,
                            layer: request.layer,
                            flags: 1, // Dirty
                            _padding: [0; 2],
                        });

                        log::debug!(
                            "Allocated page ({}, {}, L{}) -> ({}, {})",
                            request.virtual_x,
                            request.virtual_y,
                            request.layer,
                            phys_x,
                            phys_y
                        );
                    } else {
                        // No free pages, try to evict LRU page
                        if let Some((phys_x, phys_y)) = self.evict_lru_page() {
                            self.page_states[idx] = PageState::Allocated {
                                physical_x: phys_x,
                                physical_y: phys_y,
                                last_used_frame: self.current_frame,
                                dirty: true,
                            };
                            self.dirty_pages_indices.push(idx);

                            allocations.push(PageAllocation {
                                virtual_x: request.virtual_x,
                                virtual_y: request.virtual_y,
                                physical_x: phys_x,
                                physical_y: phys_y,
                                layer: request.layer,
                                flags: 1, // Dirty
                                _padding: [0; 2],
                            });

                            log::debug!(
                                "Evicted LRU and allocated page ({}, {}, L{}) -> ({}, {})",
                                request.virtual_x,
                                request.virtual_y,
                                request.layer,
                                phys_x,
                                phys_y
                            );
                        } else {
                            log::warn!("Failed to allocate page - cache full");
                        }
                    }
                }
                PageState::Allocated {
                    physical_x,
                    physical_y,
                    dirty,
                    ..
                } => {
                    // Page already allocated, just update last used frame
                    self.page_states[idx] = PageState::Allocated {
                        physical_x,
                        physical_y,
                        last_used_frame: self.current_frame,
                        dirty, // Preserve dirty state
                    };
                }
            }
        }

        allocations
    }

    /// Allocate a free physical page
    fn allocate_physical_page(&mut self) -> Option<(u32, u32)> {
        self.free_physical_pages.pop_front()
    }

    /// Evict least recently used page
    fn evict_lru_page(&mut self) -> Option<(u32, u32)> {
        let mut oldest_frame = self.current_frame;
        let mut oldest_idx = None;

        for (idx, state) in self.page_states.iter().enumerate() {
            if let PageState::Allocated {
                last_used_frame, ..
            } = state
            {
                if *last_used_frame < oldest_frame {
                    oldest_frame = *last_used_frame;
                    oldest_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = oldest_idx {
            if let PageState::Allocated {
                physical_x,
                physical_y,
                ..
            } = self.page_states[idx]
            {
                self.page_states[idx] = PageState::Free;
                return Some((physical_x, physical_y));
            }
        }

        None
    }

    /// Get physical coordinates for a virtual page
    pub fn get_physical_coords(
        &self,
        virtual_x: u32,
        virtual_y: u32,
        layer: u32,
    ) -> Option<(u32, u32)> {
        let idx = self.virtual_page_index(virtual_x, virtual_y, layer);
        match self.page_states[idx] {
            PageState::Allocated {
                physical_x,
                physical_y,
                ..
            } => Some((physical_x, physical_y)),
            _ => None,
        }
    }

    /// Get all currently allocated pages for rendering
    pub fn get_allocated_pages(&self) -> Vec<PageAllocation> {
        let mut allocations = Vec::new();

        let pages_per_layer = (self.virtual_pages_per_axis * self.virtual_pages_per_axis) as usize;

        for (idx, state) in self.page_states.iter().enumerate() {
            if let PageState::Allocated {
                physical_x,
                physical_y,
                dirty,
                ..
            } = state
            {
                // Decode flat index into (layer, y, x)
                let layer = (idx / pages_per_layer) as u32;
                let layer_local_idx = idx % pages_per_layer;
                let virtual_y = (layer_local_idx as u32) / self.virtual_pages_per_axis;
                let virtual_x = (layer_local_idx as u32) % self.virtual_pages_per_axis;

                let flags = if *dirty { 1 } else { 0 };

                allocations.push(PageAllocation {
                    virtual_x,
                    virtual_y,
                    physical_x: *physical_x,
                    physical_y: *physical_y,
                    layer,
                    flags,
                    _padding: [0; 2],
                });
            }
        }

        allocations
    }

    /// Get statistics
    pub fn stats(&self) -> PageManagerStats {
        let allocated = self
            .page_states
            .iter()
            .filter(|s| matches!(s, PageState::Allocated { .. }))
            .count();

        PageManagerStats {
            total_virtual_pages: self.page_states.len(),
            allocated_pages: allocated,
            free_physical_pages: self.free_physical_pages.len(),
            total_physical_pages: (self.physical_pages_per_axis * self.physical_pages_per_axis)
                as usize,
        }
    }

    /// Clear all allocations (for debugging)
    pub fn clear(&mut self) {
        for state in &mut self.page_states {
            *state = PageState::Free;
        }

        self.free_physical_pages.clear();
        for y in 0..self.physical_pages_per_axis {
            for x in 0..self.physical_pages_per_axis {
                self.free_physical_pages.push_back((x, y));
            }
        }

        log::info!("PageManager cleared");
    }
}

/// Page manager statistics
#[derive(Debug, Clone, Copy)]
pub struct PageManagerStats {
    pub total_virtual_pages: usize,
    pub allocated_pages: usize,
    pub free_physical_pages: usize,
    pub total_physical_pages: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_allocation() {
        // 4 Layers
        let mut manager = PageManager::new(1024, 512, 128, 4);
        manager.begin_frame(0);

        let requests = vec![
            PageRequest {
                virtual_x: 0,
                virtual_y: 0,
                priority: 1.0,
                layer: 0,
            },
            PageRequest {
                virtual_x: 1,
                virtual_y: 0,
                priority: 1.0,
                layer: 0, // Same layer
            },
            PageRequest {
                virtual_x: 0,
                virtual_y: 0,
                priority: 1.0,
                layer: 1, // Different layer, same coords - should get different allocation
            },
        ];

        let allocations = manager.process_requests(&requests);
        assert_eq!(allocations.len(), 3);

        // Verify physical coords are assigned
        assert!(manager.get_physical_coords(0, 0, 0).is_some());
        assert!(manager.get_physical_coords(1, 0, 0).is_some());
        assert!(manager.get_physical_coords(0, 0, 1).is_some());

        // Ensure different layers mapped to different physical pages (or at least different entries)
        let phys_l0 = manager.get_physical_coords(0, 0, 0).unwrap();
        let phys_l1 = manager.get_physical_coords(0, 0, 1).unwrap();
        assert_ne!(phys_l0, phys_l1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut manager = PageManager::new(512, 256, 128, 1); // 1 Layer

        // Fill all physical pages
        let physical_page_count = (256 / 128) * (256 / 128);
        let mut requests = Vec::new();
        for i in 0..physical_page_count {
            requests.push(PageRequest {
                virtual_x: i,
                virtual_y: 0,
                priority: 1.0,
                layer: 0,
            });
        }

        manager.begin_frame(0);
        manager.process_requests(&requests);

        // Request one more page - should evict LRU
        manager.begin_frame(1);
        let new_request = vec![PageRequest {
            virtual_x: physical_page_count,
            virtual_y: 0,
            priority: 1.0,
            layer: 0,
        }];

        let allocations = manager.process_requests(&new_request);
        assert_eq!(allocations.len(), 1);
    }
}
