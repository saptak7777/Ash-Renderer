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
    /// Page state for each virtual page
    page_states: Vec<PageState>,
    /// Free physical page pool (ring buffer)
    free_physical_pages: VecDeque<(u32, u32)>,
    /// Current frame index for LRU
    current_frame: u32,
}

impl PageManager {
    /// Create a new page manager
    pub fn new(virtual_resolution: u32, physical_resolution: u32, page_size: u32) -> Self {
        let virtual_pages_per_axis = virtual_resolution / page_size;
        let physical_pages_per_axis = physical_resolution / page_size;

        let virtual_page_count = (virtual_pages_per_axis * virtual_pages_per_axis) as usize;

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
            "PageManager initialized: {} virtual pages, {} physical pages",
            virtual_page_count,
            free_physical_pages.len()
        );

        Self {
            virtual_pages_per_axis,
            physical_pages_per_axis,
            page_states,
            free_physical_pages,
            current_frame: 0,
        }
    }

    /// Begin a new frame
    pub fn begin_frame(&mut self, frame_index: u32) {
        self.current_frame = frame_index;
    }

    /// Get virtual page index from coordinates
    fn virtual_page_index(&self, x: u32, y: u32) -> usize {
        (y * self.virtual_pages_per_axis + x) as usize
    }

    /// Process page requests and allocate physical pages
    ///
    /// Returns allocations that need to be written to GPU
    pub fn process_requests(&mut self, requests: &[PageRequest]) -> Vec<PageAllocation> {
        let mut allocations = Vec::new();

        for request in requests {
            let idx = self.virtual_page_index(request.virtual_x, request.virtual_y);

            match self.page_states[idx] {
                PageState::Free | PageState::Pending => {
                    // Try to allocate a physical page
                    if let Some((phys_x, phys_y)) = self.allocate_physical_page() {
                        self.page_states[idx] = PageState::Allocated {
                            physical_x: phys_x,
                            physical_y: phys_y,
                            last_used_frame: self.current_frame,
                        };

                        allocations.push(PageAllocation {
                            virtual_x: request.virtual_x,
                            virtual_y: request.virtual_y,
                            physical_x: phys_x,
                            physical_y: phys_y,
                        });

                        log::debug!(
                            "Allocated page ({}, {}) -> ({}, {})",
                            request.virtual_x,
                            request.virtual_y,
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
                            };

                            allocations.push(PageAllocation {
                                virtual_x: request.virtual_x,
                                virtual_y: request.virtual_y,
                                physical_x: phys_x,
                                physical_y: phys_y,
                            });

                            log::debug!(
                                "Evicted LRU and allocated page ({}, {}) -> ({}, {})",
                                request.virtual_x,
                                request.virtual_y,
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
                    ..
                } => {
                    // Page already allocated, just update last used frame
                    self.page_states[idx] = PageState::Allocated {
                        physical_x,
                        physical_y,
                        last_used_frame: self.current_frame,
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
    pub fn get_physical_coords(&self, virtual_x: u32, virtual_y: u32) -> Option<(u32, u32)> {
        let idx = self.virtual_page_index(virtual_x, virtual_y);
        match self.page_states[idx] {
            PageState::Allocated {
                physical_x,
                physical_y,
                ..
            } => Some((physical_x, physical_y)),
            _ => None,
        }
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
        let mut manager = PageManager::new(1024, 512, 128);
        manager.begin_frame(0);

        let requests = vec![
            PageRequest {
                virtual_x: 0,
                virtual_y: 0,
                priority: 1.0,
                _padding: 0,
            },
            PageRequest {
                virtual_x: 1,
                virtual_y: 0,
                priority: 1.0,
                _padding: 0,
            },
        ];

        let allocations = manager.process_requests(&requests);
        assert_eq!(allocations.len(), 2);

        // Verify physical coords are assigned
        assert!(manager.get_physical_coords(0, 0).is_some());
        assert!(manager.get_physical_coords(1, 0).is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let mut manager = PageManager::new(512, 256, 128);

        // Fill all physical pages
        let physical_page_count = (256 / 128) * (256 / 128);
        let mut requests = Vec::new();
        for i in 0..physical_page_count {
            requests.push(PageRequest {
                virtual_x: i,
                virtual_y: 0,
                priority: 1.0,
                _padding: 0,
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
            _padding: 0,
        }];

        let allocations = manager.process_requests(&new_request);
        assert_eq!(allocations.len(), 1);
    }
}
