//! Thread-safe LIFO deletion queue for deferred Vulkan resource cleanup.
//!
//! Ensures resources are cleaned up in the correct order (LIFO) after
//! the GPU has finished using them.
//!
//! # Example
//! ```ignore
//! let queue = DeletionQueue::new("frame_resources");
//!
//! // Queue cleanup for later
//! queue.push(move || {
//!     device.destroy_buffer(buffer, None);
//! });
//!
//! // After GPU sync, flush all pending deletions
//! queue.flush();
//! ```

use std::collections::VecDeque;
use std::sync::Mutex;

/// A thread-safe queue for deferring resource cleanup.
/// Resources are cleaned up in reverse order of addition (LIFO).
pub struct DeletionQueue {
    deletors: Mutex<VecDeque<Box<dyn FnOnce() + Send>>>,
    name: &'static str,
}

impl std::fmt::Debug for DeletionQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeletionQueue")
            .field("name", &self.name)
            .field("count", &self.len())
            .finish()
    }
}

impl DeletionQueue {
    /// Create a new deletion queue with the given name for debugging.
    pub fn new(name: &'static str) -> Self {
        Self {
            deletors: Mutex::new(VecDeque::new()),
            name,
        }
    }

    /// Add a cleanup function to the queue.
    /// The function will be called when `flush()` is called.
    pub fn push<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let mut deletors = self.deletors.lock().unwrap();
        deletors.push_back(Box::new(f));
        log::trace!(
            "[DeletionQueue] Added to '{}' (now has {} items)",
            self.name,
            deletors.len()
        );
    }

    /// Execute all cleanup functions in reverse order (LIFO).
    /// This ensures resources are cleaned up in the correct order.
    pub fn flush(&self) {
        let mut deletors = self.deletors.lock().unwrap();
        let count = deletors.len();

        if count == 0 {
            return;
        }

        log::debug!("[DeletionQueue] Flushing '{}' ({} items)", self.name, count);

        // Execute deletions in reverse order (LIFO)
        while let Some(deletor) = deletors.pop_back() {
            deletor();
        }

        log::debug!(
            "[DeletionQueue] Flushed {} items from '{}'",
            count,
            self.name
        );
    }

    /// Get the number of pending cleanup operations.
    pub fn len(&self) -> usize {
        self.deletors.lock().unwrap().len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.deletors.lock().unwrap().is_empty()
    }

    /// Get the queue name (for debugging)
    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl Drop for DeletionQueue {
    fn drop(&mut self) {
        let count = self.deletors.lock().unwrap().len();
        if count > 0 {
            log::warn!(
                "[DeletionQueue] Dropping '{}' with {} pending operations. Forcing flush...",
                self.name,
                count
            );
            self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_deletion_queue_lifo() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let queue = DeletionQueue::new("test_lifo");

        // Add cleanup functions
        let o1 = Arc::clone(&order);
        queue.push(move || o1.lock().unwrap().push(1));

        let o2 = Arc::clone(&order);
        queue.push(move || o2.lock().unwrap().push(2));

        let o3 = Arc::clone(&order);
        queue.push(move || o3.lock().unwrap().push(3));

        assert_eq!(queue.len(), 3);

        // Flush should execute in LIFO order (3, 2, 1)
        queue.flush();

        assert_eq!(*order.lock().unwrap(), vec![3, 2, 1]);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_deletion_queue_thread_safe() {
        let counter = Arc::new(AtomicU32::new(0));
        let queue = Arc::new(DeletionQueue::new("test_thread"));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let q = Arc::clone(&queue);
                let c = Arc::clone(&counter);
                std::thread::spawn(move || {
                    q.push(move || {
                        c.fetch_add(1, Ordering::SeqCst);
                    });
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(queue.len(), 10);
        queue.flush();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_drop_flushes() {
        let counter = Arc::new(AtomicU32::new(0));

        {
            let queue = DeletionQueue::new("test_drop");
            let c = Arc::clone(&counter);
            queue.push(move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
            // Queue dropped here
        }

        // Drop should have flushed
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
