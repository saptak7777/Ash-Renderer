//! Thread-safe resource pool using lock-free crossbeam channels.
//!
//! Provides efficient pooling of resources with automatic return-to-pool
//! when the handle is dropped.
//!
//! # Example
//! ```ignore
//! let pool = ThreadSafeResourcePool::new(4, || create_buffer());
//!
//! // Acquire from pool (or create new if empty)
//! let buffer = pool.acquire();
//!
//! // Use the buffer...
//! buffer.write(data);
//!
//! // Automatically returned to pool when dropped
//! drop(buffer);
//! ```

use crossbeam_channel::{unbounded, Receiver, Sender};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Pool usage statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total acquire calls
    pub acquires: u64,
    /// Acquires satisfied from pool (hits)
    pub hits: u64,
    /// Acquires that required new allocation (misses)
    pub misses: u64,
    /// Total returns to pool
    pub returns: u64,
    /// Current available resources
    pub available: usize,
    /// Peak resources in pool
    pub peak_available: usize,
}

impl PoolStats {
    /// Calculate hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.acquires == 0 {
            1.0
        } else {
            self.hits as f64 / self.acquires as f64
        }
    }

    /// Format stats as a string
    pub fn format(&self) -> String {
        format!(
            "Pool: {} acquires ({:.1}% hit), {} available (peak: {})",
            self.acquires,
            self.hit_rate() * 100.0,
            self.available,
            self.peak_available
        )
    }
}

/// A thread-safe resource pool that manages reusable resources.
///
/// Uses lock-free channels for high-performance pooling.
pub struct ThreadSafeResourcePool<T> {
    available: Receiver<T>,
    returner: Sender<T>,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    // Statistics (atomic for thread safety)
    acquires: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    returns: AtomicU64,
    peak_available: AtomicUsize,
}

/// A handle to a resource borrowed from the pool.
///
/// When dropped, the resource is automatically returned to the pool.
pub struct PooledResource<T: Send + 'static> {
    resource: Option<T>,
    returner: Sender<T>,
}

impl<T: Send + 'static> ThreadSafeResourcePool<T> {
    /// Creates a new thread-safe resource pool with the specified initial size.
    ///
    /// # Arguments
    /// * `size` - Initial number of resources to create in the pool
    /// * `factory` - A function that creates new resources when the pool is empty
    pub fn new<F>(size: usize, factory: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        // Use unbounded channel to prevent blocking when returning resources
        let (sender, receiver) = unbounded();
        let returner = sender.clone();
        let factory = Arc::new(factory);

        // Pre-populate the pool
        for _ in 0..size {
            if let Err(e) = sender.send((factory.as_ref())()) {
                log::error!("[ThreadSafePool] Failed to populate pool: {e}");
            }
        }

        log::debug!("[ThreadSafePool] Created pool with {size} resources");

        Self {
            available: receiver,
            returner,
            factory,
            acquires: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            returns: AtomicU64::new(0),
            peak_available: AtomicUsize::new(size),
        }
    }

    /// Attempts to acquire a resource from the pool without blocking.
    ///
    /// Returns `None` if no resources are immediately available.
    pub fn try_acquire(&self) -> Option<PooledResource<T>> {
        self.acquires.fetch_add(1, Ordering::Relaxed);
        self.available.try_recv().ok().map(|resource| {
            self.hits.fetch_add(1, Ordering::Relaxed);
            PooledResource::new(resource, self.returner.clone())
        })
    }

    /// Acquires a resource from the pool, creating a new one if necessary.
    ///
    /// If the pool is empty, this will create a new resource using the factory.
    pub fn acquire(&self) -> PooledResource<T> {
        self.acquires.fetch_add(1, Ordering::Relaxed);
        match self.available.try_recv() {
            Ok(resource) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                PooledResource::new(resource, self.returner.clone())
            }
            Err(_) => {
                // Create a new resource if pool is empty
                self.misses.fetch_add(1, Ordering::Relaxed);
                log::trace!("[ThreadSafePool] Pool empty, creating new resource");
                PooledResource::new((self.factory)(), self.returner.clone())
            }
        }
    }

    /// Clears all resources from the pool.
    pub fn clear(&self) {
        let mut count = 0;
        while self.available.try_recv().is_ok() {
            count += 1;
        }
        if count > 0 {
            log::debug!("[ThreadSafePool] Cleared {count} resources");
        }
    }

    /// Get the number of available resources in the pool.
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Get a clone of the sender for returning resources.
    pub fn sender(&self) -> Sender<T> {
        self.returner.clone()
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let available = self.available.len();
        // Update peak if current is higher
        let _ = self.peak_available.fetch_max(available, Ordering::Relaxed);

        PoolStats {
            acquires: self.acquires.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            returns: self.returns.load(Ordering::Relaxed),
            available,
            peak_available: self.peak_available.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics counters
    pub fn reset_stats(&self) {
        self.acquires.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.returns.store(0, Ordering::Relaxed);
        self.peak_available
            .store(self.available.len(), Ordering::Relaxed);
    }
}

impl<T> Drop for ThreadSafeResourcePool<T> {
    fn drop(&mut self) {
        // Clean up any remaining resources
        let mut count = 0;
        while let Ok(_resource) = self.available.try_recv() {
            count += 1;
        }
        if count > 0 {
            log::debug!("[ThreadSafePool] Dropped pool with {count} remaining resources");
        }
    }
}

impl<T: Send + 'static> PooledResource<T> {
    /// Creates a new pooled resource handle.
    pub fn new(resource: T, returner: Sender<T>) -> Self {
        Self {
            resource: Some(resource),
            returner,
        }
    }

    /// Takes the resource without returning it to the pool.
    ///
    /// Use this when you need permanent ownership of the resource.
    pub fn into_inner(mut self) -> T {
        self.resource.take().expect("Resource already taken")
    }

    /// Check if the resource is still available.
    pub fn is_valid(&self) -> bool {
        self.resource.is_some()
    }
}

impl<T: Send> Deref for PooledResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.resource.as_ref().expect("Resource already taken")
    }
}

impl<T: Send> DerefMut for PooledResource<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resource.as_mut().expect("Resource already taken")
    }
}

impl<T: Send + 'static> Drop for PooledResource<T> {
    fn drop(&mut self) {
        if let Some(resource) = self.resource.take() {
            // Try to return the resource to the pool
            if let Err(e) = self.returner.send(resource) {
                // Pool is dropped or full, just drop the resource
                log::trace!("[ThreadSafePool] Could not return resource: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    #[test]
    fn test_pool_basic() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let pool = ThreadSafeResourcePool::new(2, move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            42 // Dummy resource
        });

        // Initial resources should be created
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(pool.available_count(), 2);

        // Acquire and drop resources
        {
            let _res1 = pool.acquire();
            let _res2 = pool.acquire();
            assert_eq!(pool.available_count(), 0);

            // Pool is empty, next acquire creates new
            let _res3 = pool.acquire();
            assert_eq!(counter.load(Ordering::SeqCst), 3);
        }

        // Resources returned to pool
        assert_eq!(pool.available_count(), 3);
    }

    #[test]
    fn test_pool_thread_safety() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(ThreadSafeResourcePool::new(2, {
            let counter = counter.clone();
            move || {
                counter.fetch_add(1, Ordering::SeqCst);
                42
            }
        }));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let pool = pool.clone();
                thread::spawn(move || {
                    let resource = pool.acquire();
                    thread::sleep(std::time::Duration::from_millis(5));
                    drop(resource);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Resources should be reused, not many created
        assert!(counter.load(Ordering::SeqCst) <= 10);
    }

    #[test]
    fn test_into_inner() {
        let pool = ThreadSafeResourcePool::new(1, || 123);
        let resource = pool.acquire();
        let value = resource.into_inner();
        assert_eq!(value, 123);

        // Pool should be empty (resource not returned)
        assert_eq!(pool.available_count(), 0);
    }
}
