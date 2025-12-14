//! Safe RAII wrapper for Vulkan resources with automatic cleanup.
//!
//! Ensures resources are properly cleaned up when dropped.
//!
//! # Example
//! ```ignore
//! let buffer = SafeResource::new(buffer, |b| {
//!     device.destroy_buffer(b, None);
//! });
//!
//! // Use buffer...
//! buffer.write(data);
//!
//! // Automatically cleaned up on drop
//! ```

use std::ops::{Deref, DerefMut};

/// A safe wrapper around a Vulkan resource that ensures cleanup on drop.
pub struct SafeResource<T> {
    resource: Option<T>,
    cleanup: Option<Box<dyn FnOnce(T) + Send>>,
    name: Option<String>,
}

impl<T> SafeResource<T> {
    /// Create a new safe resource with a cleanup function.
    ///
    /// # Arguments
    /// * `resource` - The resource to wrap
    /// * `cleanup` - Function called with the resource when dropped
    pub fn new<F: FnOnce(T) + Send + 'static>(resource: T, cleanup: F) -> Self {
        Self {
            resource: Some(resource),
            cleanup: Some(Box::new(cleanup)),
            name: None,
        }
    }

    /// Create a named safe resource (for debugging).
    pub fn named<F: FnOnce(T) + Send + 'static>(
        resource: T,
        name: impl Into<String>,
        cleanup: F,
    ) -> Self {
        Self {
            resource: Some(resource),
            cleanup: Some(Box::new(cleanup)),
            name: Some(name.into()),
        }
    }

    /// Create a safe resource without cleanup (ownership transfer).
    pub fn unmanaged(resource: T) -> Self {
        Self {
            resource: Some(resource),
            cleanup: None,
            name: None,
        }
    }

    /// Take ownership of the resource without running cleanup.
    ///
    /// Use this when transferring ownership to another system.
    pub fn into_inner(mut self) -> T {
        self.cleanup = None; // Prevent cleanup
        self.resource.take().expect("Resource already taken")
    }

    /// Check if the resource is still valid.
    pub fn is_valid(&self) -> bool {
        self.resource.is_some()
    }

    /// Get the debug name if set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Replace the cleanup function.
    pub fn set_cleanup<F: FnOnce(T) + Send + 'static>(&mut self, cleanup: F) {
        self.cleanup = Some(Box::new(cleanup));
    }

    /// Remove the cleanup function (makes this unmanaged).
    pub fn remove_cleanup(&mut self) {
        self.cleanup = None;
    }
}

impl<T> Deref for SafeResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.resource.as_ref().expect("Resource already taken")
    }
}

impl<T> DerefMut for SafeResource<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resource.as_mut().expect("Resource already taken")
    }
}

impl<T> Drop for SafeResource<T> {
    fn drop(&mut self) {
        if let (Some(resource), Some(cleanup)) = (self.resource.take(), self.cleanup.take()) {
            if let Some(ref name) = self.name {
                log::trace!("[SafeResource] Cleaning up '{name}'");
            }
            cleanup(resource);
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for SafeResource<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeResource")
            .field("resource", &self.resource)
            .field("name", &self.name)
            .field("has_cleanup", &self.cleanup.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_cleanup_on_drop() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_clone = Arc::clone(&cleaned);

        {
            let _resource = SafeResource::new(42, move |_| {
                cleaned_clone.store(true, Ordering::SeqCst);
            });
        }

        assert!(cleaned.load(Ordering::SeqCst));
    }

    #[test]
    fn test_into_inner_skips_cleanup() {
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_clone = Arc::clone(&cleaned);

        let resource = SafeResource::new(42, move |_| {
            cleaned_clone.store(true, Ordering::SeqCst);
        });

        let value = resource.into_inner();
        assert_eq!(value, 42);
        assert!(!cleaned.load(Ordering::SeqCst));
    }

    #[test]
    fn test_deref() {
        let resource = SafeResource::unmanaged(vec![1, 2, 3]);
        assert_eq!(resource.len(), 3);
        assert_eq!(resource[0], 1);
    }

    #[test]
    fn test_named() {
        let resource = SafeResource::named(42, "test_resource", |_| {});
        assert_eq!(resource.name(), Some("test_resource"));
    }
}
