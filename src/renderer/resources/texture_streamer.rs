use ash::vk;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::renderer::resources::texture::Texture;
use crate::vulkan;

/// Request sent to the streaming thread
enum StreamRequest {
    Load {
        path: PathBuf,
        _handle: u32, // ID to notify back when done (future improvement) or just for tracking
        name: String,
    },
    Shutdown,
}

/// Manages background texture loading and uploading
pub struct TextureStreamer {
    sender: Sender<StreamRequest>,
    join_handle: Option<thread::JoinHandle<()>>,
    pending_uploads: Arc<Mutex<HashMap<PathBuf, Arc<Texture>>>>, // Placeholder for result cache
    device: Arc<ash::Device>,
    transfer_command_pool: vk::CommandPool,
}

impl TextureStreamer {
    /// Creates a new TextureStreamer and starts the background loader thread.
    ///
    /// # Safety
    /// The device and allocator must remain valid for the duration of the streamer's life.
    pub fn new(
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        transfer_command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
    ) -> Self {
        let (sender, receiver) = unbounded();
        let pending_uploads = Arc::new(Mutex::new(HashMap::new()));
        let pending_clone = Arc::clone(&pending_uploads);

        // Clone device for thread
        let device_thread = Arc::clone(&device);

        let join_handle = thread::spawn(move || {
            Self::loader_loop(
                receiver,
                allocator,
                device_thread,
                transfer_command_pool,
                transfer_queue,
                pending_clone,
            );
        });

        Self {
            sender,
            join_handle: Some(join_handle),
            pending_uploads,
            device,
            transfer_command_pool,
        }
    }

    /// Request a texture to be loaded from disk.
    /// Returns immediately.
    pub fn request_load(&self, path: impl AsRef<Path>, handle: u32, name: impl Into<String>) {
        let _ = self.sender.send(StreamRequest::Load {
            path: path.as_ref().to_path_buf(),
            _handle: handle,
            name: name.into(),
        });
    }

    /// Checks if a requested texture has been loaded.
    /// This is a temporary polling mechanism until we integrate a full asset system.
    pub fn try_get(&self, path: impl AsRef<Path>) -> Option<Arc<Texture>> {
        let mut map = self.pending_uploads.lock().unwrap();
        map.remove(path.as_ref())
    }

    fn loader_loop(
        receiver: Receiver<StreamRequest>,
        allocator: Arc<vulkan::Allocator>,
        device: Arc<ash::Device>,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        completed: Arc<Mutex<HashMap<PathBuf, Arc<Texture>>>>,
    ) {
        log::info!("TextureStreamer background thread started");

        while let Ok(request) = receiver.recv() {
            match request {
                StreamRequest::Load { path, name, .. } => {
                    log::debug!("Streaming asset: {path:?}");

                    // 1. Try to load .ash_tex first
                    let ash_tex_path = path.with_extension("ash_tex");
                    let result = if ash_tex_path.exists() {
                        unsafe {
                            Texture::load_from_ash_tex(
                                &ash_tex_path,
                                Arc::clone(&allocator),
                                Arc::clone(&device),
                                command_pool,
                                queue,
                                Some(&name),
                            )
                        }
                    } else {
                        // Fallback to slow load (TODO: Remove this restriction in strict mode?)
                        // For now, we only support .ash_tex in streaming path to enforce the pipeline
                        log::warn!("Streamer received non-cooked asset: {path:?}. Skipping.");
                        continue;
                    };

                    match result {
                        Ok(texture) => {
                            let mut map = completed.lock().unwrap();
                            map.insert(path, Arc::new(texture));
                            log::debug!("Finished streaming: {name}");
                        }
                        Err(e) => {
                            log::error!("Failed to stream texture {path:?}: {e}");
                        }
                    }
                }
                StreamRequest::Shutdown => break,
            }
        }

        log::info!("TextureStreamer shutting down");
    }
}

impl Drop for TextureStreamer {
    fn drop(&mut self) {
        log::info!("Shutting down TextureStreamer...");
        let _ = self.sender.send(StreamRequest::Shutdown);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }

        // Cleanup command pool
        unsafe {
            self.device
                .destroy_command_pool(self.transfer_command_pool, None);
        }
        log::info!("TextureStreamer shut down complete");
    }
}
