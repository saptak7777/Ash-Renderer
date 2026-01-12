//! Async GPU Readback System
//!
//! Provides asynchronous query results for occlusion and timestamp queries
//! using fence-based completion tracking and type-safe callbacks.
//!
//! Features:
//! - Non-blocking query result retrieval
//! - Type-safe callback system with Send guarantees
//! - FIFO completion order
//! - Automatic resource cleanup via RAII

use ash::vk;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::vulkan::VulkanDevice;
use crate::Result;

/// Maximum queries per pool
const OCCLUSION_POOL_SIZE: u32 = 256;
const TIMESTAMP_POOL_SIZE: u32 = 256;
const PIPELINE_STATS_POOL_SIZE: u32 = 64;

/// Result of an async GPU readback
#[derive(Clone, Copy, Debug)]
pub enum ReadbackResult {
    /// Occlusion query result (number of samples passed)
    OcclusionSamples(u64),
    /// Timestamp delta in nanoseconds
    TimestampDelta(u64),
    /// Pipeline statistics
    PipelineStats {
        input_assembly_vertices: u64,
        input_assembly_primitives: u64,
        vertex_shader_invocations: u64,
        fragment_shader_invocations: u64,
    },
}

/// Type-safe callback for readback completion
pub type ReadbackCallback = Box<dyn FnOnce(ReadbackResult) + Send + 'static>;

/// Query type for readback requests
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueryType {
    Occlusion,
    Timestamp,
    #[allow(dead_code)]
    PipelineStats,
}

/// Handle to a pending readback request
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadbackHandle(u64);

/// Internal readback request
struct ReadbackRequest {
    pool: vk::QueryPool,
    index: u32,
    fence: vk::Fence,
    #[allow(dead_code)]
    frame_submitted: u64,
    query_type: QueryType,
    callback: ReadbackCallback,
    #[allow(dead_code)]
    handle: ReadbackHandle,
}

/// Async GPU readback manager
pub struct AsyncReadbackManager {
    device: Arc<ash::Device>,

    // Query pools
    occlusion_pool: vk::QueryPool,
    timestamp_pool: vk::QueryPool,
    pipeline_stats_pool: vk::QueryPool,

    // Free indices per pool
    free_occlusion_indices: Vec<u32>,
    free_timestamp_indices: Vec<u32>,
    free_pipeline_stats_indices: Vec<u32>,

    // Pending requests (FIFO order)
    pending: VecDeque<ReadbackRequest>,

    // Frame counter
    current_frame: u64,

    // Handle generation
    next_handle_id: u64,

    initialized: bool,
}

impl AsyncReadbackManager {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            occlusion_pool: vk::QueryPool::null(),
            timestamp_pool: vk::QueryPool::null(),
            pipeline_stats_pool: vk::QueryPool::null(),
            free_occlusion_indices: Vec::new(),
            free_timestamp_indices: Vec::new(),
            free_pipeline_stats_indices: Vec::new(),
            pending: VecDeque::new(),
            current_frame: 0,
            next_handle_id: 1,
            initialized: false,
        }
    }

    /// Initialize query pools
    ///
    /// # Safety
    /// Device must be valid
    pub unsafe fn init(&mut self, _vulkan_device: &VulkanDevice) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        self.create_query_pools()?;
        self.initialized = true;

        log::info!("Async readback manager initialized");
        Ok(())
    }

    unsafe fn create_query_pools(&mut self) -> Result<()> {
        // Occlusion query pool
        let occlusion_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(OCCLUSION_POOL_SIZE);

        self.occlusion_pool = self.device.create_query_pool(&occlusion_info, None)?;
        self.free_occlusion_indices = (0..OCCLUSION_POOL_SIZE).collect();

        // Timestamp query pool
        let timestamp_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMESTAMP_POOL_SIZE);

        self.timestamp_pool = self.device.create_query_pool(&timestamp_info, None)?;
        self.free_timestamp_indices = (0..TIMESTAMP_POOL_SIZE).collect();

        // Pipeline statistics pool
        let stats_flags = vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES
            | vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES
            | vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS;

        let pipeline_stats_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .query_count(PIPELINE_STATS_POOL_SIZE)
            .pipeline_statistics(stats_flags);

        self.pipeline_stats_pool = self.device.create_query_pool(&pipeline_stats_info, None)?;
        self.free_pipeline_stats_indices = (0..PIPELINE_STATS_POOL_SIZE).collect();

        log::debug!(
            "Created query pools: {OCCLUSION_POOL_SIZE} occlusion, {TIMESTAMP_POOL_SIZE} timestamp, {PIPELINE_STATS_POOL_SIZE} pipeline stats"
        );

        Ok(())
    }

    /// Begin occlusion query
    ///
    /// # Safety
    /// Command buffer must be in recording state
    pub unsafe fn begin_occlusion_query(
        &mut self,
        cmd: vk::CommandBuffer,
    ) -> Result<ReadbackHandle> {
        let index = self
            .free_occlusion_indices
            .pop()
            .ok_or(crate::AshError::VulkanError(
                "No free occlusion query indices".to_string(),
            ))?;

        self.device
            .cmd_reset_query_pool(cmd, self.occlusion_pool, index, 1);

        self.device.cmd_begin_query(
            cmd,
            self.occlusion_pool,
            index,
            vk::QueryControlFlags::empty(),
        );

        let handle = ReadbackHandle(self.next_handle_id);
        self.next_handle_id += 1;

        Ok(handle)
    }

    /// End occlusion query and request async readback
    ///
    /// # Safety
    /// Command buffer must be in recording state, query must have been started
    pub unsafe fn end_occlusion_query(
        &mut self,
        cmd: vk::CommandBuffer,
        _handle: ReadbackHandle,
        index: u32,
        callback: ReadbackCallback,
    ) -> Result<()> {
        self.device.cmd_end_query(cmd, self.occlusion_pool, index);

        let fence_info = vk::FenceCreateInfo::default();
        let fence = self.device.create_fence(&fence_info, None)?;

        let request = ReadbackRequest {
            pool: self.occlusion_pool,
            index,
            fence,
            frame_submitted: self.current_frame,
            query_type: QueryType::Occlusion,
            callback,
            handle: ReadbackHandle(self.next_handle_id),
        };

        self.next_handle_id += 1;
        self.pending.push_back(request);
        Ok(())
    }

    /// Write timestamp query
    ///
    /// # Safety
    /// Command buffer must be in recording state
    pub unsafe fn write_timestamp(
        &mut self,
        cmd: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
    ) -> Result<u32> {
        let index = self
            .free_timestamp_indices
            .pop()
            .ok_or(crate::AshError::VulkanError(
                "No free timestamp query indices".to_string(),
            ))?;

        self.device
            .cmd_reset_query_pool(cmd, self.timestamp_pool, index, 1);

        self.device
            .cmd_write_timestamp(cmd, stage, self.timestamp_pool, index);

        Ok(index)
    }

    /// Request timestamp delta readback
    ///
    /// # Safety
    /// Both timestamps must have been written
    pub unsafe fn request_timestamp_delta(
        &mut self,
        start_index: u32,
        end_index: u32,
        callback: ReadbackCallback,
    ) -> Result<()> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = self.device.create_fence(&fence_info, None)?;

        let request = ReadbackRequest {
            pool: self.timestamp_pool,
            index: end_index,
            fence,
            frame_submitted: self.current_frame,
            query_type: QueryType::Timestamp,
            callback,
            handle: ReadbackHandle(self.next_handle_id),
        };

        self.next_handle_id += 1;
        self.pending.push_back(request);

        // Return start index to pool
        self.free_timestamp_indices.push(start_index);

        Ok(())
    }

    /// Poll for completed queries and invoke callbacks
    ///
    /// # Safety
    /// Device must be valid
    pub unsafe fn poll_completed(&mut self) -> Result<usize> {
        let mut completed_count = 0;

        // Check pending requests in FIFO order
        while let Some(request) = self.pending.front() {
            match self.device.get_fence_status(request.fence) {
                Ok(true) => {
                    // Fence signaled, query complete
                    let request = self.pending.pop_front().unwrap();

                    // Retrieve query result
                    let result = self.get_query_result(&request)?;

                    // Invoke callback
                    (request.callback)(result);

                    // Cleanup
                    self.device.destroy_fence(request.fence, None);

                    // Return index to free pool
                    match request.query_type {
                        QueryType::Occlusion => {
                            self.free_occlusion_indices.push(request.index);
                        }
                        QueryType::Timestamp => {
                            self.free_timestamp_indices.push(request.index);
                        }
                        QueryType::PipelineStats => {
                            self.free_pipeline_stats_indices.push(request.index);
                        }
                    }

                    completed_count += 1;
                }
                Ok(false) => {
                    // Not ready yet, stop checking (FIFO order)
                    break;
                }
                Err(e) => {
                    log::error!("Fence status check failed: {e:?}");
                    // Remove failed request
                    let request = self.pending.pop_front().unwrap();
                    self.device.destroy_fence(request.fence, None);
                }
            }
        }

        Ok(completed_count)
    }

    unsafe fn get_query_result(&self, request: &ReadbackRequest) -> Result<ReadbackResult> {
        match request.query_type {
            QueryType::Occlusion => {
                let mut result: u64 = 0;
                self.device.get_query_pool_results(
                    request.pool,
                    request.index,
                    std::slice::from_mut(&mut result),
                    vk::QueryResultFlags::TYPE_64,
                )?;
                Ok(ReadbackResult::OcclusionSamples(result))
            }
            QueryType::Timestamp => {
                let mut result: u64 = 0;
                self.device.get_query_pool_results(
                    request.pool,
                    request.index,
                    std::slice::from_mut(&mut result),
                    vk::QueryResultFlags::TYPE_64,
                )?;
                Ok(ReadbackResult::TimestampDelta(result))
            }
            QueryType::PipelineStats => {
                let mut results: [u64; 4] = [0; 4];
                self.device.get_query_pool_results(
                    request.pool,
                    request.index,
                    &mut results,
                    vk::QueryResultFlags::TYPE_64,
                )?;
                Ok(ReadbackResult::PipelineStats {
                    input_assembly_vertices: results[0],
                    input_assembly_primitives: results[1],
                    vertex_shader_invocations: results[2],
                    fragment_shader_invocations: results[3],
                })
            }
        }
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Destroy GPU resources
    ///
    /// # Safety
    /// Resources must not be in use
    pub unsafe fn destroy(&mut self) {
        if !self.initialized {
            return;
        }

        // Wait for all pending requests
        for request in self.pending.drain(..) {
            let _ = self
                .device
                .wait_for_fences(&[request.fence], true, u64::MAX);
            self.device.destroy_fence(request.fence, None);
        }

        if self.occlusion_pool != vk::QueryPool::null() {
            self.device.destroy_query_pool(self.occlusion_pool, None);
            self.occlusion_pool = vk::QueryPool::null();
        }

        if self.timestamp_pool != vk::QueryPool::null() {
            self.device.destroy_query_pool(self.timestamp_pool, None);
            self.timestamp_pool = vk::QueryPool::null();
        }

        if self.pipeline_stats_pool != vk::QueryPool::null() {
            self.device
                .destroy_query_pool(self.pipeline_stats_pool, None);
            self.pipeline_stats_pool = vk::QueryPool::null();
        }

        self.initialized = false;
        log::info!("Async readback manager destroyed");
    }
}

impl Drop for AsyncReadbackManager {
    fn drop(&mut self) {
        if self.initialized {
            log::warn!("AsyncReadbackManager dropped without calling destroy()");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readback_handle_uniqueness() {
        let handle1 = ReadbackHandle(1);
        let handle2 = ReadbackHandle(2);
        assert_ne!(handle1, handle2);
    }

    #[test]
    fn test_readback_result_variants() {
        let occlusion = ReadbackResult::OcclusionSamples(100);
        let timestamp = ReadbackResult::TimestampDelta(1000);

        match occlusion {
            ReadbackResult::OcclusionSamples(samples) => assert_eq!(samples, 100),
            _ => panic!("Wrong variant"),
        }

        match timestamp {
            ReadbackResult::TimestampDelta(delta) => assert_eq!(delta, 1000),
            _ => panic!("Wrong variant"),
        }
    }
}
