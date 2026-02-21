//! Virtual Shadow Maps (VSM) - AAA shadow rendering system
//!
//! Implements a virtual memory system for shadow maps, allowing for
//! extremely high resolution shadows (16k+) without the memory cost.
//!
//! # Architecture
//! - Physical Cache: Small texture atlas (4k x 4k) storing actual shadow data
//! - Page Table: Maps virtual coordinates to physical cache locations
//! - Request Buffer: GPU writes requested pages during analysis pass
//! - Allocation: CPU or GPU assigns physical memory to requested pages

pub mod buffer;
pub mod clipmap;
pub mod clipmap_manager;
pub mod compute_pipelines;
pub mod data;
pub mod manager;
pub mod page_manager;
pub mod resources;
pub mod shadow_pass;

pub use clipmap_manager::{ClipmapLevel, ClipmapManager};
pub use compute_pipelines::VsmComputePipelines;
pub use data::{VsmGlobalInfo, VsmLightData, VsmPageData};
pub use manager::{
    VsmManager, VsmShadowArgs, default_vsm_config, high_quality_vsm_config, performance_vsm_config,
};
pub use page_manager::PageManager;
pub use resources::{PageAllocation, VsmConfig, VsmResources};
pub use shadow_pass::VsmShadowPass;
