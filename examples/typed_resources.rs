//! Example demonstrating the new typed GPU resource wrappers.
//!
//! This example shows how to use `GpuBuffer<T>` and `GpuTexture` for type-safe
//! GPU resource management. New code should prefer these wrappers over raw
//! `BufferHandle` and `ImageHandle`.

use ash_renderer::prelude::*;
use ash_renderer::renderer::resources::{GpuBuffer, GpuTexture};
use glam::{Vec3, Vec4};

/// Example vertex type for demonstration
#[repr(C)]
#[derive(Clone, Copy)]
struct ExampleVertex {
    position: Vec3,
    color: Vec4,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Typed GPU Resources Example");
    println!("===========================\n");

    // This example demonstrates the API without actually running Vulkan
    // In real code, you would have a valid renderer instance

    println!("1. Type-Safe Buffers:");
    println!("   - GpuBuffer<Vec3> enforces vertex data type");
    println!("   - GpuBuffer<u32> enforces index data type");
    println!("   - Compiler prevents binding wrong buffer to shader\n");

    println!("2. Automatic Cleanup:");
    println!("   - RAII ensures buffers are destroyed when dropped");
    println!("   - No manual cleanup required\n");

    println!("3. Element Counting:");
    println!("   - Tracks both byte size and element count");
    println!("   - Prevents off-by-one errors\n");

    // Example API usage (would require valid Vulkan context):
    /*
    unsafe {
        // Create typed vertex buffer
        let vertex_buffer: GpuBuffer<ExampleVertex> = GpuBuffer::new(
            allocator.clone(),
            1024,  // 1024 vertices
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            Some("VertexBuffer".to_string()),
        )?;

        // Create typed index buffer
        let index_buffer: GpuBuffer<u32> = GpuBuffer::new(
            allocator.clone(),
            3072,  // 3072 indices (1024 triangles)
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk_mem::MemoryUsage::AutoPreferDevice,
            Some("IndexBuffer".to_string()),
        )?;

        // Create texture
        let texture = GpuTexture::new_2d(
            device.clone(),
            allocator.clone(),
            1024,
            1024,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            Some("AlbedoMap".to_string()),
        )?;

        // Compiler error: cannot bind vertex buffer to index binding
        // bind_index_buffer(&vertex_buffer);  // ERROR!

        // Automatic cleanup when buffers go out of scope
    }
    */

    println!("See src/renderer/resources/gpu_buffer.rs for implementation.");
    println!("See src/renderer/resources/gpu_texture.rs for implementation.");

    Ok(())
}
