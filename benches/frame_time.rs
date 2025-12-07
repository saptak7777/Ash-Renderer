//! Frame time benchmark.

use criterion::{criterion_group, criterion_main, Criterion};

fn frame_time_benchmark(_c: &mut Criterion) {
    // TODO: Implement frame time benchmarks
    // This requires headless Vulkan context for CI
}

criterion_group!(benches, frame_time_benchmark);
criterion_main!(benches);
