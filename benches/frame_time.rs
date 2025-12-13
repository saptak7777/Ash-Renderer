//! Renderer benchmarks.
//!
//! Includes both CPU-side math benchmarks and headless renderer benchmarks.
//! Headless benchmarks require a Vulkan driver (hardware or software e.g. Lavapipe).

use ash_renderer::{prelude::*, HeadlessSurfaceProvider};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::{Mat4, Vec3};

/// Benchmark CPU-side matrix calculations (view/projection setup)
fn matrix_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    // Benchmark view matrix creation
    group.bench_function("look_at_rh", |b| {
        b.iter(|| Mat4::look_at_rh(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y))
    });

    // Benchmark perspective projection
    group.bench_function("perspective_rh", |b| {
        b.iter(|| Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0))
    });

    // Benchmark combined view-projection multiplication
    group.bench_function("view_proj_multiply", |b| {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        b.iter(|| proj * view)
    });

    group.finish();
}

/// Benchmark frustum culling math (simulates light culling CPU overhead)
fn frustum_culling_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("frustum_culling");

    // Simulate sphere-frustum intersection test (CPU side)
    group.bench_function("sphere_frustum_test", |b| {
        let planes: [Vec3; 6] = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let sphere_center = Vec3::new(1.0, 2.0, 3.0);
        let sphere_radius = 1.0_f32;

        b.iter(|| {
            let mut inside = true;
            for plane in &planes {
                let dist = sphere_center.dot(*plane);
                if dist < -sphere_radius {
                    inside = false;
                    break;
                }
            }
            inside
        })
    });

    // Benchmark batch culling (100 objects)
    for count in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("batch_cull", count), &count, |b, &n| {
            let planes: [Vec3; 6] = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, -1.0),
            ];
            let objects: Vec<(Vec3, f32)> = (0..n)
                .map(|i| (Vec3::new(i as f32, 0.0, 0.0), 1.0))
                .collect();

            b.iter(|| {
                let mut visible = 0;
                for (center, radius) in &objects {
                    let mut inside = true;
                    for plane in &planes {
                        if center.dot(*plane) < -radius {
                            inside = false;
                            break;
                        }
                    }
                    if inside {
                        visible += 1;
                    }
                }
                visible
            })
        });
    }

    group.finish();
}

/// Benchmark shadow cascade split calculation
fn shadow_cascade_splits(c: &mut Criterion) {
    c.bench_function("cascade_split_calculation", |b| {
        let near = 0.1_f32;
        let far = 100.0_f32;
        let cascade_count = 4;
        let lambda = 0.5_f32; // Practical split scheme blend factor

        b.iter(|| {
            let mut splits = Vec::with_capacity(cascade_count + 1);
            splits.push(near);

            for i in 1..=cascade_count {
                let p = i as f32 / cascade_count as f32;
                let log_split = near * (far / near).powf(p);
                let uniform_split = near + (far - near) * p;
                let split = lambda * log_split + (1.0 - lambda) * uniform_split;
                splits.push(split);
            }
            splits
        })
    });
}

/// Headless renderer benchmark (requires Vulkan driver)
fn headless_render_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("renderer_headless");

    let surface_provider = HeadlessSurfaceProvider::new(800, 600);

    match Renderer::new(&surface_provider) {
        Ok(mut renderer) => {
            let cube = Mesh::create_cube();
            renderer.set_mesh(cube);

            let view = Mat4::look_at_rh(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
            let mut proj = Mat4::perspective_rh(45.0_f32.to_radians(), 800.0 / 600.0, 0.1, 100.0);
            proj.y_axis.y *= -1.0;
            let camera_pos = Vec3::new(0.0, 2.0, 5.0);

            group.bench_function("render_frame", |b| {
                b.iter(|| {
                    renderer
                        .render_frame(view, proj, camera_pos)
                        .expect("Render frame failed during benchmark");
                })
            });
        }
        Err(e) => {
            eprintln!("Failed to initialize headless renderer for benchmark: {e}");
            eprintln!("Skipping headless benchmarks.");
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    matrix_calculations,
    frustum_culling_math,
    shadow_cascade_splits,
    headless_render_loop
);
criterion_main!(benches);
