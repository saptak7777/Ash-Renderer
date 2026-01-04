# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Buffer Safety Architecture**:
  - Introduced `BufferBuilder` for type-safe, fluent buffer creation.
  - Added `BufferDescriptor` for explicit buffer configuration.
  - Implemented runtime validation in `Allocator` to catch common errors (e.g., mapping GPU-only memory, conflicting usage flags).
  - Added preset methods to `Allocator` for common buffer types (`create_joint_buffer`, `create_uniform_buffer`, etc.).
  - Added internal tracking registry for all buffer allocations with debug names and creation timestamps.
  - Added `allocator.print_buffer_stats()` for detailed memory usage logging.
- **GLB Material Pipeline**:
  - Implemented automatic extraction and registration of PBR material properties (metallic, roughness, emissive) from GLB files.
  - Added smart material deduplication to share material handles across identical meshes.
  - Added fallback logic to `submit_render_commands` to automatically use a mesh's registered material if none is strictly provided.
  - Added `SubmeshDescriptor` groundwork for future multi-material support.

### Changed
- Refactored `JointMatricesBuffer` to use the new `create_joint_buffer` preset, fixing potential `HOST_ACCESS` configuration issues.
- Updated `Allocator` to use modern VMA `MemoryUsage` variants (`AutoPreferDevice`, `AutoPreferHost`) instead of deprecated specific flags.
- Enhanced `register_mesh_handle` to populate `mesh_material_mapping` for automatic material selection.

### Fixed
- Fixed critical bug where GLB material properties were loaded but never registered, causing all imported models to render as default white.
- Fixed `clippy` warnings in `allocator.rs` related to uninlined format arguments and nested if-statements.
- Fixed potential safety risks in buffer mapping by enforcing explicit `HOST_ACCESS` flags during creation.
