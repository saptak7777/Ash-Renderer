# Contributing to ASH Renderer

Thank you for your interest in contributing! I welcome all contributions, from bug fixes and documentation improvements to new features.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/ash_renderer.git
    cd ash_renderer
    ```
3.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/amazing-feature
    ```

## Development Environment

-   **Rust**: Ensure you have the latest stable Rust toolchain installed via [rustup](https://rustup.rs/).
-   **Vulkan SDK**: You need the Vulkan SDK installed for validation layers and shader compilation tools (though mostly handled by `build.rs`).

## Coding Standards

I follow standard Rust community guidelines.

1.  **Formatting**: Run `cargo fmt` before committing.
2.  **Linting**: Ensure `cargo clippy` passes without warnings.
    ```bash
    cargo clippy --all-targets --all-features
    ```
3.  **Testing**: Run tests and examples to verify your changes.
    ```bash
    cargo test
    cargo run --example 02_cube
    ```

## Project Structure

-   `src/vulkan/`: Low-level Vulkan wrappers and abstractions.
-   `src/renderer/`: High-level rendering logic (ECS-agnostic).
-   `shaders/`: GLSL shader source files.
-   `examples/`: Sample applications demonstrating usage.

## Submitting Changes

1.  **Commit your changes** with clear, descriptive messages.
    -   Use imperative mood ("Add feature" not "Added feature").
2.  **Push to your fork**:
    ```bash
    git push origin feature/amazing-feature
    ```
3.  **Open a Pull Request (PR)**:
    -   Describe your changes in detail.
    -   Link to any relevant issues.
    -   If implementing a large feature, consider opening an issue first for discussion.

## License

By contributing, you agree that your contributions will be licensed under the project's [Apache 2.0 License](LICENSE).
