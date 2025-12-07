#version 450

// Shadow map fragment shader - depth-only, no color output
// The fragment shader can be empty for depth-only passes,
// but we include it for explicit control.

void main() {
    // Depth is written automatically by the rasterizer
    // No color output needed for shadow maps
}
